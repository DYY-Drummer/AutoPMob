"""同じ式を 1 本と数える（延べなし）Recall@20・MAP@20：LLM の答えと提案手法の上位 20 件を、同じ規則で採点する.

背景（2026-11-09、本人の指示「重複した同じ式を延べなしで recall を計算してください」）：データベースには同じ式が別の ID で
入っている（表記が同じ式の組 506、式 1,130 本。同じ本の別のファイル、同じ論文の別のファイル、同じファイルの中の再掲）。
本人はこれをデータベースの正常な現象とみなす。ID で判定すると、正解と別の ID の同じ式を選んだ答えは不正解になるので、
式の同一性を表記（set_aware_reranker.latex_key：空白と \\left・\\right・\\,・\\;・\\!・\\quad を除いた LaTeX）で決め、
正解集合でも答えでも同じ式は 1 本と数える。

規則:
  - 正解集合の中の同じ式は 1 本にする（K はその本数。設定 A では 10 件で K が 1 減る）。
  - 答えを順に見て、同じ式の 2 本目以降は数えない（枝を詰める）。DB にない ID は不正解のまま順位を占める。
  - まだ数えていない正解の式と同じ式が出たら、その順位を正解の順位とする。上位 20 件の外は見つからなかったものとする。
既存の *_equiv（results_*__v4.json の summary_equiv：別の出典の同じ式だけを正解とみなし、K はそのまま）との違いは、
正解集合の 1 本化、同じ出典の中の同じ式、答えの枝詰めの 3 点。

出力: experiments/llm_db_baseline/recall_dedup.json。使い方: python3 llm_recall_dedup.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_llm_db_baseline as rl  # noqa: E402

OUT_JSON = rl.OUT_DIR / "recall_dedup.json"
TOP20_DIR = rl.OUT_DIR / "proposed_top20"
LLM_RESULTS = {"Opus 5.5": "results_claude-opus-5-5__eqd__ex-none__high__v4.json",
               "Haiku 5.5": "results_claude-haiku-5-5__eqd__ex-none__high__v4.json"}


def dedup_ranks(ranked_keys, correct_keys, identity: dict, miss_rank: int = rl.MISS_RANK) -> tuple:
    """同じ式を 1 本と数えた、正解の式の順位（昇順。並べた中になければ miss_rank）と、正解の式の本数.

    identity[式の ID] = 同じ式の印（表記）。印がない ID は、その ID 自身を印とする。
    """
    def ident(k):
        return identity.get(k) or k
    correct_ids = list(dict.fromkeys(ident(k) for k in correct_keys))
    open_gt, seen, got, r = set(correct_ids), set(), {}, 0
    for k in ranked_keys:
        if k is None:                      # DB にない ID：不正解のまま順位を占める
            r += 1
            continue
        i = ident(k)
        if i in seen:                      # 同じ式の 2 本目以降は数えない
            continue
        seen.add(i)
        r += 1
        if i in open_gt:
            got[i] = r
            open_gt.remove(i)
    return sorted(got.get(i, miss_rank) for i in correct_ids), len(correct_ids)


def dedup_metrics(ranked_keys, correct_keys, identity: dict) -> dict:
    ranks, k = dedup_ranks(ranked_keys, correct_keys, identity)
    return {**rl.metrics_from_ranks(ranks), "K": k, "ranks": ranks}


def parse_weight_label(label: str) -> tuple:
    """baseline の重みの名前（kato1020_baselines.wlabel の逆）→（文章類似度の重み, 変数の重複率の重み）."""
    a, b = label.lstrip("w").split("-")
    return int(a) / 100, int(b) / 100


def baseline_top20(setting: str, seeds, identity: dict) -> dict:
    """baseline（TF-IDF の文章類似度と変数の重複率の重みつき和。重みは分割ごとに検証用のケースで選んだもの）の
    上位 20 件を、kato1020_baselines.py と同じ計算で作り直し、ID 判定の順位が保存した per-case と一致することを確かめる.

    返り値: {種: {case_id: {"strict": metrics, "dedup": metrics}}}
    """
    from external_baselines import jaccard_matrix, load_setting, text_scores
    from two_stage_query_conditioned import io_vars
    fixed = json.loads(rl.SPLITS_JSON.read_text(encoding="utf-8"))
    ex = set(fixed["provenance"]["excluded_case_ids"])
    data = load_setting(setting)
    keep = [i for i, c in enumerate(data["cases"]) if c.get("case_id") not in ex]
    cases = [data["cases"][i] for i in keep]
    cl = [data["correct_lists"][i] for i in keep]
    ek = data["eq_keys"]
    T = text_scores("tfidf", data["eq_texts"], data["cases"], None)[keep]
    J = jaccard_matrix([io_vars(c) for c in cases], data["eq_vars"])
    idx = {c["case_id"]: i for i, c in enumerate(cases)}
    weights = json.loads((rl.REF_RUN / "main.json").read_text(encoding="utf-8"))["selected_weights"][setting]["tfidf"]
    saved = json.loads((rl.REF_RUN / "raw" / f"baselines_percase_{setting}.json").read_text(encoding="utf-8"))
    saved = {(r["seed"], r["case_id"]): r["ranks"] for r in saved["results"]["baseline"]["per_case"]}
    out = {}
    for sd in seeds:
        wt, wv = parse_weight_label(weights[str(sd)])
        per = {}
        for cid in fixed["splits"][setting][str(sd)]["test"]:
            ci = idx[cid]
            order = np.argsort(-(wt * T[ci] + wv * J[ci]))
            pos = np.empty(len(order), dtype=np.int64)
            pos[order] = np.arange(1, len(order) + 1)
            ranks = sorted(int(pos[j]) for j in cl[ci])
            if ranks != saved.get((sd, cid)):
                sys.exit(f"baseline {setting} seed {sd} {cid}: 作り直した順位が保存した per-case と食い違う")
            top = [ek[j] for j in order[:rl.TOP_N]]
            per[cid] = {"strict": rl.metrics_from_ranks(ranks),
                        "dedup": dedup_metrics(top, [ek[j] for j in cl[ci]], identity)}
        out[str(sd)] = per
    return out


def case_kind(c: dict) -> str:
    v = c.get("variant_type", "")
    return "single" if v == "original" else "cross" if v.startswith("multisource_") else "synth" if v.startswith("dae_") else v


def main() -> None:
    from two_stage_query_conditioned import load_equations, eq_key, norm
    from set_aware_reranker import latex_key
    eqs = load_equations()
    identity = {eq_key(e): latex_key(e.get("equation")) for e in eqs if eq_key(e)}
    cases, splits = rl.load_case_index(), rl.load_splits()

    def correct_of(cid):
        import ast
        v = cases[cid].get("correct_model_ids") or []
        v = ast.literal_eval(v) if isinstance(v, str) else v
        return [norm(x) for x in v if norm(x) in identity]

    n_dup_correct = {s: sum(1 for cid in {c for p in splits[s]["42"].values() for c in p}
                            if len({identity[k] or k for k in correct_of(cid)}) < len(correct_of(cid)))
                     for s in ("A", "B")}
    out = {"rule": __doc__.split("規則:")[1].split("出力:")[0].strip(),
           "n_cases_whose_correct_set_has_a_duplicate": n_dup_correct, "models": {}, "proposed": {}}
    seeds = rl.ALL_SEEDS
    KINDS = ("single", "cross", "synth")

    def by_kind(per_seed_cases: dict) -> dict:
        """{種: {case_id: {"recall@20", "AP@20"}}} → 種類ごとの、分割のテストケースの平均の、分割間の平均 ± SD（設定 A 用）."""
        res = {}
        for kind in KINDS:
            vals = {sd: [m for cid, m in per.items() if case_kind(cases[cid]) == kind] for sd, per in per_seed_cases.items()}
            vals = {sd: v for sd, v in vals.items() if v}
            if vals:
                res[kind] = {"n_per_split": {sd: len(v) for sd, v in vals.items()},
                             "recall@20": rl._mean_sd([float(np.mean([m["recall@20"] for m in v])) for v in vals.values()]),
                             "MAP@20": rl._mean_sd([float(np.mean([m["AP@20"] for m in v])) for v in vals.values()])}
        return res

    # ---- 提案手法：保存した上位 20 件（dump_proposed_top20.py。既存の順位と一致を確認済み）
    prop = {}
    for s in ("A", "B"):
        per_seed, per_seed_cases = {}, {}
        for sd in seeds:
            d = json.loads((TOP20_DIR / f"{s}__{sd}.json").read_text(encoding="utf-8"))
            ms = {cid: dedup_metrics(v["top"], correct_of(cid), identity) for cid, v in d["cases"].items()}
            strict = {cid: rl.metrics_from_ranks(v["ranks"]) for cid, v in d["cases"].items()}
            per_seed_cases[str(sd)] = ms
            per_seed[str(sd)] = {"n": len(ms),
                                 "recall@20": float(np.mean([m["recall@20"] for m in ms.values()])),
                                 "MAP@20": float(np.mean([m["AP@20"] for m in ms.values()])),
                                 "recall@20_strict": float(np.mean([m["recall@20"] for m in strict.values()])),
                                 "MAP@20_strict": float(np.mean([m["AP@20"] for m in strict.values()]))}
        prop[s] = {"per_seed": per_seed, "by_kind": by_kind(per_seed_cases),
                   **{k: rl._mean_sd([v[k] for v in per_seed.values()])
                      for k in ("recall@20", "MAP@20", "recall@20_strict", "MAP@20_strict")}}
    out["proposed"] = prop

    # ---- baseline：同じ計算で上位 20 件を作り直す（ID 判定の順位は保存した per-case と一致を確認）
    base = {}
    for s in ("A", "B"):
        per_seed, per_seed_cases = {}, {}
        for sd, per in baseline_top20(s, seeds, identity).items():
            per_seed_cases[sd] = {cid: m["dedup"] for cid, m in per.items()}
            per_seed[sd] = {"n": len(per),
                            "recall@20": float(np.mean([m["dedup"]["recall@20"] for m in per.values()])),
                            "MAP@20": float(np.mean([m["dedup"]["AP@20"] for m in per.values()])),
                            "recall@20_strict": float(np.mean([m["strict"]["recall@20"] for m in per.values()])),
                            "MAP@20_strict": float(np.mean([m["strict"]["AP@20"] for m in per.values()]))}
        base[s] = {"weights": json.loads((rl.REF_RUN / "main.json").read_text(encoding="utf-8"))["selected_weights"][s]["tfidf"],
                   "per_seed": per_seed, "by_kind": by_kind(per_seed_cases),
                   **{k: rl._mean_sd([v[k] for v in per_seed.values()])
                      for k in ("recall@20", "MAP@20", "recall@20_strict", "MAP@20_strict")}}
    out["baseline"] = base

    # ---- LLM：ケースごとに 3 回の平均、分割のテストケースの平均、分割間の平均 ± SD
    for name, f in LLM_RESULTS.items():
        recs = json.loads((rl.OUT_DIR / f).read_text(encoding="utf-8"))["per_case"]
        by = defaultdict(list)
        for r in recs:
            m = dedup_metrics(r["ranked_keys"], correct_of(r["case_id"]), identity)
            by[r["case_id"]].append({"recall@20": m["recall@20"], "AP@20": m["AP@20"],
                                     "recall@20_strict": r["recall@20"], "AP@20_strict": r["AP@20"],
                                     "recall@20_equiv_old": r.get("recall@20_equiv"), "AP@20_equiv_old": r.get("AP@20_equiv"),
                                     "gain": m["recall@20"] - r["recall@20"]})
        per_case = {c: {k: float(np.mean([x[k] for x in xs])) for k in xs[0] if xs[0][k] is not None} for c, xs in by.items()}
        sds = [float(np.std([x["recall@20"] for x in xs], ddof=1)) for xs in by.values() if len(xs) > 1]
        entry = {"n_cases": len(per_case), "n_answers": len(recs),
                 "answers_where_dedup_raised_recall": sum(1 for xs in by.values() for x in xs if x["gain"] > 1e-12),
                 "answers_where_dedup_lowered_recall": sum(1 for xs in by.values() for x in xs if x["gain"] < -1e-12),
                 # 回の間のばらつき（延べなしの Recall@20）：ケースごとの 3 回の標準偏差（ddof=1）の平均と、3 回とも同じ値だったケースの割合
                 "runs_recall@20_sd_mean": float(np.mean(sds)) if sds else None,
                 "cases_all_runs_same_recall@20": float(np.mean([len({round(x["recall@20"], 9) for x in xs}) == 1
                                                                 for xs in by.values()]))}
        for s in ("A", "B"):
            e = {"dedup": rl.split_summary(per_case, splits, s, seeds),
                 "strict": rl.split_summary(per_case, splits, s, seeds, suffix="_strict"),
                 "by_kind": by_kind({str(sd): {c: per_case[c] for c in splits[s][str(sd)]["test"]} for sd in seeds})}
            if all("recall@20_equiv_old" in v for v in per_case.values()):
                e["equiv_old"] = rl.split_summary(per_case, splits, s, seeds, suffix="_equiv_old")
            e["vs_proposed_dedup"] = {
                "recall@20": rl.compare_splits({k: v["recall@20"] for k, v in e["dedup"]["per_seed"].items()},
                                               {k: v["recall@20"] for k, v in prop[s]["per_seed"].items()}),
                "MAP@20": rl.compare_splits({k: v["MAP@20"] for k, v in e["dedup"]["per_seed"].items()},
                                            {k: v["MAP@20"] for k, v in prop[s]["per_seed"].items()})}
            entry[s] = e
        out["models"][name] = entry

    out["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"cases whose correct set contains the same equation twice: {n_dup_correct}")
    for s in ("A", "B"):
        print(f"\n### setting {s} (10 splits, mean ± SD)          strict (ID)        same eq. elsewhere   dedup (same eq. = 1)")
        for name, p in (("baseline", base[s]), ("proposed", prop[s])):
            print(f"  {name:14s} Recall@20  {p['recall@20_strict']['mean']:.3f} ± {p['recall@20_strict']['sd']:.3f}"
                  f"      {'—':^14s}      {p['recall@20']['mean']:.3f} ± {p['recall@20']['sd']:.3f}")
            print(f"  {'':14s} MAP@20     {p['MAP@20_strict']['mean']:.3f} ± {p['MAP@20_strict']['sd']:.3f}"
                  f"      {'—':^14s}      {p['MAP@20']['mean']:.3f} ± {p['MAP@20']['sd']:.3f}")
        for name, entry in out["models"].items():
            e = entry[s]
            for key, lab in (("recall@20", "Recall@20"), ("MAP@20", "MAP@20   ")):
                old = e.get("equiv_old", {}).get(key)
                c = e["vs_proposed_dedup"][key]
                print(f"  {name if key == 'recall@20' else '':14s} {lab}  {e['strict'][key]['mean']:.3f} ± {e['strict'][key]['sd']:.3f}"
                      f"      {old['mean']:.3f} ± {old['sd']:.3f}" if old else f"      {'—':^14s}", end="")
                print(f"      {e['dedup'][key]['mean']:.3f} ± {e['dedup'][key]['sd']:.3f}"
                      f"   (LLM higher than proposed in {c['wins']}/{c['n']} splits, diff {c['mean_diff']:+.3f})")
    print("\n### setting A by case kind (dedup; mean over splits of the test-case mean)")
    for kind in KINDS:
        row = []
        for name, p in (("baseline", base["A"]), ("proposed", prop["A"]), *((n, e["A"]) for n, e in out["models"].items())):
            k = p["by_kind"].get(kind)
            row.append(f"{name} R {k['recall@20']['mean']:.3f} MAP {k['MAP@20']['mean']:.3f}" if k else f"{name} —")
        print(f"  {kind:6s} " + " | ".join(row))
    for name, entry in out["models"].items():
        print(f"\n{name}: answers {entry['n_answers']}, dedup raised Recall@20 in {entry['answers_where_dedup_raised_recall']}, "
              f"lowered in {entry['answers_where_dedup_lowered_recall']}; runs: Recall@20 SD mean {entry['runs_recall@20_sd_mean']:.3f}, "
              f"all runs same Recall@20 in {entry['cases_all_runs_same_recall@20']:.3f} of cases")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
