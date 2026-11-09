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

    # ---- 提案手法：保存した上位 20 件（dump_proposed_top20.py。既存の順位と一致を確認済み）
    prop = {}
    for s in ("A", "B"):
        per_seed = {}
        for sd in seeds:
            d = json.loads((TOP20_DIR / f"{s}__{sd}.json").read_text(encoding="utf-8"))
            ms = {cid: dedup_metrics(v["top"], correct_of(cid), identity) for cid, v in d["cases"].items()}
            strict = {cid: rl.metrics_from_ranks(v["ranks"]) for cid, v in d["cases"].items()}
            per_seed[str(sd)] = {"n": len(ms),
                                 "recall@20": float(np.mean([m["recall@20"] for m in ms.values()])),
                                 "MAP@20": float(np.mean([m["AP@20"] for m in ms.values()])),
                                 "recall@20_strict": float(np.mean([m["recall@20"] for m in strict.values()])),
                                 "MAP@20_strict": float(np.mean([m["AP@20"] for m in strict.values()]))}
        prop[s] = {"per_seed": per_seed,
                   **{k: rl._mean_sd([v[k] for v in per_seed.values()])
                      for k in ("recall@20", "MAP@20", "recall@20_strict", "MAP@20_strict")}}
    out["proposed"] = prop

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
        entry = {"n_cases": len(per_case), "n_answers": len(recs),
                 "answers_where_dedup_raised_recall": sum(1 for xs in by.values() for x in xs if x["gain"] > 1e-12),
                 "answers_where_dedup_lowered_recall": sum(1 for xs in by.values() for x in xs if x["gain"] < -1e-12)}
        for s in ("A", "B"):
            e = {"dedup": rl.split_summary(per_case, splits, s, seeds),
                 "strict": rl.split_summary(per_case, splits, s, seeds, suffix="_strict")}
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
        p = prop[s]
        print(f"  {'proposed':14s} Recall@20  {p['recall@20_strict']['mean']:.3f} ± {p['recall@20_strict']['sd']:.3f}"
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
    for name, entry in out["models"].items():
        print(f"\n{name}: answers {entry['n_answers']}, dedup raised Recall@20 in {entry['answers_where_dedup_raised_recall']}, "
              f"lowered in {entry['answers_where_dedup_lowered_recall']}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
