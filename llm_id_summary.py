"""LLM との比較を、式の ID で判定した値（同義性の判定なし）でまとめる：報告書と修論の数値の出典.

判定：手法が答えた式が、正解集合の式のどれかと同じ ID を持つなら正解、そうでなければ不正解。DB にない ID は不正解で
その順位を占める。同じ ID の 2 回目以降は除く（run_llm_db_baseline.judge_ids）。上位 20 件。
2026-11-16 本人の決定：報告書と修論は、この判定の値だけを載せる（llm_recall_dedup.py の値は使わない）。

入力:
  - LLM: experiments/llm_db_baseline/results_claude-{opus,haiku}-5-5__eqd__ex-none__high__v4.json の per_case（ケース × 3 回）
  - 提案手法: experiments/kato1021/raw/select/{設定}_{w}_{neg}__{種}.json（検証で選んだ設定。reranker-10S）
  - baseline: experiments/kato1021/raw/baselines_percase_{設定}.json（重みは分割ごとに検証で選んだもの）
集計: ケースの値（LLM は 3 回の平均）→ 分割のテストケースの平均 → 10 通りの分割の平均 ± 標準偏差（ddof=1）。
提案手法と baseline の値が experiments/kato1021/main.json と、LLM の値が results_*__v4.json の summary と
一致することを確かめ、食い違えば止める。

各設定の by_overlap は、同じ値を、テスト用のケースのうち訓練用のケースと正解式を共有するもの・しないものに分けて
平均したもの（分割、学習済みのモデル、LLM の答えは元のまま。分け方は overlap_groups）。

出力: experiments/llm_db_baseline/id_summary.json。使い方: python3 llm_id_summary.py
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
import analyze_split_overlap as so  # noqa: E402
import run_llm_db_baseline as rl  # noqa: E402

OUT_JSON = rl.OUT_DIR / "id_summary.json"
LLM_RESULTS = {"Opus 5.5": "results_claude-opus-5-5__eqd__ex-none__high__v4.json",
               "Haiku 5.5": "results_claude-haiku-5-5__eqd__ex-none__high__v4.json"}
KINDS = ("single", "cross", "synth")
KEYS = ("recall@20", "AP@20")
MAIN_KEYS = {"proposed": "proposed_static", "baseline": "baseline_tfidf"}
# テスト用のケースを、訓練用のケースとの正解式の共有で分けた組（overlap_groups）
OVERLAP_GROUPS = ("shared_equation", "no_shared_equation", "no_shared_equation_train_val", "no_shared_source")


def split_means(per_seed_cases: dict, key: str, kind_of: dict | None = None, kind: str | None = None) -> dict:
    """{種: {case_id: 指標}} → {種: その分割のテストケースの key の平均}。kind を与えると、その種類のケースだけで平均する
    （その種類のケースがない分割は除く）."""
    out = {}
    for sd, per in per_seed_cases.items():
        vals = [m[key] for c, m in per.items() if kind is None or kind_of[c] == kind]
        if vals:
            out[sd] = float(np.mean(vals))
    return out


def subset_means(per_seed_cases: dict, key: str, subset: dict) -> dict:
    """{種: {case_id: 指標}} と {種: ケースの集合} → {種: その集合のケースの key の平均}。集合が空の分割は除く."""
    out = {}
    for sd, per in per_seed_cases.items():
        vals = [per[c][key] for c in subset.get(sd, ())]
        if vals:
            out[sd] = float(np.mean(vals))
    return out


def overlap_groups(split: dict, gt_of: dict) -> dict:
    """1 つの分割のテスト用のケースを、訓練用のケースとの正解式の共有で分ける（式は ID で照合する）.

    shared_equation：正解式を 1 本以上、訓練用のケースの正解集合と共有する。no_shared_equation：1 本も共有しない。
    no_shared_equation_train_val：訓練用と検証用のどのケースとも共有しない。
    no_shared_source：正解式の出典のどれも、訓練用のケースの正解式の出典でない（no_shared_equation の一部）."""
    test = set(split["test"])
    tr = so.shared_with_training(split, gt_of)
    tv = so.shared_with_training({**split, "train": list(split["train"]) + list(split["val"])}, gt_of)
    return {"shared_equation": set(tr["cases_shared_equation"]),
            "no_shared_equation": test - set(tr["cases_shared_equation"]),
            "no_shared_equation_train_val": test - set(tv["cases_shared_equation"]),
            "no_shared_source": test - set(tr["cases_shared_source"])}


def by_overlap(psc: dict, splits_s: dict, gt_of: dict, kind_of: dict) -> dict:
    """各手法を、overlap_groups の組ごとに採点する：{組: {n_test, n_test_by_kind, methods, compare}}.

    分割も学習済みのモデルも LLM の答えも元のままで、平均を取るテスト用のケースだけを変える。methods は
    {手法: {指標: 10 通りの平均 ± 標準偏差, by_kind}}（by_kind は、その種類のケースがある分割が 2 つ以上のときだけ）。
    compare は「提案手法 − 他の手法」の差の平均と、提案手法が上回った分割の数."""
    groups = {sd: overlap_groups(d, gt_of) for sd, d in splits_s.items()}
    out = {}
    for g in OVERLAP_GROUPS:
        sub = {sd: groups[sd][g] for sd in groups}
        methods = {}
        for name, per in psc.items():
            m = {key: mean_sd(subset_means(per, key, sub)) for key in KEYS}
            m["by_kind"] = {}
            for k in KINDS:
                subk = {sd: {c for c in cs if kind_of[c] == k} for sd, cs in sub.items()}
                vals = {key: subset_means(per, key, subk) for key in KEYS}
                if len(vals[KEYS[0]]) >= 2:
                    m["by_kind"][k] = {key: mean_sd(v) for key, v in vals.items()}
            methods[name] = m
        out[g] = {"n_test": {sd: len(cs) for sd, cs in sub.items()},
                  "n_test_by_kind": {k: sum(1 for cs in sub.values() for c in cs if kind_of[c] == k) for k in KINDS},
                  "methods": methods,
                  "compare": {f"proposed vs {name}": {key: rl.compare_splits(methods["proposed"][key]["per_seed"],
                                                                             methods[name][key]["per_seed"])
                                                      for key in KEYS}
                              for name in psc if name != "proposed"}}
    return out


def runs_agreement(runs: dict) -> dict:
    """{case_id: [回ごとの Recall@20]} → 回の間の標準偏差（ddof=1）のケース平均と、全回で同じ値だったケースの割合."""
    multi = {c: v for c, v in runs.items() if len(v) > 1}
    return {"n_cases": len(multi),
            "recall@20_sd_mean": float(np.mean([np.std(v, ddof=1) for v in multi.values()])),
            "cases_all_runs_same": float(np.mean([len({round(x, 9) for x in v}) == 1 for v in multi.values()]))}


def answer_format(rows: list, top_n: int = 20) -> dict:
    """呼び出しごとの記録（per_case）から、答えの形を数える。指示は「top_n 個の ID を、どれも 1 回ずつ、データベースにあるものだけ」.

    exactly / more / fewer は、重複を除いた ID の数で分けた 3 つで、足すと n になる（run_llm_db_baseline の
    quality.answers_with_20 は「top_n 個以上」で、top_n 個を超えた答えも含むので、ここでは使わない）.
    as_instructed は、返した ID がちょうど top_n 個で、重複もデータベースにない ID もない答え."""
    def distinct(r):  # 重複を除いた ID の数（ranked_ids は top_n で打ち切ってあるので、切った分を足す）
        return len(r["ranked_ids"]) + r["n_truncated"]
    return {
        "n": len(rows),
        "as_instructed": sum(1 for r in rows if r["n_raw"] == top_n and r["n_duplicates"] == 0 and r["n_invalid"] == 0),
        "exactly_20_distinct": sum(1 for r in rows if distinct(r) == top_n),
        "more_than_20_distinct": sum(1 for r in rows if distinct(r) > top_n),
        "fewer_than_20_distinct": sum(1 for r in rows if distinct(r) < top_n),
        "empty": sum(1 for r in rows if not r["ranked_ids"]),
        "with_repeats": sum(1 for r in rows if r["n_duplicates"] > 0),
        "with_absent_id": sum(1 for r in rows if r["n_invalid"] > 0),
        "refusals": sum(1 for r in rows if r["stop_reason"] == "refusal"),
    }


def cost_breakdown(usages: list, model: str) -> dict:
    """API の呼び出しの費用（ドル）と、そのうちキャッシュしたデータベースの読み出しの分.

    usages は呼び出しごとの usage。料金は run_llm_db_baseline.cost_usd と price_for（プロンプトの長さで決まる）."""
    total = read = 0.0
    for u in usages:
        total += rl.cost_usd(u, model)
        n_prompt = sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        read += (u.get("cache_read_input_tokens") or 0) * rl.price_for(model, n_prompt)["read"] / 1e6
    return {"n_calls": len(usages), "total": total, "read": read, "read_share": read / total if total else 0.0}


def time_range(values) -> dict:
    """秒数の最小と最大（記録のない None は除く）."""
    v = [x for x in values if x is not None]
    return {"min": min(v), "max": max(v), "n": len(v)}


def case_kind(c: dict) -> str:
    v = c.get("variant_type", "")
    return "single" if v == "original" else "cross" if v.startswith("multisource_") else "synth" if v.startswith("dae_") else v


def mean_sd(per_seed: dict) -> dict:
    a = np.asarray(list(per_seed.values()), dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "n": int(a.size), "per_seed": per_seed}


def main() -> None:
    cases, splits = rl.load_case_index(), rl.load_splits()
    kind_of = {c: case_kind(v) for c, v in cases.items()}
    seeds = [str(s) for s in rl.ALL_SEEDS]
    main_json = json.loads((rl.REF_RUN / "main.json").read_text(encoding="utf-8"))
    llm_raw = {n: json.loads((rl.OUT_DIR / f).read_text(encoding="utf-8")) for n, f in LLM_RESULTS.items()}
    out = {"rule": __doc__.split("判定：")[1].split("入力:")[0].strip(), "settings": {}, "models": {}}

    for name, res in llm_raw.items():
        by = defaultdict(list)
        for r in res["per_case"]:
            by[r["case_id"]].append(r["recall@20"])
        q = res["quality"]
        raw = rl.OUT_DIR / "raw" / LLM_RESULTS[name].replace("results_", "").replace(".json", ".jsonl")
        usages = [json.loads(line)["usage"] for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
        cost = cost_breakdown(usages, res["config"]["model"])
        if abs(cost["total"] - q["cost_usd_all_calls"]) > 0.01:
            sys.exit(f"{name}: 費用の合計が results の quality.cost_usd_all_calls と食い違う")
        out["models"][name] = {**runs_agreement(by),
                               "jaccard_mean": res["variability"]["jaccard_mean"],
                               "same_list_share": res["variability"]["same_list_share"],
                               "n_answers": res["n_scored"],
                               "format": answer_format(res["per_case"]),
                               "output_tokens_mean": q["output_tokens_mean"],
                               "cost": cost}

    for s in ("A", "B"):
        psc = {}
        # 提案手法
        psc["proposed"] = {}
        t_train, t_eval = [], []
        for sd in seeds:
            sel = rl._selected_ref(s, int(sd))
            f = rl.REF_RUN / "raw" / "select" / f"{s}_{sel['w']}_{sel['neg']}__{sd}.json"
            res10 = json.loads(f.read_text(encoding="utf-8"))["results"]["reranker-10S"]
            recs = rl.split_test_records(res10["per_case"], int(sd))
            psc["proposed"][sd] = {r["case_id"]: rl.metrics_from_ranks(r["ranks"]) for r in recs}
            t_train.append(res10["per_seed"][0].get("time_train_sec"))
            t_eval.append(res10["per_seed"][0].get("time_eval_sec"))
        # baseline
        psc["baseline"] = defaultdict(dict)
        pc = json.loads((rl.REF_RUN / "raw" / f"baselines_percase_{s}.json").read_text(encoding="utf-8"))
        for r in pc["results"]["baseline"]["per_case"]:
            psc["baseline"][str(r["seed"])][r["case_id"]] = rl.metrics_from_ranks(r["ranks"])
        psc["baseline"] = dict(psc["baseline"])
        # LLM（ケースの値は 3 回の平均）
        for name, res in llm_raw.items():
            by = defaultdict(list)
            for r in res["per_case"]:
                by[r["case_id"]].append(r)
            mean = {c: {"recall@20": float(np.mean([x["recall@20"] for x in xs])),
                        "AP@20": float(np.mean([x["AP@20"] for x in xs]))} for c, xs in by.items()}
            psc[name] = {sd: {c: mean[c] for c in splits[s][sd]["test"]} for sd in seeds}
        # テストケースの並びがそろっているか
        for name, per in psc.items():
            for sd in seeds:
                if set(per[sd]) != set(splits[s][sd]["test"]):
                    sys.exit(f"{name} {s} {sd}: テストケースが splits.json と食い違う")

        st = {"n_test_by_kind": {k: {sd: sum(1 for c in splits[s][sd]["test"] if kind_of[c] == k) for sd in seeds}
                                 for k in KINDS},
              # 提案手法（reranker-10S）の、1 つの分割の訓練とテストケースの採点にかかった秒数（結果を出した実行の記録）
              "proposed_time_sec": {"train": time_range(t_train), "eval": time_range(t_eval)},
              "methods": {}, "compare": {}}
        for name, per in psc.items():
            m = {key: mean_sd(split_means(per, key)) for key in ("recall@20", "AP@20")}
            m["by_kind"] = {k: {key: mean_sd(v) for key in ("recall@20", "AP@20")
                                if (v := split_means(per, key, kind_of, k))} for k in KINDS}
            st["methods"][name] = m
        # 既存の値との照合
        for name, mk in MAIN_KEYS.items():
            for key, mkey in (("recall@20", "Recall@20"), ("AP@20", "MAP@20")):
                ref = main_json["table"][s][mk][mkey]["per_seed"]
                got = st["methods"][name][key]["per_seed"]
                if any(abs(got[sd] - ref[sd]) > 1e-9 for sd in seeds):
                    sys.exit(f"{name} {s} {key}: kato1021/main.json の値と食い違う")
        for name, res in llm_raw.items():
            ref = res["summary"][f"{s}_10splits"]["per_seed"]
            for key, rkey in (("recall@20", "recall@20"), ("AP@20", "MAP@20")):
                got = st["methods"][name][key]["per_seed"]
                if any(abs(got[sd] - ref[sd][rkey]) > 1e-9 for sd in seeds):
                    sys.exit(f"{name} {s} {key}: results の summary と食い違う")
        # 比較（a − b、a が上回った分割の数）
        for a in LLM_RESULTS:
            for b in ("proposed", "baseline"):
                st["compare"][f"{a} vs {b}"] = {
                    key: rl.compare_splits(st["methods"][a][key]["per_seed"], st["methods"][b][key]["per_seed"])
                    for key in ("recall@20", "AP@20")}
        st["compare"]["proposed vs baseline"] = {
            key: rl.compare_splits(st["methods"]["proposed"][key]["per_seed"], st["methods"]["baseline"][key]["per_seed"])
            for key in ("recall@20", "AP@20")}
        # 訓練用のケースと正解式を共有するテスト用のケース、共有しないテスト用のケースに分けた値
        gt_of = {c: so.ground_truth_ids(cases[c]) for d in splits[s].values() for part in ("train", "val", "test")
                 for c in d[part]}
        st["by_overlap"] = by_overlap(psc, splits[s], gt_of, kind_of)
        if so.OUT.exists():   # analyze_split_overlap.py の件数と一致すること
            ref = json.loads(so.OUT.read_text(encoding="utf-8"))["settings"][s]
            got = st["by_overlap"]
            if (sum(got["shared_equation"]["n_test"].values()) != ref["shared_equation"]
                    or sum(got["no_shared_source"]["n_test"].values()) != ref["n_test_records"] - ref["shared_source"]):
                sys.exit(f"{s}: 正解式を共有するテスト用のケースの数が split_overlap.json と食い違う")
        out["settings"][s] = st

    out["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for s, st in out["settings"].items():
        print(f"### setting {s} (ID judgment; mean ± SD over 10 splits)")
        for name, m in st["methods"].items():
            print(f"  {name:9s} R@20 {m['recall@20']['mean']:.3f} ± {m['recall@20']['sd']:.3f}  "
                  f"MAP@20 {m['AP@20']['mean']:.3f} ± {m['AP@20']['sd']:.3f}")
        for k, v in st["compare"].items():
            print(f"    {k:22s} R@20 {v['recall@20']['mean_diff']:+.3f} ({v['recall@20']['wins']}/10)  "
                  f"MAP@20 {v['AP@20']['mean_diff']:+.3f} ({v['AP@20']['wins']}/10)")
        if s == "A":
            for k in KINDS:
                print(f"  {k:6s} " + " | ".join(
                    f"{n} R {m['by_kind'][k]['recall@20']['mean']:.3f}±{m['by_kind'][k]['recall@20']['sd']:.3f} "
                    f"MAP {m['by_kind'][k]['AP@20']['mean']:.3f}±{m['by_kind'][k]['AP@20']['sd']:.3f}"
                    for n, m in st["methods"].items()))
            print("  test cases per split by kind:", {k: (min(v.values()), max(v.values())) for k, v in st["n_test_by_kind"].items()})
    for name, v in out["models"].items():
        print(f"{name}: runs R@20 SD mean {v['recall@20_sd_mean']:.3f}, all runs same {v['cases_all_runs_same']:.3f}, "
              f"Jaccard {v['jaccard_mean']:.3f}, same list {v['same_list_share']:.4f}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
