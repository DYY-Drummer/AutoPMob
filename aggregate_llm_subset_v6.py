"""v6 用：LLM 直接生成 vs Stage1(baseline) vs Stage2(reranker-10S) の公平な三者比較.

現行 DB（11,146 式）で `set_aware_reranker.py --save-per-case` を実行した
per-case 出力（experiments/phase6_full_percase.json）から、
LLM ベースライン評価と同一の対象ケース（単一ソース 29 件・複数ソース 20 件）
だけを抽出して集計する。各対象ケースは「自身のソースがテスト分割に入った
シードでのみ」評価されるため、reranker/baseline は当該ケースを学習せずに
評価する（held-out）。これにより LLM 直接生成と公平に比較できる。

出力: experiments/phase6_llm_subset_comparison.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"

PERCASE = EXP / "phase6_full_percase.json"
LLM_SINGLE = EXP / "llm_baseline_results.json"
LLM_MULTI = EXP / "llm_baseline_multisource_results.json"
OUT = EXP / "phase6_llm_subset_comparison.json"

# 比較する指標（3 主指標）。per-case 記録では MAP は "AP"、
# Recall@K_correct は "Precision@C" と同名で保存されている。
METRICS = ["MRR_first", "Recall@K_correct", "MAP"]


def get_metric(rec: dict, key: str):
    """per-case 記録のキー揺れを吸収。"""
    if key in rec and rec[key] is not None:
        return rec[key]
    if key == "Recall@K_correct" and rec.get("Precision@C") is not None:
        return rec["Precision@C"]
    if key == "MAP" and rec.get("AP") is not None:  # per-case の平均適合率
        return rec["AP"]
    return None


def aggregate_subset(per_case_records: list[dict], target_ids: list[str]):
    """対象ケースごとにシード平均→ケース平均で集計."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for r in per_case_records:
        by_case[r.get("case_id")].append(r)

    per_case_means = {m: [] for m in METRICS}
    n_seeds_per_case = []
    covered = []
    missing = []
    for cid in target_ids:
        recs = by_case.get(cid, [])
        if not recs:
            missing.append(cid)
            continue
        covered.append(cid)
        n_seeds_per_case.append(len(recs))
        for m in METRICS:
            vals = [get_metric(r, m) for r in recs]
            vals = [v for v in vals if v is not None]
            if vals:
                per_case_means[m].append(mean(vals))

    summary = {}
    for m in METRICS:
        vals = per_case_means[m]
        summary[m] = round(mean(vals), 4) if vals else None
    summary["n_target"] = len(target_ids)
    summary["n_covered"] = len(covered)
    summary["n_missing"] = len(missing)
    summary["missing_ids"] = missing
    summary["avg_test_seeds_per_case"] = round(mean(n_seeds_per_case), 2) if n_seeds_per_case else 0
    return summary


def llm_summary(path: Path):
    j = json.load(open(path))
    s = j["summary"]
    out = {}
    for m in METRICS:
        if m == "Recall@K_correct":
            out[m] = round(s.get("Precision@C"), 4) if s.get("Precision@C") is not None else None
        else:
            out[m] = round(s.get(m), 4) if s.get(m) is not None else None
    out["n_cases"] = s.get("n_cases")
    return out


def main():
    pc = json.load(open(PERCASE))
    results = pc["results"]

    single_ids = [d["case_id"] for d in json.load(open(LLM_SINGLE))["detailed"]]
    multi_ids = [d["case_id"] for d in json.load(open(LLM_MULTI))["detailed"]]

    out = {"db_equations": None, "note": "現行 DB(11,146 式)・10 seed・held-out 評価。"
           " reranker/baseline は対象ケースをテスト分割で評価（未学習）。",
           "subsets": {}}

    for label, ids, llm_path in [
        ("single_29", single_ids, LLM_SINGLE),
        ("multi_20", multi_ids, LLM_MULTI),
    ]:
        row = {"target_ids_n": len(ids)}
        row["LLM_direct"] = llm_summary(llm_path)
        for mode in ["baseline", "reranker-10S"]:
            recs = results.get(mode, {}).get("per_case", [])
            row[mode] = aggregate_subset(recs, ids)
        out["subsets"][label] = row

    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=2)

    # 読みやすい表を表示
    for label in ["single_29", "multi_20"]:
        row = out["subsets"][label]
        print(f"\n===== {label} (n={row['target_ids_n']}) =====")
        hdr = f"{'method':18s} " + " ".join(f"{m:>16s}" for m in METRICS)
        print(hdr)
        for method in ["LLM_direct", "baseline", "reranker-10S"]:
            r = row[method]
            cov = ""
            if method != "LLM_direct":
                cov = f"  (covered {r['n_covered']}/{r['n_target']}, ~{r['avg_test_seeds_per_case']} test-seeds/case)"
            print(f"{method:18s} " + " ".join(f"{(r.get(m) if r.get(m) is not None else float('nan')):16.4f}" for m in METRICS) + cov)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
