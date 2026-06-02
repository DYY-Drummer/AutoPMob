"""Phase 4.1: 特性別 (stratified) 性能分析.

入力:
  experiments/phase4_per_case.json (--save-per-case で生成された JSON)

層別軸:
  - 正解式数 n_correct: 1, 2, 3, 4, 5-10
  - 横断ソース数 n_sources: 1, 2, 3+
  - 入力変数数 n_input: 1-3, 4-6, 7-10, 11+
  - 出力変数数 n_output: 1, 2, 3+

各層について、baseline vs reranker-10S の以下を出力:
  - 件数 (n)、MRR_first 平均、Recall@K_correct 平均、MAP 平均
  - reranker 改善幅 Δ (差の平均)

出力:
  experiments/phase4_characteristic_analysis.json
  docs/figures/fig_phase4_stratified.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

ROOT = Path(__file__).parent
EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


# ---- 層別バケット定義 ----
def bucket_n_correct(n):
    if n <= 4:
        return f"{n}"
    return "5-10"


def bucket_n_sources(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def bucket_n_input(n):
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    if n <= 10:
        return "7-10"
    return "11+"


def bucket_n_output(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


BUCKETERS = {
    "n_correct": (bucket_n_correct, ["1", "2", "3", "4", "5-10"]),
    "n_sources": (bucket_n_sources, ["1", "2", "3+"]),
    "n_input": (bucket_n_input, ["1-3", "4-6", "7-10", "11+"]),
    "n_output": (bucket_n_output, ["1", "2", "3+"]),
}

METRICS = ["MRR_first", "Recall@K_correct", "AP"]
METRIC_LABELS = {
    "MRR_first": "MRR (最良順位)",
    "Recall@K_correct": "Recall@K_correct",
    "AP": "MAP",
}


def load_per_case(path: Path):
    data = json.loads(path.read_text())
    results = data["results"]
    out = {}
    for mode, summary in results.items():
        pc = summary.get("per_case")
        if pc is None:
            print(f"  WARNING: mode '{mode}' has no per_case dump", file=sys.stderr)
            continue
        out[mode] = pc
    return out, data.get("config", {})


def stratify(per_case_by_mode, axis):
    """軸 axis (例 'n_correct') で per_case 結果を層別化."""
    bucket_fn, order = BUCKETERS[axis]
    out = {}  # bucket -> mode -> [case dicts]
    for bucket in order:
        out[bucket] = {mode: [] for mode in per_case_by_mode}
    for mode, recs in per_case_by_mode.items():
        for r in recs:
            v = r.get(axis)
            if v is None:
                continue
            b = bucket_fn(v)
            if b not in out:
                out[b] = {m: [] for m in per_case_by_mode}
            out[b][mode].append(r)
    return out, order


def summarize_bucket(records, metrics=METRICS):
    """ケースリスト → 各メトリクスの平均."""
    if not records:
        return {m: None for m in metrics} | {"n": 0}
    out = {"n": len(records)}
    for m in metrics:
        vals = [r.get(m) for r in records if r.get(m) is not None]
        out[m] = float(np.mean(vals)) if vals else None
    return out


def compute_delta(bucket_summary, mode_a, mode_b):
    """mode_b − mode_a の差 (per-case mean of diffs)."""
    deltas = {}
    for axis_bucket, by_mode in bucket_summary.items():
        a = by_mode.get(mode_a, {})
        b = by_mode.get(mode_b, {})
        d = {}
        for m in METRICS:
            if a.get(m) is None or b.get(m) is None:
                d[m] = None
            else:
                d[m] = b[m] - a[m]
        d["n"] = a.get("n", 0)
        deltas[axis_bucket] = d
    return deltas


def per_case_paired_diffs(per_case_by_mode, mode_a, mode_b, axis):
    """seed × case_id でペアを作って差を集める."""
    bucket_fn, _ = BUCKETERS[axis]
    a_idx = {(r["seed"], r["case_id"]): r for r in per_case_by_mode.get(mode_a, [])}
    b_idx = {(r["seed"], r["case_id"]): r for r in per_case_by_mode.get(mode_b, [])}
    common = set(a_idx) & set(b_idx)
    pairs_by_bucket = defaultdict(list)
    for k in common:
        a, b = a_idx[k], b_idx[k]
        v = a.get(axis)
        if v is None:
            continue
        bucket = bucket_fn(v)
        pairs_by_bucket[bucket].append({
            m: (b.get(m) - a.get(m)) if (a.get(m) is not None and b.get(m) is not None) else None
            for m in METRICS
        })
    return pairs_by_bucket


def run_analysis(per_case_path: Path, baseline_mode="baseline", reranker_mode="reranker-10S"):
    per_case_by_mode, config = load_per_case(per_case_path)
    print(f"Loaded per-case dump:")
    for mode, recs in per_case_by_mode.items():
        seeds_used = sorted({r["seed"] for r in recs})
        n_unique_cases = len({r["case_id"] for r in recs})
        print(f"  {mode:14s}: {len(recs)} records, {len(seeds_used)} seeds, {n_unique_cases} unique cases")

    all_results = {}
    for axis in BUCKETERS:
        strata, order = stratify(per_case_by_mode, axis)
        bucket_summary = {}
        for bucket in order:
            by_mode = strata.get(bucket, {})
            bucket_summary[bucket] = {mode: summarize_bucket(recs) for mode, recs in by_mode.items()}
        # paired deltas (case-level, more powerful than mean-of-means)
        pairs = per_case_paired_diffs(per_case_by_mode, baseline_mode, reranker_mode, axis)
        paired_delta = {}
        for bucket in order:
            ps = pairs.get(bucket, [])
            if not ps:
                paired_delta[bucket] = {m: None for m in METRICS} | {"n_pairs": 0}
                continue
            d = {"n_pairs": len(ps)}
            for m in METRICS:
                vals = [p[m] for p in ps if p.get(m) is not None]
                d[m] = float(np.mean(vals)) if vals else None
            paired_delta[bucket] = d
        all_results[axis] = {
            "buckets": bucket_summary,
            "paired_delta": paired_delta,
            "order": order,
        }

    return all_results, config


def print_table(all_results, baseline_mode="baseline", reranker_mode="reranker-10S"):
    for axis, res in all_results.items():
        print(f"\n{'='*78}")
        print(f"  軸: {axis}")
        print(f"{'='*78}")
        order = res["order"]
        print(f"{'bucket':<10s}{'n':>6s}  "
              f"{'base MRR':>9s}{'rer MRR':>9s}{'ΔMRR':>8s}  "
              f"{'base R@K':>9s}{'rer R@K':>9s}{'ΔR@K':>8s}  "
              f"{'base MAP':>9s}{'rer MAP':>9s}{'ΔMAP':>8s}")
        print("-" * 78)
        for bucket in order:
            b = res["buckets"].get(bucket, {})
            pd = res["paired_delta"].get(bucket, {})
            base = b.get(baseline_mode, {})
            rer = b.get(reranker_mode, {})
            n = base.get("n", 0)
            if n == 0:
                print(f"{bucket:<10s}{0:>6d}  (no cases)")
                continue
            fmt = lambda v: f"{v:>9.4f}" if v is not None else f"{'-':>9s}"
            dfmt = lambda v: f"{v:>+8.3f}" if v is not None else f"{'-':>8s}"
            print(f"{bucket:<10s}{n:>6d}  "
                  f"{fmt(base.get('MRR_first'))}{fmt(rer.get('MRR_first'))}{dfmt(pd.get('MRR_first'))}  "
                  f"{fmt(base.get('Recall@K_correct'))}{fmt(rer.get('Recall@K_correct'))}{dfmt(pd.get('Recall@K_correct'))}  "
                  f"{fmt(base.get('AP'))}{fmt(rer.get('AP'))}{dfmt(pd.get('AP'))}")


def plot_stratified(all_results, out_png: Path,
                    baseline_mode="baseline", reranker_mode="reranker-10S"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    metric_to_plot = "Recall@K_correct"
    for ax, (axis, res) in zip(axes, all_results.items()):
        order = res["order"]
        xs, n_list, base_vals, rer_vals, delta = [], [], [], [], []
        for bucket in order:
            b = res["buckets"].get(bucket, {})
            pd = res["paired_delta"].get(bucket, {})
            base = b.get(baseline_mode, {})
            rer = b.get(reranker_mode, {})
            if base.get("n", 0) == 0:
                continue
            xs.append(bucket); n_list.append(base.get("n", 0))
            base_vals.append(base.get(metric_to_plot) or 0)
            rer_vals.append(rer.get(metric_to_plot) or 0)
            delta.append(pd.get(metric_to_plot) or 0)
        x_pos = np.arange(len(xs))
        w = 0.38
        ax.bar(x_pos - w/2, base_vals, w, label="baseline", color="#888")
        ax.bar(x_pos + w/2, rer_vals, w, label="reranker-10S", color="#1f77b4")
        for i, (bv, rv, n, d) in enumerate(zip(base_vals, rer_vals, n_list, delta)):
            ax.text(i, max(bv, rv) + 0.02, f"n={n}\nΔ={d:+.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x_pos); ax.set_xticklabels(xs)
        ax.set_title(f"{axis} 別の {METRIC_LABELS[metric_to_plot]}")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(METRIC_LABELS[metric_to_plot])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Phase 4.1: 特性別性能 (baseline vs reranker-10S)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved figure: {out_png}")


def find_method_shines_cases(per_case_by_mode, baseline_mode, reranker_mode,
                              metric="Recall@K_correct", top_n=20):
    """reranker が baseline より大きく改善したケース上位 N 件 (seed 平均で比較)."""
    a_by_id = defaultdict(list)
    b_by_id = defaultdict(list)
    meta_by_id = {}
    for r in per_case_by_mode.get(baseline_mode, []):
        a_by_id[r["case_id"]].append(r.get(metric))
        meta_by_id[r["case_id"]] = {k: r.get(k) for k in
                                    ("variant", "n_correct", "n_input", "n_output", "n_sources")}
    for r in per_case_by_mode.get(reranker_mode, []):
        b_by_id[r["case_id"]].append(r.get(metric))
    rows = []
    for cid in a_by_id:
        a = [v for v in a_by_id[cid] if v is not None]
        b = [v for v in b_by_id.get(cid, []) if v is not None]
        if not a or not b:
            continue
        a_mean, b_mean = float(np.mean(a)), float(np.mean(b))
        rows.append({"case_id": cid,
                     "baseline": a_mean,
                     "reranker": b_mean,
                     "delta": b_mean - a_mean,
                     "n_seeds": min(len(a), len(b)),
                     **meta_by_id.get(cid, {})})
    rows.sort(key=lambda r: -r["delta"])
    return rows[:top_n]


def print_method_shines(rows, metric="Recall@K_correct"):
    print(f"\n=== reranker が最も効くケース TOP {len(rows)} ({metric}) ===")
    print(f"{'case_id':<30s} {'variant':<22s} {'#cor':>5s} {'#in':>4s} {'#out':>4s} {'#src':>5s}  "
          f"{'base':>6s} {'rer':>6s} {'Δ':>7s}")
    print("-" * 100)
    for r in rows:
        print(f"{r['case_id']:<30s} {r.get('variant',''):<22s} "
              f"{r.get('n_correct',0):>5d} {r.get('n_input',0):>4d} "
              f"{r.get('n_output',0):>4d} {r.get('n_sources',0):>5d}  "
              f"{r['baseline']:>6.3f} {r['reranker']:>6.3f} {r['delta']:>+7.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="experiments/phase4_per_case.json",
                        help="--save-per-case で生成された JSON")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--reranker", default="reranker-10S")
    parser.add_argument("--output-json", default="experiments/phase4_characteristic_analysis.json")
    parser.add_argument("--output-png", default="docs/figures/fig_phase4_stratified.png")
    parser.add_argument("--top-shines", type=int, default=20,
                        help="reranker が最も効くケースを N 件表示")
    args = parser.parse_args()

    per_case_path = ROOT / args.input
    if not per_case_path.exists():
        print(f"ERROR: per-case dump not found: {per_case_path}")
        print(f"  hint: re-run set_aware_reranker.py with --save-per-case")
        sys.exit(1)

    all_results, config = run_analysis(per_case_path, args.baseline, args.reranker)
    print_table(all_results, args.baseline, args.reranker)

    # method-shines extraction
    per_case_by_mode, _ = load_per_case(per_case_path)
    shines = find_method_shines_cases(per_case_by_mode, args.baseline, args.reranker,
                                      top_n=args.top_shines)
    print_method_shines(shines)

    out_path = ROOT / args.output_json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": config, "axes": all_results,
                   "method_shines_top": shines},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {out_path}")

    plot_stratified(all_results, ROOT / args.output_png, args.baseline, args.reranker)


if __name__ == "__main__":
    main()
