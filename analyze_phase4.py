"""Phase 4: 特性別分析 + 仮説 B 統計検定.

【4.1 特性別性能分析】
各ケースを以下で層別化し評価指標との関係を分析：
  - 正解式数 (1, 2, 3, 4, 5-10)
  - 横断ソース数 (1, 2, 3+)
  - 入力変数数 (1-3, 4-6, 7-10, 11+)
  - 出力変数数 (1, 2, 3+)

【4.2 仮説 B 統計検定】
Full DB vs Narrow DB の本手法 MRR 差について:
  - paired t 検定
  - Wilcoxon signed-rank test
  - 効果量 (Cohen's d)

【出力】
  - experiments/phase4_characteristic_analysis.json
  - experiments/phase4_hypothesis_b_test.json
  - docs/figures/fig_phase4_*.png
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


def _one_metric_test(f_per_seed, n_per_seed, metric_name):
    """Helper: paired t-test + Wilcoxon + Cohen's d for one metric."""
    if len(f_per_seed) != len(n_per_seed):
        n = min(len(f_per_seed), len(n_per_seed))
        f_per_seed = f_per_seed[:n]
        n_per_seed = n_per_seed[:n]
    diff = np.array(f_per_seed) - np.array(n_per_seed)
    mean_diff = diff.mean()
    sd_diff = diff.std(ddof=1) if len(diff) > 1 else 0.0
    t_stat, p_t = stats.ttest_rel(f_per_seed, n_per_seed)
    try:
        w_stat, p_w = stats.wilcoxon(f_per_seed, n_per_seed)
    except ValueError:  # all-zero differences
        w_stat, p_w = float("nan"), 1.0
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0
    return {
        "metric": metric_name,
        "n_seeds": len(f_per_seed),
        "full_mean": float(np.mean(f_per_seed)),
        "narrow_mean": float(np.mean(n_per_seed)),
        "diff_mean": float(mean_diff),
        "diff_sd": float(sd_diff),
        "paired_t": {"t_stat": float(t_stat), "p_value": float(p_t),
                     "significant_005": bool(p_t < 0.05),
                     "significant_001": bool(p_t < 0.01)},
        "wilcoxon": {"stat": float(w_stat), "p_value": float(p_w),
                     "significant_005": bool(p_w < 0.05)},
        "cohens_d": float(cohens_d),
        "effect_size_interp": ("small" if abs(cohens_d) < 0.5
                               else "medium" if abs(cohens_d) < 0.8 else "large"),
    }


def hypothesis_b_test():
    """仮説 B：Full DB vs Narrow DB の本手法 MRR / Recall@K_correct / MAP 差の検定"""
    full = json.load(open(ROOT / "experiments" / "set_aware_results.before_largeDB.bak.json"))
    narrow = json.load(open(ROOT / "experiments" / "set_aware_results_narrow_chE_only.json"))

    # Recall@K_correct ≡ Precision@C (定義上同値): 旧バックアップに無い場合は後者で代用
    metrics = ["MRR_first", "Recall@K_correct", "MAP"]
    fallback = {"Recall@K_correct": "Precision@C"}
    results = {}
    print("\n=== 仮説 B 統計検定結果 (reranker-10S, Full DB vs Narrow DB) ===")
    print(f"  Full DB n_seeds: {len(full['results']['reranker-10S']['per_seed'])}, "
          f"Narrow DB n_seeds: {len(narrow['results']['reranker-10S']['per_seed'])}")
    for m in metrics:
        keys_to_try = [m] + ([fallback[m]] if m in fallback else [])
        f_per = n_per = None
        used_key = None
        for k in keys_to_try:
            f_try = [s.get(k) for s in full["results"]["reranker-10S"]["per_seed"] if s.get(k) is not None]
            n_try = [s.get(k) for s in narrow["results"]["reranker-10S"]["per_seed"] if s.get(k) is not None]
            if f_try and n_try:
                f_per, n_per, used_key = f_try, n_try, k
                break
        if not f_per or not n_per:
            print(f"\n  {m}: per-seed values not available — skip")
            continue
        if used_key != m:
            print(f"\n  (note: '{m}' は per_seed に無いため、定義上同値の '{used_key}' を使用)")
        r = _one_metric_test(f_per, n_per, m)
        r["source_key"] = used_key
        results[m] = r
        sig = "★有意" if r["paired_t"]["significant_005"] else "n.s."
        print(f"\n  -- {m} --")
        print(f"    Full={r['full_mean']:.4f}  Narrow={r['narrow_mean']:.4f}  "
              f"Δ={r['diff_mean']:+.4f} (±{r['diff_sd']:.4f})")
        print(f"    paired t={r['paired_t']['t_stat']:+.3f}, p={r['paired_t']['p_value']:.4f} ({sig})  "
              f"  Wilcoxon p={r['wilcoxon']['p_value']:.4f}  Cohen's d={r['cohens_d']:+.3f} ({r['effect_size_interp']})")
    return {"method": "reranker-10S", "metrics": results}


def characteristic_analysis(per_case_path: Path,
                             out_json: Path,
                             out_png: Path,
                             baseline_mode="baseline",
                             reranker_mode="reranker-10S"):
    """特性別 (n_correct, n_input, n_output, n_sources) で性能を分析.

    per_case_path は set_aware_reranker.py --save-per-case で生成された JSON.
    """
    if not per_case_path.exists():
        print(f"\n=== Phase 4.1 スキップ: {per_case_path.name} が見つかりません ===")
        print(f"  生成方法: python3 set_aware_reranker.py --save-per-case --output {per_case_path.name} ...")
        return None
    # 遅延 import で hypothesis B のみ実行する場合にも依存しないように
    from analyze_phase4_characteristic import (
        run_analysis, print_table, plot_stratified,
        load_per_case, find_method_shines_cases, print_method_shines,
    )

    print(f"\n=== Phase 4.1 特性別分析（{per_case_path.name}）===")
    all_results, config = run_analysis(per_case_path, baseline_mode, reranker_mode)
    print_table(all_results, baseline_mode, reranker_mode)

    per_case_by_mode, _ = load_per_case(per_case_path)
    shines = find_method_shines_cases(per_case_by_mode, baseline_mode, reranker_mode, top_n=20)
    print_method_shines(shines)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"config": config, "axes": all_results,
                   "method_shines_top": shines},
                  f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_json}")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot_stratified(all_results, out_png, baseline_mode, reranker_mode)
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hyp-b", action="store_true",
                        help="仮説 B 検定をスキップ")
    parser.add_argument("--per-case",
                        default="experiments/phase4_per_case_A.json",
                        help="特性別分析用の per-case JSON")
    parser.add_argument("--char-out-json",
                        default="experiments/phase4_characteristic_analysis.json")
    parser.add_argument("--char-out-png",
                        default="docs/figures/fig_phase4_stratified.png")
    args = parser.parse_args()

    if not args.skip_hyp_b:
        print("=" * 60)
        print("Phase 4.2: 仮説 B 統計検定")
        print("=" * 60)
        hb_result = hypothesis_b_test()
        with open(ROOT / "experiments" / "phase4_hypothesis_b_test.json", "w") as f:
            json.dump(hb_result, f, ensure_ascii=False, indent=2)
        print(f"\n保存: experiments/phase4_hypothesis_b_test.json")

    print("\n" + "=" * 60)
    print("Phase 4.1: 特性別性能分析")
    print("=" * 60)
    characteristic_analysis(
        per_case_path=ROOT / args.per_case,
        out_json=ROOT / args.char_out_json,
        out_png=ROOT / args.char_out_png,
    )


if __name__ == "__main__":
    main()
