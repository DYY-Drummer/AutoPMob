"""特徴量 × 難易度クロス分析：機構の「指紋」図と対応のある検定.

入力 : set_aware_reranker.py --save-per-case の出力 JSON
        （modes: reranker-7, reranker-7+Comp, reranker-7+Coh, reranker-7+Dom, reranker-10S）
出力 :
  - experiments/feature_x_difficulty_stats.json  … per-bin デルタ＋seed単位の対応のある検定
  - docs/figures/fig_mechanism_fingerprint.png   … 機構の指紋図（特徴別 × 難易度）

考え方:
  各 set-aware 特徴の「上乗せ」を reranker-7（基本7特徴のMLP）からの差で測る。
  - lift_X(bin) = mean Recall@K[reranker-7+X] - mean Recall@K[reranker-7]  （bin = 難易度層）
  仮説: Comp の lift は正解式数とともに立ち上がり、Dom は平坦。
  検定: seed を対応の単位として paired t / Wilcoxon（mode vs reranker-7）。
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
METRIC = "Recall@K_correct"

# モード表示名・色
MODES = {
    "reranker-7":      ("基本7特徴量のみ",        "#9e9e9e"),
    "reranker-7+Comp": ("＋補完性",              "#1f77b4"),
    "reranker-7+Coh":  ("＋一貫性",              "#2ca02c"),
    "reranker-7+Dom":  ("＋ドメイン一致",         "#d62728"),
    "reranker-10S":    ("＋3つすべて（reranker-10S）", "#000000"),
}
SET_FEATS = ["reranker-7+Comp", "reranker-7+Coh", "reranker-7+Dom"]


def load_percase(path):
    """{mode: [ {seed, case_id, n_correct, n_input, ..., METRIC}, ... ]} を返す."""
    d = json.load(open(path, encoding="utf-8"))
    res = d.get("results", d)
    out = {}
    for mode, r in res.items():
        pc = r.get("per_case")
        if pc:
            out[mode] = pc
    return out, d.get("config", {})


def per_seed_mean(records, bin_key=None, bin_val=None, sub=None):
    """seed -> mean(METRIC) を返す。bin_key/bin_val 指定時はその層に限定。
    sub: (key, predicate) で追加フィルタ（例 multi: n_correct>=2）。"""
    by_seed = defaultdict(list)
    for r in records:
        if METRIC not in r:
            continue
        if bin_key is not None and not bin_val(r.get(bin_key)):
            continue
        if sub is not None and not sub(r):
            continue
        by_seed[r["seed"]].append(r[METRIC])
    return {s: float(np.mean(v)) for s, v in by_seed.items() if v}


def paired_test(mode_records, base_records, sub=None):
    """seed 単位の対応のある検定（mode vs base）。"""
    m = per_seed_mean(mode_records, sub=sub)
    b = per_seed_mean(base_records, sub=sub)
    seeds = sorted(set(m) & set(b))
    a = np.array([m[s] for s in seeds]); c = np.array([b[s] for s in seeds])
    diff = a - c
    res = {
        "n_seeds": len(seeds),
        "mean_mode": round(float(a.mean()), 4),
        "mean_base": round(float(c.mean()), 4),
        "mean_delta": round(float(diff.mean()), 4),
        "std_delta": round(float(diff.std(ddof=1)) if len(diff) > 1 else 0.0, 4),
    }
    if len(seeds) >= 2 and diff.std() > 0:
        t, p = stats.ttest_rel(a, c)
        res["t"] = round(float(t), 3); res["p_ttest"] = round(float(p), 5)
        # Cohen's d (対応のある, dz)
        res["cohen_dz"] = round(float(diff.mean() / diff.std(ddof=1)), 3)
        try:
            w, pw = stats.wilcoxon(a, c)
            res["p_wilcoxon"] = round(float(pw), 5)
        except Exception:
            res["p_wilcoxon"] = None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/ablation_x_difficulty.json")
    ap.add_argument("--out-json", default="experiments/feature_x_difficulty_stats.json")
    ap.add_argument("--out-fig", default="docs/figures/fig_mechanism_fingerprint.png")
    args = ap.parse_args()

    data, config = load_percase(ROOT / args.input)
    assert "reranker-7" in data, "reranker-7 (base) が入力に無い"
    base = data["reranker-7"]

    out = {"config": config, "metric": METRIC, "n_per_mode": {m: len(v) for m, v in data.items()}}

    # ---- 1. 全体＋multi での対応のある検定 ----
    out["overall_tests"] = {}
    out["multi_tests"] = {}
    for mode in data:
        if mode == "reranker-7":
            continue
        out["overall_tests"][mode] = paired_test(data[mode], base)
        out["multi_tests"][mode] = paired_test(data[mode], base, sub=lambda r: r.get("n_correct", 1) >= 2)
    # Comp vs Dom（直接比較）
    if "reranker-7+Comp" in data and "reranker-7+Dom" in data:
        out["comp_vs_dom"] = paired_test(data["reranker-7+Comp"], data["reranker-7+Dom"])

    # ---- 2. 難易度層別 lift（seed平均→層平均、±SEM）----
    def bin_lift(bin_key, bins):
        """bins: [(label, predicate), ...] -> {mode: [(label, n, lift_mean, lift_sem), ...]}"""
        result = {}
        base_seedmean = {lbl: per_seed_mean(base, bin_key, pred) for lbl, pred in bins}
        for mode in SET_FEATS + ["reranker-10S"]:
            if mode not in data:
                continue
            rows = []
            for lbl, pred in bins:
                msm = per_seed_mean(data[mode], bin_key, pred)
                bsm = base_seedmean[lbl]
                seeds = sorted(set(msm) & set(bsm))
                if not seeds:
                    rows.append((lbl, 0, None, None)); continue
                diffs = np.array([msm[s] - bsm[s] for s in seeds])
                n_cases = sum(1 for r in data[mode] if bin_key in r and pred(r.get(bin_key)))
                rows.append((lbl, n_cases,
                             round(float(diffs.mean()), 4),
                             round(float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0, 4)))
            result[mode] = rows
        return result

    ncorr_bins = [(str(k), (lambda v, k=k: v == k)) for k in range(1, 11)]
    nin_bins = [("1-3", lambda v: v is not None and v <= 3),
                ("4-6", lambda v: v is not None and 4 <= v <= 6),
                ("7-10", lambda v: v is not None and 7 <= v <= 10),
                ("11-15", lambda v: v is not None and 11 <= v <= 15),
                ("16+", lambda v: v is not None and v >= 16)]

    out["lift_by_n_correct"] = bin_lift("n_correct", ncorr_bins)
    out["lift_by_n_input"] = bin_lift("n_input", nin_bins)

    json.dump(out, open(ROOT / args.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved stats: {args.out_json}")

    # ---- 3. 指紋図 ----
    try:
        import japanize_matplotlib  # noqa
    except Exception:
        for fam in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "AppleGothic"]:
            try:
                plt.rcParams["font.family"] = fam; break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    def plot_panel(ax, lift_dict, title, xlabel):
        for mode in SET_FEATS + ["reranker-10S"]:
            if mode not in lift_dict:
                continue
            rows = [r for r in lift_dict[mode] if r[2] is not None and r[1] >= 5]
            if not rows:
                continue
            xs = list(range(len(rows)))
            ys = [r[2] for r in rows]
            es = [r[3] for r in rows]
            lbl, col = MODES[mode]
            style = dict(marker="o", lw=2.2, color=col, label=lbl)
            if mode == "reranker-10S":
                style.update(lw=1.4, ls="--", marker="s", alpha=0.7)
            ax.errorbar(xs, ys, yerr=es, capsize=3, **style)
        ax.axhline(0, color="#bbbbbb", lw=1, zorder=0)
        ref = lift_dict.get("reranker-7+Comp") or next(iter(lift_dict.values()))
        labels = [r[0] for r in ref if r[2] is not None and r[1] >= 5]
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_xlabel(xlabel)
        ax.set_ylabel("Recall@K の上乗せ（基本7特徴量のみとの差）")
        ax.grid(alpha=0.25); ax.legend(fontsize=9, loc="upper left")

    plot_panel(axes[0], out["lift_by_n_correct"], "(a) 正解式数別", "正解式数")
    plot_panel(axes[1], out["lift_by_n_input"], "(b) 入力変数数別", "入力変数数")
    fig.tight_layout()
    Path(ROOT / args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / args.out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {args.out_fig}")

    # ---- 4. コンソール要約 ----
    print("\n=== 全体（seed単位の対応のある検定, vs reranker-7）===")
    print(f"{'mode':18s} {'Δ':>8s} {'±sd':>7s} {'t':>7s} {'p(t)':>9s} {'dz':>6s} {'p(W)':>9s}")
    for mode, r in out["overall_tests"].items():
        print(f"{mode:18s} {r['mean_delta']:+8.4f} {r.get('std_delta',0):7.4f} "
              f"{r.get('t',float('nan')):7.3f} {r.get('p_ttest',float('nan')):9.5f} "
              f"{r.get('cohen_dz',float('nan')):6.3f} {str(r.get('p_wilcoxon')):>9s}")
    print("\n=== 複数式のみ (n_correct>=2) ===")
    for mode, r in out["multi_tests"].items():
        print(f"{mode:18s} Δ={r['mean_delta']:+.4f}±{r.get('std_delta',0):.4f} "
              f"t={r.get('t',float('nan')):.3f} p={r.get('p_ttest',float('nan')):.5f} dz={r.get('cohen_dz',float('nan')):.3f}")
    if "comp_vs_dom" in out:
        r = out["comp_vs_dom"]
        print(f"\nComp vs Dom: Δ={r['mean_delta']:+.4f} t={r.get('t',float('nan')):.3f} p={r.get('p_ttest',float('nan')):.5f}")

    print("\n=== lift by n_correct (Δ over reranker-7) ===")
    hdr = out["lift_by_n_correct"]
    for mode in SET_FEATS:
        if mode in hdr:
            cells = " ".join(f"{r[0]}:{r[2]:+.3f}" if r[2] is not None else f"{r[0]}:--"
                             for r in hdr[mode])
            print(f"{mode:16s} {cells}")


if __name__ == "__main__":
    main()
