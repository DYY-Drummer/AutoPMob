"""A1: 第1段の混合重みスイープの集計（重み × モードの Recall@K と対応検定）.

入力 : experiments/xs1/{mode}_{wlabel}__{seed}.json（run_stage1_weights.sh の出力。
        mode = reranker-10S | baseline、wlabel = w50-50 | w30-70 | w00-100）
参照 : 現行重み w70-30 は再実行せず既存の正典を使う（同じ層化分割・同じ seed）:
        - reranker-10S : experiments/xd/reranker-10S__{seed}.json
        - baseline     : experiments/strat_A.json results.baseline.per_case
出力 : experiments/stage1_weight_stats.json
検定 : seed 対応 paired t / Wilcoxon（各重み vs w70-30、モード内比較）。
        被覆率（experiments/stage1_weight_coverage.json, k=50）を各重みに併記し、
        「上限の上げ幅」と「最終指標の上げ幅」を並べて読めるようにする。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_greedy_3way import paired, seed_means

ROOT = Path(__file__).parent
METRIC = "Recall@K_correct"
MODES = ["baseline", "reranker-10S"]
WLABELS = ["w70-30", "w50-50", "w30-70", "w00-100"]  # w70-30 は参照（既存）

NAME_RE = re.compile(r"^(?P<mode>.+?)_(?P<wl>w\d{2}-\d{2,3})__(?P<seed>\d+)\.json$")


def parse_name(name: str):
    """"{mode}_{wlabel}__{seed}.json" を分解する。合わなければ None."""
    m = NAME_RE.match(name)
    if not m:
        return None
    return (m.group("mode"), m.group("wl"), int(m.group("seed")))


def load_sweep(xs1_dir: Path):
    """(mode, wlabel) -> [per_case records]."""
    out = defaultdict(list)
    for f in sorted(glob.glob(str(xs1_dir / "*.json"))):
        parsed = parse_name(Path(f).name)
        if parsed is None:
            continue
        mode, wl, seed = parsed
        d = json.load(open(f, encoding="utf-8"))
        for _, r in d.get("results", {}).items():
            for rec in r.get("per_case", []):
                out[(mode, wl)].append(dict(rec))
    return out


def load_reference_10s(xd_dir: Path):
    recs = []
    for f in sorted(glob.glob(str(xd_dir / "reranker-10S__*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        for _, r in d.get("results", {}).items():
            recs.extend(dict(rec) for rec in r.get("per_case", []))
    return recs


def load_reference_baseline(strat_json: Path):
    d = json.load(open(strat_json, encoding="utf-8"))
    return [dict(r) for r in d["results"]["baseline"].get("per_case", [])]


def overall_entry(recs) -> dict:
    """per-case 記録から seed 平均の要約（箱ひげ図用の per_seed を含む）を作る."""
    sm = seed_means(recs)
    vals = list(sm.values())
    return {
        "mean": round(float(np.mean(vals)), 4) if vals else None,
        "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
        "n_seeds": len(vals), "n_records": len(recs),
        "per_seed": {str(s): round(float(v), 4) for s, v in sorted(sm.items())},
    }


def make_report_figure(out_png="docs/figures/fig_stage1_redesign_ja.png"):
    """進捗報告用の日本語ラベル図（英語版 fig_stage1_redesign と同レイアウト）.

    (a) 重み別の被覆率 vs 候補件数 k、(b) k=50 の Recall@K 箱ひげ（乱数 10 通り）。
    データ: experiments/stage1_weight_coverage.json / stage1_weight_stats.json
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    try:
        import japanize_matplotlib  # noqa
    except Exception:
        for fam in ["Hiragino Sans", "Hiragino Maru Gothic Pro", "Yu Gothic", "AppleGothic"]:
            try:
                plt.rcParams["font.family"] = fam; break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42

    cov = json.load(open(ROOT / "experiments" / "stage1_weight_coverage.json"))
    st = json.load(open(ROOT / "experiments" / "stage1_weight_stats.json"))
    ks = cov["config"]["ks"]
    order = ["w70-30", "w50-50", "w30-70", "w00-100"]
    color = {"w70-30": "#9aa0a6", "w50-50": "#f9ab00",
             "w30-70": "#1a73e8", "w00-100": "#188038"}
    label = {"w70-30": "現行（文章 0.7＋変数 0.3）", "w50-50": "文章 0.5＋変数 0.5",
             "w30-70": "文章 0.3＋変数 0.7", "w00-100": "変数のみ"}
    tick = {"w70-30": "0.7 / 0.3\n（現行）", "w50-50": "0.5 / 0.5",
            "w30-70": "0.3 / 0.7", "w00-100": "0.0 / 1.0"}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    ax = axes[0]
    for wl in order:
        ys = [cov["coverage"][wl][str(k)]["mean"] for k in ks]
        es = [cov["coverage"][wl][str(k)]["sem"] for k in ks]
        ax.errorbar(ks, ys, yerr=es, marker="o", ms=4.5, lw=2.0, capsize=3,
                    color=color[wl], label=label[wl])
    for kk in (50, 200):
        ax.axvline(kk, color="#888888", ls=":", lw=1.1)
    ax.set_xscale("log")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("候補件数 k（対数軸。点線は実験で使う k=50 と k=200）")
    ax.set_ylabel("正解式が候補に入る割合（平均±標準誤差）")
    ax.set_title("(a) 第 1 段の重み別の被覆率＝第 2 段の上限")
    ax.set_ylim(0.70, 1.005)
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    rng = np.random.default_rng(0)
    centers = np.arange(len(order)) * 2.2
    for i, wl in enumerate(order):
        for off, method, col in ((-0.42, "baseline", "#9aa0a6"),
                                 (0.42, "reranker-10S", "#1a73e8")):
            vals = list(st["overall"][method][wl]["per_seed"].values())
            bp = ax.boxplot([vals], positions=[centers[i] + off], widths=0.7,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.2))
            bp["boxes"][0].set_facecolor(col); bp["boxes"][0].set_alpha(0.85)
            x = rng.normal(centers[i] + off, 0.05, size=len(vals))
            ax.scatter(x, vals, s=12, color="black", alpha=0.5, zorder=3)
    ax.set_xticks(centers)
    ax.set_xticklabels([tick[wl] for wl in order], fontsize=9)
    ax.set_xlabel("第 1 段の重み（文章／変数）")
    ax.set_ylabel("Recall@K（乱数 10 通りそれぞれの平均）")
    ax.set_title("(b) 同じ重みで訓練し直したときの Recall@K（k=50）")
    handles = [Patch(facecolor="#9aa0a6", alpha=0.85, label="古典的情報検索手法（baseline）"),
               Patch(facecolor="#1a73e8", alpha=0.85, label="本手法（reranker-10S、訓練し直し）")]
    ax.legend(handles=handles, fontsize=8.5, loc="center right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = ROOT / out_png
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(out).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_png} (+.pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="experiments/xs1")
    ap.add_argument("--out-json", default="experiments/stage1_weight_stats.json")
    ap.add_argument("--report-figure", action="store_true",
                    help="進捗報告用の日本語図 docs/figures/fig_stage1_redesign_ja を生成")
    args = ap.parse_args()
    xs1 = ROOT / args.dir

    data = load_sweep(xs1)
    data[("reranker-10S", "w70-30")] = load_reference_10s(ROOT / "experiments" / "xd")
    data[("baseline", "w70-30")] = load_reference_baseline(
        ROOT / "experiments" / "strat_A.json")

    try:
        covj = json.load(open(ROOT / "experiments" / "stage1_weight_coverage.json",
                              encoding="utf-8"))
        cov50 = {wl: covj["coverage"][wl]["50"]["mean"] for wl in WLABELS
                 if wl in covj.get("coverage", {})}
    except FileNotFoundError:
        cov50 = {}

    out = {"metric": METRIC, "setting": "A", "top_k": 50,
           "reference": "w70-30 (reranker-10S: experiments/xd, baseline: strat_A.json)",
           "coverage_k50": cov50, "overall": {}, "tests": {}, "by_n_correct": {}}

    for mode in MODES:
        out["overall"][mode] = {}
        for wl in WLABELS:
            out["overall"][mode][wl] = overall_entry(data.get((mode, wl), []))

    for mode in MODES:
        ref = data.get((mode, "w70-30"), [])
        out["tests"][mode] = {}
        for wl in WLABELS[1:]:
            out["tests"][mode][f"{wl}_vs_w70-30"] = paired(data.get((mode, wl), []), ref)

    # 正解式数別（reranker-10S のみ。§6.7 の難ケース議論に対応）
    for wl in WLABELS:
        recs = data.get(("reranker-10S", wl), [])
        rows = []
        for x in range(1, 11):
            sm = seed_means(recs, sub=lambda r, x=x: r.get("n_correct") == x)
            vals = list(sm.values())
            rows.append({"X": x,
                         "mean": round(float(np.mean(vals)), 4) if vals else None,
                         "sem": round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4)
                                if len(vals) > 1 else None})
        out["by_n_correct"][wl] = rows
    # 最難端 X>=8 の対応検定（reranker-10S、各重み vs 現行）
    out["tests_hard_X8plus"] = {}
    ref10 = data.get(("reranker-10S", "w70-30"), [])
    for wl in WLABELS[1:]:
        out["tests_hard_X8plus"][f"{wl}_vs_w70-30"] = paired(
            data.get(("reranker-10S", wl), []), ref10,
            sub=lambda r: r.get("n_correct", 1) >= 8)

    # --- フェーズ2: 最強構成（学習版 greedy・k=200）で w30-70 が効くか ---
    # 参照 = 現行重み w70-30 の学習版 greedy（experiments/xg_A/train__{seed}.json）
    k200_dir = ROOT / "experiments" / "xs1_k200"
    if k200_dir.exists():
        d200 = load_sweep(k200_dir)
        new200 = d200.get(("train", "w30-70"), [])
        ref200 = []
        for f in sorted(glob.glob(str(ROOT / "experiments" / "xg_A" / "train__*.json"))):
            dd = json.load(open(f, encoding="utf-8"))
            for _, r in dd.get("results", {}).items():
                ref200.extend(dict(rec) for rec in r.get("per_case", []))
        if new200 and ref200:
            out["k200_train_greedy"] = {
                "w30-70": overall_entry(new200),
                "w70-30_ref": overall_entry(ref200),
                "test_w30-70_vs_w70-30": paired(new200, ref200),
                "test_hard_X8plus": paired(new200, ref200,
                                           sub=lambda r: r.get("n_correct", 1) >= 8),
            }

    out_path = ROOT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved: {args.out_json}")

    print(f"\n=== {METRIC}（設定A・k=50・seed平均±sd）/ 被覆率(k=50) ===")
    for mode in MODES:
        for wl in WLABELS:
            o = out["overall"][mode][wl]
            if o["mean"] is None:
                continue
            cov = cov50.get(wl)
            cov_s = f"  cov={cov:.4f}" if cov is not None else ""
            print(f"  {mode:14s} {wl:8s} {o['mean']:.4f} ± {o['std'] or 0:.4f} "
                  f"(n_seeds={o['n_seeds']}){cov_s}")
    print("\n=== 対応検定（vs w70-30）===")
    for mode in MODES:
        for name, r in out["tests"][mode].items():
            if "mean_delta" in r:
                print(f"  {mode:14s} {name:22s} Δ={r['mean_delta']:+.4f} "
                      f"t p={r.get('p_ttest','-')} W p={r.get('p_wilcoxon','-')} "
                      f"dz={r.get('cohen_dz','-')}")
    print("\n=== 最難端 X>=8（reranker-10S）===")
    for name, r in out["tests_hard_X8plus"].items():
        if "mean_delta" in r:
            print(f"  {name:22s} Δ={r['mean_delta']:+.4f} t p={r.get('p_ttest','-')} "
                  f"W p={r.get('p_wilcoxon','-')}")
    if args.report_figure:
        make_report_figure()
    if "k200_train_greedy" in out:
        g = out["k200_train_greedy"]
        print("\n=== フェーズ2: 学習版 greedy・k=200 ===")
        print(f"  w30-70  {g['w30-70']['mean']} ± {g['w30-70']['std']} "
              f"(n_seeds={g['w30-70']['n_seeds']})")
        print(f"  w70-30  {g['w70-30_ref']['mean']} ± {g['w70-30_ref']['std']} (参照 xg_A/train)")
        t = g["test_w30-70_vs_w70-30"]
        if "mean_delta" in t:
            print(f"  Δ={t['mean_delta']:+.4f} t p={t.get('p_ttest','-')} "
                  f"W p={t.get('p_wilcoxon','-')} dz={t.get('cohen_dz','-')}")
        th = g["test_hard_X8plus"]
        if "mean_delta" in th:
            print(f"  X>=8: Δ={th['mean_delta']:+.4f} W p={th.get('p_wilcoxon','-')}")


if __name__ == "__main__":
    main()
