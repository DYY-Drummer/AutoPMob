"""PFI 結果図：Permutation Feature Importance（最良モデル reranker-10S）.

experiments/pfi_results.json を読み、
  (a) 個別特徴量の PFI（大きい順・群で色分け・エラーバー）
  (b) 群 PFI（変数重なり群 vs 話題群）＝機構の主役
を figure/fig_pfi_importance.pdf (+ .png) に出力。
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
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

JP = {
    "text_sim": "テキスト類似", "io_jaccard": "入出力Jaccard", "svd_sim": "意味類似(SVD)",
    "input_cov": "入力被覆", "output_cov": "出力被覆", "specificity": "特化度",
    "domain": "分野一致(基本)", "gComp": "補完性", "gCoh": "一貫性", "gDom": "分野一致(集合)",
}
GROUP = {  # 特徴量 -> 群
    "text_sim": "topic", "io_jaccard": "var", "svd_sim": "topic", "input_cov": "var",
    "output_cov": "var", "specificity": "var", "domain": "topic", "gComp": "var",
    "gCoh": "var", "gDom": "topic",
}
C_VAR, C_TOPIC = "#2c7fb8", "#d95f0e"
SCOPE = "case"


def main():
    d = json.load(open(ROOT / "experiments" / "pfi_results.json"))
    res = d["results"][SCOPE]
    base = d["baseline_mean"]

    # --- 個別特徴量（大きい順）---
    feats = list(JP.keys())
    feats.sort(key=lambda f: -res[f]["importance_mean"])
    vals = [res[f]["importance_mean"] for f in feats]
    errs = [res[f]["importance_std"] for f in feats]
    cols = [C_VAR if GROUP[f] == "var" else C_TOPIC for f in feats]
    labels = [JP[f] for f in feats]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [2.3, 1]})

    y = list(range(len(feats)))
    ax1.barh(y, vals, xerr=errs, color=cols, capsize=3, height=0.66,
             error_kw=dict(ecolor="#444", lw=1))
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=11); ax1.invert_yaxis()
    ax1.set_xlabel("PFI＝置換による Recall@K の低下（大きいほど重要）", fontsize=11)
    ax1.axvline(0, color="#bbb", lw=1, zorder=0)
    ax1.grid(axis="x", alpha=0.25)
    for yi, v, e in zip(y, vals, errs):
        ax1.text(v + e + 0.012, yi, f"{v:.3f}", va="center", fontsize=9)
    ax1.set_title("(a) 個別特徴量の重要度（reranker-10S・10 seed）", fontsize=12)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=C_VAR, label="変数重なり群"),
                        Patch(color=C_TOPIC, label="話題群")],
               fontsize=10, loc="lower right")
    ax1.set_xlim(right=max(v + e for v, e in zip(vals, errs)) * 1.15)

    # --- 群 PFI ---
    gv, gvs = res["GROUP_var"]["importance_mean"], res["GROUP_var"]["importance_std"]
    gt, gts = res["GROUP_topic"]["importance_mean"], res["GROUP_topic"]["importance_std"]
    ax2.bar([0, 1], [gv, gt], yerr=[gvs, gts], color=[C_VAR, C_TOPIC],
            capsize=5, width=0.6, error_kw=dict(ecolor="#444", lw=1.2))
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["変数重なり群\n(6 特徴)", "話題群\n(4 特徴)"], fontsize=11)
    ax2.set_ylabel("群を丸ごと置換した時の\nRecall@K の低下", fontsize=11)
    ax2.axhline(base, color="#888", ls=":", lw=1.2)
    ax2.text(1.02, base, f"baseline\nR@K={base:.3f}", fontsize=8.5, va="center", color="#555")
    for xi, v, e in zip([0, 1], [gv, gt], [gvs, gts]):
        ax2.text(xi, v + e + 0.015, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(base, gv + gvs) * 1.12)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_title("(b) 群ごとの重要度＝機構", fontsize=12)

    fig.suptitle("順列特徴量重要度：モデルは変数重なりに依存し、話題にはほぼ依存しない",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    (ROOT / "figure").mkdir(exist_ok=True)
    png = ROOT / "figure" / "fig_pfi_importance.png"
    pdf = ROOT / "figure" / "fig_pfi_importance.pdf"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png} + {pdf}")


if __name__ == "__main__":
    main()
