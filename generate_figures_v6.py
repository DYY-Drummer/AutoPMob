"""進捗報告 v6 用の図セットを生成（機械学習研究者視点）.

各図は主張を1つに絞り，シード分散のエラーバー・差分注記・統一カラーで提示する。
データはすべて experiments/ の実験 JSON と training_cases.json から読み込む。

  図A  fig_v6_dataset.png       : データセット内訳（variant）＋ 難易度分布（正解式数）
  図B  fig_v6_gradient.png      : 難易度勾配（baseline vs reranker-10S，R@K_correct）★中核
  図C  fig_v6_impact.png        : 学習への影響（正解式数別の改善 ＋ 指標飽和の対比）
  図D  fig_v6_llm.png           : 外部比較（LLM 直接生成 vs baseline vs reranker-10S）

保存先: docs/figures/
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# 日本語フォント
_fp = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
if os.path.exists(_fp):
    font_manager.fontManager.addfont(_fp)
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.axisbelow"] = True

# 統一カラー
C_BASE = "#999999"   # 比較対象（baseline / Stage1）
C_OURS = "#4A90E2"   # 本手法（reranker-10S / Stage2）
C_LLM = "#E27D60"    # LLM 直接生成
C_DAE = "#16A085"    # DAE 難ケース強調
C_GAP = "#D64550"    # ギャップ注記


def _load(name):
    return json.load(open(EXP / name))


def _bar_labels(ax, bars, fmt="{:.3f}", dy=0.008, fs=9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs, fontweight="bold")


# ============================================================
# 図A：データセット内訳 ＋ 難易度分布
# ============================================================
def fig_dataset():
    cases = json.load(open(ROOT / "training_cases.json"))
    vt = Counter(c.get("variant_type", "?") for c in cases)
    # variant をファミリに集約
    fam = {
        "original / 言い換え\n(単一・原型)": vt["original"] + vt["context_paraphrased"],
        "random_io\n(単一・構成的)": vt["random_io_from_models"],
        "multisource\n(文献横断)": vt["multisource_original"] + vt["multisource_random_io"] + vt["multisource_v3"],
        "DAE\n(微分代数・難)": sum(v for k, v in vt.items() if k.startswith("dae_")),
    }
    labels = list(fam.keys())
    vals = list(fam.values())
    colors = [C_BASE, C_BASE, C_OURS, C_DAE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) variant 内訳（横棒）
    ypos = np.arange(len(labels))
    bars = ax1.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    for b, v in zip(bars, vals):
        ax1.text(v + 12, b.get_y() + b.get_height() / 2, f"{v:,}",
                 va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("ケース数", fontsize=12)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.set_title(f"(a) 訓練ケースの内訳（全 {len(cases):,} 件）\n"
                  "v5: 1,107 件 → v6: 2,838 件（うち DAE 1,000 件を追加）", fontsize=12)

    # (b) 正解式数（難易度）分布
    nc = Counter(len(c.get("correct_model_ids", [])) for c in cases)
    ks = sorted(k for k in nc if k >= 1)
    vs = [nc[k] for k in ks]
    multi_ratio = sum(v for k, v in zip(ks, vs) if k >= 2) / sum(vs)
    barcolors = [C_BASE if k == 1 else C_OURS for k in ks]
    bars = ax2.bar(ks, vs, color=barcolors, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vs):
        if v >= 20:
            ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}",
                     ha="center", fontsize=8.5)
    ax2.set_xlabel("正解モデルに含まれる数式の数（問題の難しさ）", fontsize=12)
    ax2.set_ylabel("ケース数", fontsize=12)
    ax2.set_xticks(ks)
    ax2.set_title(f"(b) 正解式数の分布\n複数式（2 式以上）が {multi_ratio*100:.0f}%（青）— 本手法の主戦場", fontsize=12)
    # 凡例
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=C_BASE, edgecolor="black", label="単一式"),
                        Patch(facecolor=C_OURS, edgecolor="black", label="複数式")],
               fontsize=10, loc="upper right")

    fig.suptitle("図A：データセットの規模・内訳と難易度分布", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "fig_v6_dataset.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_v6_dataset.png  multi_ratio=%.3f" % multi_ratio)


# ============================================================
# 図B：難易度勾配（中核図）
# ============================================================
def _rkc(d):
    """results dict から Recall@K_correct(=Precision@C) の mean/std を取り出す."""
    for key in ("Recall@K_correct", "Precision@C"):
        if key in d and isinstance(d[key], dict):
            return d[key]["mean"], d[key]["std"]
    return None, None


def fig_gradient():
    broad = _load("set_aware_results.json")["results"]
    a = _load("phase3_A_original_multi_dae.json")["results"]
    b = _load("phase3_B_dae_only.json")["results"]

    regimes = ["(i) 広域混在\n(DAE 追加前)", "(ii) original+\nmulti_v3+DAE", "(iii) DAE のみ\n(最難)"]
    base = [_rkc(broad["baseline"]), _rkc(a["baseline"]), _rkc(b["baseline"])]
    ours = [_rkc(broad["reranker-10S"]), _rkc(a["reranker-10S"]), _rkc(b["reranker-10S"])]
    base_m = [x[0] for x in base]; base_s = [x[1] for x in base]
    ours_m = [x[0] for x in ours]; ours_s = [x[1] for x in ours]

    x = np.arange(len(regimes)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    b1 = ax.bar(x - w/2, base_m, w, yerr=base_s, capsize=4, color=C_BASE,
                edgecolor="black", linewidth=0.8, label="baseline (古典 IR, Stage1)",
                error_kw=dict(ecolor="#555", lw=1.2))
    b2 = ax.bar(x + w/2, ours_m, w, yerr=ours_s, capsize=4, color=C_OURS,
                edgecolor="black", linewidth=0.8, label="reranker-10S (本手法, Stage2)",
                error_kw=dict(ecolor="#1a3a5c", lw=1.2))
    _bar_labels(ax, b1, dy=0.012, fs=9)
    _bar_labels(ax, b2, dy=0.012, fs=9)

    # ギャップ注記
    for i in range(len(regimes)):
        gap = ours_m[i] - base_m[i]
        ytop = max(base_m[i] + base_s[i], ours_m[i] + ours_s[i]) + 0.06
        ax.annotate("", xy=(x[i] + w/2, ytop), xytext=(x[i] - w/2, ytop),
                    arrowprops=dict(arrowstyle="<->", color=C_GAP, lw=1.5))
        ax.text(x[i], ytop + 0.012, f"+{gap:.3f}", ha="center", color=C_GAP,
                fontsize=12, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=11)
    ax.set_ylabel("Recall@K$_{correct}$（正解集合の再現率）", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title("図B：難易度勾配 — 難ケースが増えるほど本手法の優位が拡大\n"
                 "（10 シード平均，エラーバー=標準偏差）", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v6_gradient.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_v6_gradient.png  gaps=",
          [round(o - b, 3) for o, b in zip(ours_m, base_m)])


# ============================================================
# 図C：学習への影響（正解式数別の改善 ＋ 指標飽和の対比）
# ============================================================
def fig_impact():
    ch = _load("phase4_characteristic_analysis.json")
    nc = ch["axes"]["n_correct"]["buckets"]
    order = ch["axes"]["n_correct"]["order"]
    xs = order
    base = [nc[k]["baseline"]["Recall@K_correct"] for k in order]
    ours = [nc[k]["reranker-10S"]["Recall@K_correct"] for k in order]

    hb = _load("phase4_hypothesis_b_test.json")["metrics"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

    # (a) R@K_c vs 正解式数
    xi = np.arange(len(xs))
    ax1.plot(xi, base, "o-", color=C_BASE, lw=2.2, ms=8, label="baseline")
    ax1.plot(xi, ours, "s-", color=C_OURS, lw=2.4, ms=8, label="reranker-10S")
    ax1.fill_between(xi, base, ours, color=C_OURS, alpha=0.15)
    for i in range(len(xs)):
        g = ours[i] - base[i]
        ax1.annotate(f"+{g:.2f}", (xi[i], (base[i] + ours[i]) / 2),
                     ha="center", va="center", fontsize=9.5, color=C_GAP, fontweight="bold")
    ax1.set_xticks(xi); ax1.set_xticklabels(xs)
    ax1.set_xlabel("正解式数 $n_{correct}$（問題の複雑さ）", fontsize=12)
    ax1.set_ylabel("Recall@K$_{correct}$", fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("(a) 改善は複数式・複雑な問題ほど大きい\n"
                  "単一式では差なし／複数式で +0.21〜0.26", fontsize=12)
    ax1.legend(fontsize=11, loc="lower left"); ax1.grid(alpha=0.3)

    # (b) 指標飽和の対比（仮説B：Full vs Narrow DB）
    metrics = ["MRR_first", "Recall@K_correct"]
    mlabels = ["MRR_first\n(飽和・差が見えない)", "Recall@K$_{correct}$\n(飽和せず・差が見える)"]
    narrow = [hb[m]["narrow_mean"] for m in metrics]
    full = [hb[m]["full_mean"] for m in metrics]
    pvals = [hb[m]["paired_t"]["p_value"] for m in metrics]
    xi2 = np.arange(len(metrics)); w = 0.36
    bn = ax2.bar(xi2 - w/2, narrow, w, color=C_BASE, edgecolor="black",
                 linewidth=0.8, label="限定 DB（化学工学のみ）")
    bf = ax2.bar(xi2 + w/2, full, w, color=C_OURS, edgecolor="black",
                 linewidth=0.8, label="フル DB（11,146 式・全分野）")
    _bar_labels(ax2, bn, dy=0.006, fs=9); _bar_labels(ax2, bf, dy=0.006, fs=9)
    for i, m in enumerate(metrics):
        diff = full[i] - narrow[i]
        sig = "有意 $p$=%.3f$^\\star$" % pvals[i] if pvals[i] < 0.05 else "n.s. $p$=%.2f" % pvals[i]
        yann = 1.16  # 注記は上端に固定（バー値ラベルと重ねない）
        ax2.annotate("", xy=(xi2[i] + w/2, max(narrow[i], full[i]) + 0.02),
                     xytext=(xi2[i] - w/2, max(narrow[i], full[i]) + 0.02),
                     arrowprops=dict(arrowstyle="<->",
                                     color=(C_GAP if pvals[i] < 0.05 else "#888"), lw=1.2))
        ax2.text(xi2[i], yann, f"差 +{diff:.3f}\n{sig}", ha="center", va="top",
                 fontsize=10, color=(C_GAP if pvals[i] < 0.05 else "#555"), fontweight="bold")
    ax2.set_xticks(xi2); ax2.set_xticklabels(mlabels, fontsize=10.5)
    ax2.set_ylabel("評点（10 シード平均）", fontsize=12)
    ax2.set_ylim(0, 1.22)
    ax2.set_title("(b) なぜ評価指標を変えたか（DB 規模の効果）\n"
                  "飽和した MRR_first は差を隠す", fontsize=12)
    ax2.legend(fontsize=10, loc="lower center"); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("図C：本手法はどこで効くか／なぜ Recall@K$_{correct}$ を主指標にしたか",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "fig_v6_impact.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_v6_impact.png")


# ============================================================
# 図D：外部比較（LLM 直接生成 vs baseline vs reranker-10S）
# ============================================================
def fig_llm():
    cmp = _load("phase6_llm_subset_comparison.json")["subsets"]
    subsets = [("single_29", "単一ソース\n(29 件)"), ("multi_20", "複数ソース\n(20 件)")]
    methods = [("LLM_direct", "LLM 直接生成", C_LLM),
               ("baseline", "baseline (Stage1)", C_BASE),
               ("reranker-10S", "reranker-10S (Stage2)", C_OURS)]

    x = np.arange(len(subsets)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for j, (mkey, mlab, col) in enumerate(methods):
        vals = [cmp[s][mkey]["Recall@K_correct"] for s, _ in subsets]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=col, edgecolor="black",
                      linewidth=0.8, label=mlab)
        _bar_labels(ax, bars, dy=0.008, fs=9.5)

    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in subsets], fontsize=11)
    ax.set_ylabel("Recall@K$_{correct}$", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title("図D：外部ベースライン（LLM 直接生成）との同一条件比較\n"
                 "現行 11,146 式 DB・held-out（複数ソースで baseline < LLM < 本手法）",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.3)

    # 複数ソースの逆転を注記
    ax.annotate("古典 IR は LLM 未満\n→ 本手法で逆転", xy=(1 - w, cmp["multi_20"]["baseline"]["Recall@K_correct"]),
                xytext=(1.15, 0.40), fontsize=9.5, color=C_GAP, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GAP, lw=1.3), ha="center")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v6_llm.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_v6_llm.png")


if __name__ == "__main__":
    fig_dataset()
    fig_gradient()
    fig_impact()
    fig_llm()
    print("done -> docs/figures/fig_v6_*.png")
