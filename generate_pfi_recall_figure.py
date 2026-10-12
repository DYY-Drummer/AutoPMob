"""PFI 結果図（改訂版）：シャッフル後の Recall@K を 1 枚にまとめる.

加藤先生の 8/31 報告への注釈（2026-09-25）に応えて、旧図 fig_pfi_importance の 2 枚組
（(a) 個別の低下量、(b) 群の低下量）を 1 枚にまとめる。
  - 縦に「変数の重複率を測る 6 特徴量」と「話題の近さを測る 4 特徴量」の 2 群を並べ、
    各群の先頭に「その群をまとめてシャッフル」した棒を置く（単一特徴量と比べられる）。
  - 横軸は低下量でなく、シャッフル後の Recall@K そのもの。シャッフル前の値は縦の点線で示す。
入力: experiments/pfi_results.json（scope=case: ケース内で候補どうしの値を入れ替え、20 回平均、10 seed）。
棒 = シャッフル後の Recall@K の 10 seed 平均（= シャッフル前の平均 − 低下量の平均）。
誤差棒 = 低下量の 10 seed 間の標準偏差。
出力: docs/figures/fig_pfi_recall_ja.{png,pdf} と docs/ゼミlatexテンプレ/figure/fig_pfi_recall_ja.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parent
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

JP = {  # 改訂版報告の用語（プログラム識別子は使わない）
    "text_sim": "文章類似度", "io_jaccard": "変数の一致度", "svd_sim": "潜在意味類似度",
    "input_cov": "入力変数の被覆率", "output_cov": "出力変数の被覆率", "specificity": "式の特化度",
    "domain": "分野ラベルの一致", "gComp": "補完性", "gCoh": "一貫性",
    "gDom": "分野ラベルの一致度（集合）",
}
VAR = ["io_jaccard", "input_cov", "output_cov", "specificity", "gComp", "gCoh"]
TOPIC = ["text_sim", "svd_sim", "domain", "gDom"]
C_VAR, C_VAR_G = "#6baed6", "#08519c"
C_TOP, C_TOP_G = "#fdae6b", "#a63603"


def main():
    d = json.load(open(ROOT / "experiments" / "pfi_results.json"))
    res = d["results"]["case"]
    base = d["baseline_mean"]

    rows = []  # (label, value, err, color, is_group)
    for grp, feats, gkey, c, cg, gname in (
        ("var", VAR, "GROUP_var", C_VAR, C_VAR_G, "変数の重複率を測る 6 特徴量をまとめて"),
        ("topic", TOPIC, "GROUP_topic", C_TOP, C_TOP_G, "話題の近さを測る 4 特徴量をまとめて"),
    ):
        g = res[gkey]
        rows.append((gname, base - g["importance_mean"], g["importance_std"], cg, True))
        for f in sorted(feats, key=lambda f: -res[f]["importance_mean"]):
            r = res[f]
            rows.append(("  " + JP[f], base - r["importance_mean"], r["importance_std"], c, False))
        rows.append(None)  # 群の間の空き
    rows = rows[:-1]

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    y, ticks, labels = 0, [], []
    for row in rows:
        if row is None:
            y += 0.6
            continue
        lab, v, e, col, is_g = row
        ax.barh(y, v, xerr=e, color=col, height=0.72, capsize=3,
                error_kw=dict(ecolor="#333", lw=1), edgecolor="#222" if is_g else col,
                linewidth=1.2 if is_g else 0)
        ax.text(min(v + e, base) + 0.012 if v < base - 0.05 else base + 0.012, y,
                f"{v:.3f}", va="center", fontsize=11, fontweight="bold" if is_g else "normal")
        ticks.append(y)
        labels.append(lab)
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if not lab.startswith("  "):
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.axvline(base, color="#555", ls=":", lw=1.5)
    ax.text(base - 0.01, -1.05, f"シャッフル前 {base:.3f}", ha="right", va="center", fontsize=11, color="#333")
    ax.set_xlim(0, 0.86)
    ax.set_ylim(y - 0.4, -1.6)
    ax.set_xlabel("シャッフル後の Recall@K（設定 A、乱数 10 通りの平均）", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[Patch(color=C_VAR, label="変数の重複率を測る特徴量"),
                       Patch(color=C_TOP, label="話題の近さを測る特徴量")],
              fontsize=11, loc="upper left", bbox_to_anchor=(0.36, 0.93), framealpha=1.0)
    fig.tight_layout()

    outs = [ROOT / "docs" / "figures" / "fig_pfi_recall_ja.png",
            ROOT / "docs" / "figures" / "fig_pfi_recall_ja.pdf",
            ROOT / "docs" / "ゼミlatexテンプレ" / "figure" / "fig_pfi_recall_ja.pdf"]
    for p in outs:
        fig.savefig(p, dpi=200 if p.suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
