#!/usr/bin/env python3
"""層化分割実験（strat_A / strat_B）の結果図と特性別分析を生成する。

- strat_A.json: variant = original / multisource_ / dae_（1,823 件, 設定 A）
- strat_B.json: variant = dae_ のみ（1,000 件, 設定 B）
両者とも 10 シード・ソース単位の層化分割・top_k=50。
本スクリプトは
  (1) 設定 A / B の全体指標（表用に標準出力へ）
  (2) strat_A の per_case から R@K の特性別分析（数式数・ソース数・入力変数数）
を計算し，docs/figures/fig_strat_characteristic.png を出力する。
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

_fp = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
if os.path.exists(_fp):
    font_manager.fontManager.addfont(_fp)
    matplotlib.rcParams["font.family"] = "Hiragino Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType埋め込み（和文をPDFに出すため）

C_BASE = "#999999"   # baseline（古典 IR, Stage1）
C_OURS = "#4A90E2"   # reranker-10S（本手法, Stage2）
C_LLM = "#E27D60"    # LLM 直接生成（外部比較）
C_DAE = "#16A085"    # DAE 難ケース強調
C_GAP = "#D64550"


def load(name):
    return json.load(open(EXP / name))


# ----------------------------------------------------------------------
# (0) データセットの内訳と難易度分布（ゼミ報告用：前回ゼミ 1,338 → 今回 2,838）
# ----------------------------------------------------------------------
def dataset():
    from collections import Counter
    cases = json.load(open(ROOT / "training_cases.json"))
    vt = Counter(c.get("variant_type", "?") for c in cases)
    fam = {
        "原型": vt["original"],
        "言い換え": vt["context_paraphrased"],
        "無作為入替": vt["random_io_from_models"] + vt["swap_io"],
        "複数文献": vt["multisource_original"] + vt["multisource_random_io"] + vt["multisource_v3"],
        "DAE\n（微分代数・難）": sum(v for k, v in vt.items() if k.startswith("dae_")),
    }
    labels, vals = list(fam.keys()), list(fam.values())
    # 青＋緑＝評価の設定 A（original+multisource+DAE）, 灰＝拡張のため設定 A から除外
    colors = [C_OURS, C_BASE, C_BASE, C_OURS, C_DAE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ypos = np.arange(len(labels))
    bars = ax1.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_yticks(ypos); ax1.set_yticklabels(labels, fontsize=10); ax1.invert_yaxis()
    for b, v in zip(bars, vals):
        ax1.text(v + 12, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("ケース数", fontsize=12)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.set_title(f"(a) 訓練ケースの内訳（全 {len(cases):,} 件）\n"
                  "前回ゼミ 1,338 件 → 今回 2,838 件（DAE 1,000 件を新規追加）", fontsize=12)
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor=C_OURS, edgecolor="black", label="設定 A：評価対象（計 1,823 件）"),
                        Patch(facecolor=C_DAE, edgecolor="black", label="└ うち DAE（1,000 件）"),
                        Patch(facecolor=C_BASE, edgecolor="black", label="拡張：設定 A から除外")],
               fontsize=9, loc="upper right")

    nc = Counter(len(c.get("correct_model_ids", [])) for c in cases)
    ks = sorted(k for k in nc if k >= 1); vs = [nc[k] for k in ks]
    multi_ratio = sum(v for k, v in zip(ks, vs) if k >= 2) / sum(vs)
    barcolors = [C_BASE if k == 1 else C_OURS for k in ks]
    bars = ax2.bar(ks, vs, color=barcolors, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vs):
        if v >= 20:
            ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}", ha="center", fontsize=8.5)
    ax2.set_xlabel("正解モデルに含まれる数式の数", fontsize=12)
    ax2.set_ylabel("ケース数", fontsize=12); ax2.set_xticks(ks)
    ax2.set_title(f"(b) 正解式数（問題の難しさ）の分布\n複数式（2 式以上）が {multi_ratio*100:.1f}%（青）", fontsize=12)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=C_BASE, edgecolor="black", label="単一式"),
                        Patch(facecolor=C_OURS, edgecolor="black", label="複数式")],
               fontsize=10, loc="upper right")
    fig.suptitle("データベースの拡大：規模・内訳・難易度", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "fig_strat_dataset.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(FIG / "fig_strat_dataset.pdf", bbox_inches="tight")
    print(f"saved {out}  multi_ratio={multi_ratio:.3f}")


# ----------------------------------------------------------------------
# (1) 全体指標（設定 A / B）
# ----------------------------------------------------------------------
def overall():
    print("=" * 70)
    print("(1) 全体指標  R@K / MAP / Recall@20 / MRR_first")
    print("=" * 70)
    for tag, fn, n in [("全データ (拡張込み, 2838)", "strat_full.json", 2838),
                       ("設定A (orig+multi+DAE, 1823)", "strat_A.json", 1823),
                       ("設定B (DAEのみ, 1000)", "strat_B.json", 1000)]:
        d = load(fn)
        print(f"\n--- {tag} ---")
        for mode in ["baseline", "reranker-10S"]:
            m = d["results"][mode]
            g = lambda k: (m[k]["mean"], m[k]["std"])
            rk = g("Recall@K_correct"); mp = g("MAP")
            r20 = g("Recall@20"); mf = g("MRR_first")
            print(f"  {mode:14s} R@K={rk[0]:.3f}±{rk[1]:.3f}  "
                  f"MAP={mp[0]:.3f}  R@20={r20[0]:.3f}  MRR_first={mf[0]:.3f}")
        b = d["results"]["baseline"]["Recall@K_correct"]["mean"]
        r = d["results"]["reranker-10S"]["Recall@K_correct"]["mean"]
        print(f"  => R@K の差（reranker - baseline）= +{r-b:.3f}")


# ----------------------------------------------------------------------
# (2) 特性別分析（strat_A per_case）
# ----------------------------------------------------------------------
def bucket_stats(per_case_by_mode, keyfn, order):
    """keyfn(entry) -> bucket label. return {bucket: {mode: mean R@K, n}}"""
    out = {}
    # paired delta: per case_id 平均してから対応差
    for b in order:
        out[b] = {}
    for mode, entries in per_case_by_mode.items():
        agg = defaultdict(list)
        for e in entries:
            b = keyfn(e)
            if b is not None:
                agg[b].append(e["Recall@K_correct"])
        for b in order:
            vals = agg.get(b, [])
            out[b][mode] = (float(np.mean(vals)) if vals else float("nan"), len(vals))
    return out


def characteristic():
    d = load("strat_A.json")
    pcm = {mode: d["results"][mode]["per_case"] for mode in d["results"]}

    print("\n" + "=" * 70)
    print("(2) 特性別 R@K（strat_A, 設定 A, per_case をシード横断で平均）")
    print("=" * 70)

    # --- 数式数 n_correct: 1..10 ---
    ncs = list(range(1, 11))
    nc = bucket_stats(pcm, lambda e: e["n_correct"] if 1 <= e["n_correct"] <= 10 else (10 if e["n_correct"] > 10 else None), ncs)
    print("\n[数式数 n_correct]")
    nc_base, nc_ours = [], []
    for k in ncs:
        b = nc[k]["baseline"][0]; r = nc[k]["reranker-10S"][0]
        n = nc[k]["baseline"][1]
        nc_base.append(b); nc_ours.append(r)
        print(f"  {k:>3}式 (n={n:4d})  base={b:.3f}  ours={r:.3f}  Δ=+{r-b:.3f}")

    # --- ソース数 n_sources: 1,2,>=3 ---
    def srcbucket(e):
        s = e["n_sources"]
        if s == 1: return "1"
        if s == 2: return "2"
        return "3+"
    sorder = ["1", "2", "3+"]
    sc = bucket_stats(pcm, srcbucket, sorder)
    print("\n[横断ソース数 n_sources]")
    sc_base, sc_ours = [], []
    for k in sorder:
        b = sc[k]["baseline"][0]; r = sc[k]["reranker-10S"][0]; n = sc[k]["baseline"][1]
        sc_base.append(b); sc_ours.append(r)
        print(f"  {k:>3}源 (n={n:4d})  base={b:.3f}  ours={r:.3f}  Δ=+{r-b:.3f}")

    # --- 入力変数数 n_input: <=5,6-10,11-15,16-20,21+ ---
    def inbucket(e):
        v = e["n_input"]
        if v <= 5: return "5 以下"
        if v <= 10: return "6-10"
        if v <= 15: return "11-15"
        if v <= 20: return "16-20"
        return "21 以上"
    iorder = ["5 以下", "6-10", "11-15", "16-20", "21 以上"]
    ic = bucket_stats(pcm, inbucket, iorder)
    print("\n[入力変数数 n_input]")
    ic_base, ic_ours = [], []
    for k in iorder:
        b = ic[k]["baseline"][0]; r = ic[k]["reranker-10S"][0]; n = ic[k]["baseline"][1]
        ic_base.append(b); ic_ours.append(r)
        print(f"  {k:>5} (n={n:4d})  base={b:.3f}  ours={r:.3f}  Δ=+{r-b:.3f}")

    # ---------------- 図（3 パネル） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) 数式数
    ax = axes[0]
    ax.plot(ncs, nc_base, "o-", color=C_BASE, lw=2.2, ms=7, label="baseline（古典 IR）")
    ax.plot(ncs, nc_ours, "s-", color=C_OURS, lw=2.4, ms=7, label="reranker-10S（本手法）")
    ax.fill_between(ncs, nc_base, nc_ours, color=C_OURS, alpha=0.15)
    ax.set_xlabel("正解モデルに含まれる数式の数", fontsize=12)
    ax.set_ylabel("Recall@K（正解集合の再現率）", fontsize=12)
    ax.set_xticks(ncs)
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) 数式が多いほど難しくなる", fontsize=12)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) ソース数
    ax = axes[1]
    x = np.arange(len(sorder)); w = 0.38
    ax.bar(x - w / 2, sc_base, w, color=C_BASE, edgecolor="black", linewidth=0.8, label="baseline")
    ax.bar(x + w / 2, sc_ours, w, color=C_OURS, edgecolor="black", linewidth=0.8, label="reranker-10S")
    for i in range(len(sorder)):
        ax.text(x[i] - w / 2, sc_base[i] + 0.012, f"{sc_base[i]:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, sc_ours[i] + 0.012, f"{sc_ours[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["1 つ", "2 つ", "3 つ以上"], fontsize=11)
    ax.set_xlabel("数式が由来する文献（ソース）の数", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) ソース数による差は小さい（優位は一定）", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # (c) 入力変数数
    ax = axes[2]
    x = np.arange(len(iorder))
    ax.plot(x, ic_base, "o-", color=C_BASE, lw=2.2, ms=7, label="baseline")
    ax.plot(x, ic_ours, "s-", color=C_OURS, lw=2.4, ms=7, label="reranker-10S")
    ax.fill_between(x, ic_base, ic_ours, color=C_OURS, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(iorder, fontsize=11)
    ax.set_xlabel("入力変数の数", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) 入力変数が多いほど難しくなる", fontsize=12)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle("特性別の Recall@K：問題が複雑になるほど baseline は落ちるが，本手法は差を広げて持ちこたえる",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "fig_strat_characteristic.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    fig.savefig(FIG / "fig_strat_characteristic.pdf", bbox_inches="tight")
    print(f"\nsaved {out}")


# ----------------------------------------------------------------------
# (3) データ拡張の影響：全データ／設定A／設定B の差（gap）比較
# ----------------------------------------------------------------------
def augmentation():
    # strat（baseline/reranker, 10シード全件）＋ LLM直接生成（各設定の一様サンプル・等価判定）
    files = [("全データ\n（拡張込み, 2,838）", "strat_full.json", "llm_set_full_equiv_results.json"),
             ("設定 A\n（拡張除外, 1,823）", "strat_A.json", "llm_set_A_equiv_results.json"),
             ("設定 B\n（DAE のみ, 1,000）", "strat_B.json", "llm_set_B_equiv_results.json")]
    base_m, base_s, ours_m, ours_s, llm_m, llm_cov, llm_n, gaps = [], [], [], [], [], [], [], []
    for _, strat_fn, llm_fn in files:
        d = load(strat_fn)
        b = d["results"]["baseline"]["Recall@K_correct"]
        r = d["results"]["reranker-10S"]["Recall@K_correct"]
        base_m.append(b["mean"]); base_s.append(b["std"])
        ours_m.append(r["mean"]); ours_s.append(r["std"])
        gaps.append(r["mean"] - b["mean"])
        ld = json.load(open(EXP / llm_fn))
        llm_m.append(ld["Recall@K_correct"]); llm_cov.append(ld.get("coverage", 0.0))
        llm_n.append(ld["n_cases"])

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = np.arange(3); w = 0.26
    ax.bar(x - w, base_m, w, yerr=base_s, capsize=3, color=C_BASE,
           edgecolor="black", linewidth=0.8, label="baseline（古典 IR）",
           error_kw=dict(ecolor="#555", lw=1.0))
    ax.bar(x, llm_m, w, color=C_LLM, edgecolor="black", linewidth=0.8,
           label="LLM 直接生成（外部比較・等価判定・サンプル）")
    ax.bar(x + w, ours_m, w, yerr=ours_s, capsize=3, color=C_OURS,
           edgecolor="black", linewidth=0.8, label="reranker-10S（本手法）",
           error_kw=dict(ecolor="#1a3a5c", lw=1.0))
    for i in range(3):
        ax.text(x[i] - w, base_m[i] + base_s[i] + 0.012, f"{base_m[i]:.3f}",
                ha="center", fontsize=8.5)
        ax.text(x[i], llm_m[i] + 0.012, f"{llm_m[i]:.3f}",
                ha="center", fontsize=8.5, color="#9c4a2f")
        ax.text(x[i] + w, ours_m[i] + ours_s[i] + 0.012, f"{ours_m[i]:.3f}",
                ha="center", fontsize=8.5, fontweight="bold")
        ytop = max(base_m[i] + base_s[i], ours_m[i] + ours_s[i]) + 0.065
        ax.text(x[i] + w / 2, ytop, f"差 +{gaps[i]:.3f}", ha="center", color=C_GAP,
                fontsize=10, fontweight="bold")
    # LLM の coverage（順位を問わず等価式を生成できた割合）を破線で重ねる
    for i in range(3):
        lbl = "LLM coverage（順位不問）" if i == 0 else None
        ax.plot([x[i] - w * 0.55, x[i] + w * 0.55], [llm_cov[i]] * 2,
                color="#7a2e18", lw=1.6, ls=(0, (3, 2)), zorder=6, label=lbl)
        ax.text(x[i] + w * 0.6, llm_cov[i], f"cov {llm_cov[i]:.2f}", va="center",
                ha="left", fontsize=7.5, color="#7a2e18")
    ax.set_xticks(x); ax.set_xticklabels([t for t, _, _ in files], fontsize=11)
    ax.set_ylabel("Recall@K（正解集合の再現率）", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("3 手法の比較：reranker（本手法）＞ baseline（古典 IR）＞ LLM 直接生成（上位K件）\n"
                 "破線=LLM coverage（順位を問わず等価式を生成できた割合）。拡張は baseline を水増しし手法差を縮める",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    out = FIG / "fig_strat_augmentation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(FIG / "fig_strat_augmentation.pdf", bbox_inches="tight")
    print(f"saved {out}  gaps={[round(g,3) for g in gaps]}  "
          f"LLM R@K={[round(v,3) for v in llm_m]} (n={llm_n})")


if __name__ == "__main__":
    dataset()
    overall()
    augmentation()
    characteristic()
