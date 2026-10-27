#!/usr/bin/env python3
"""修士論文用の英語ラベル図を一括生成する.

出力: thesis/master_thesis/figures/*.{png,pdf}
データ（2026-10-27 改訂。説明文を書き直し core_019_v1 を外した再計算 8b77e73 の結果。日本語版 8/31 報告 8a0de85 と同じ値）:
    training_cases.json / experiments/kato1020/excluded_cases_v3.json（評価と訓練から外した 92 件。2026-10-21(2) から）/
    experiments/kato1021/main.json（全体比較、正解式の数と入力変数の数で分けた値。2026-10-27 から、説明文 113 件を書き直し
    core_019_v1 を外した 1,731 件の再計算 8b77e73）/ mech.json（特徴量を足したときの上乗せ）/
    pfi.json / greedy.json / n1_coverage_w30-70.json
方針: 10 通りの分割の比較は箱ひげ図（G7）。日本語フォント設定は入れない（英語ラベルのみ）。suptitle は付けない。
2026-10-21: 自由度ゼロ停止（fig_dof_stop）と第 1 段の重みの再設計（fig_stage1_redesign）の図は、報告書と同じく
    削除した。LLM 直接生成の点も削除した（DB を渡す新しい LLM との比較は未実行）。話題の特徴量の診断
    （fig_topic_diagnosis、diagnose_topic_features.py）は、報告書と同じく表（AUC）に置き換えた。
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
K20 = EXP / "kato1020"   # 除外の一覧（excluded_cases_v3.json）
K21 = EXP / "kato1021"   # 結果（2026-10-27 の再計算。設定 A 1,731 件）
FIG = ROOT / "thesis" / "master_thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["pdf.fonttype"] = 42

C_BASE = "#9aa0a6"   # baseline: gray
C_PROP = "#1a73e8"   # proposed (static reranking): blue
C_INFER = "#f9ab00"  # greedy selection at inference only: amber
C_TRAIN = "#188038"  # learned greedy: green
C_DAE = "#16A085"    # synthesized cases (Setting B)
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / name}.png/.pdf")


def _box(ax, data, positions, colors, widths=0.55):
    """10 分割の値の箱ひげ＋ジッター散点（fig_v6_boxplot のスタイル踏襲）."""
    bp = ax.boxplot(data, positions=positions, widths=widths, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, data):
        x = rng.normal(pos, 0.05, size=len(vals))
        ax.scatter(x, vals, s=12, color="black", alpha=0.5, zorder=3)
    return bp


def _excluded_ids():
    """評価と訓練から外すケース（v3、92 件）: 未知の変数が決まらない出典横断の 91 件（kato1020_exclusions.py、
    analyze_kato_0831_followup.gt_checks と同じ判定）と、2 冊の教科書の式から成る単一出典の core_019_v1（2026-10-21 の本人の決定）.

    結果の図（method_comparison など）は、同じ 92 件を外した 1,731 件の再計算（experiments/kato1021/）の JSON を読む.
    """
    return set(json.load(open(K20 / "excluded_cases_v3.json"))["excluded_case_ids"])


def _family(c):
    v = c.get("variant_type", "")
    if v == "original":
        return "single"
    if v.startswith("multisource_"):
        return "cross"
    if v.startswith("dae_"):
        return "synth"
    return "aug"


# ----------------------------------------------------------------------
# dataset() — 評価に使う 1,731 件と、使わない 92 件・拡張 1,015 件
# ----------------------------------------------------------------------
def dataset():
    cases = json.load(open(ROOT / "training_cases.json"))
    ex = _excluded_ids()
    used, dropped = Counter(), Counter()
    for c in cases:
        (dropped if c["case_id"] in ex else used)[_family(c)] += 1
    keys = ["single", "cross", "synth", "aug"]
    labels = ["Single-source", "Cross-source", "Synthesized", "Augmented"]
    colors = [C_PROP, C_PROP, C_DAE, "white"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    ypos = np.arange(len(keys))
    for y, k, col in zip(ypos, keys, colors):
        v = used[k]
        hatch = "//" if k == "aug" else None
        ax1.barh(y, v, color=col, edgecolor="black", linewidth=0.8, hatch=hatch)
        if dropped[k]:
            ax1.barh(y, dropped[k], left=v, color="white", edgecolor="black", linewidth=0.8, hatch="xx")
        tot = v + dropped[k]
        txt = f"{v:,}" if not dropped[k] else f"{v:,} + {dropped[k]:,} excluded"
        ax1.text(tot + 15, y, txt, va="center", fontsize=11, fontweight="bold")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of cases", fontsize=12)
    ax1.set_xlim(0, max(used[k] + dropped[k] for k in keys) * 1.45)
    ax1.set_title("(a) Composition of the cases", fontsize=12)
    n_eval = sum(used[k] for k in ("single", "cross", "synth"))
    ax1.legend(handles=[Patch(facecolor=C_PROP, edgecolor="black", label=f"Setting A: evaluation ({n_eval:,} cases)"),
                        Patch(facecolor=C_DAE, edgecolor="black", label=f"  of which synthesized = Setting B ({used['synth']:,})"),
                        Patch(facecolor="white", edgecolor="black", hatch="xx",
                              label=f"Excluded from training and evaluation ({sum(dropped.values()):,})"),
                        Patch(facecolor="white", edgecolor="black", hatch="//",
                              label=f"Augmented: used for neither training nor testing ({used['aug']:,})")],
               fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, framealpha=1.0)

    ev = [c for c in cases if c["case_id"] not in ex and _family(c) != "aug"]
    nc = Counter(len(c.get("correct_model_ids", [])) for c in ev)
    ks = sorted(k for k in nc if k >= 1)
    vs = [nc[k] for k in ks]
    bars = ax2.bar(ks, vs, color=[C_BASE if k == 1 else C_PROP for k in ks], edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}", ha="center", fontsize=8.5)
    ax2.set_xlabel("Number of ground-truth equations", fontsize=12)
    ax2.set_ylabel("Number of cases", fontsize=12)
    ax2.set_xticks(ks)
    ax2.set_title(f"(b) Ground-truth size in Setting A ({len(ev):,} cases)", fontsize=12)
    ax2.legend(handles=[Patch(facecolor=C_BASE, edgecolor="black", label="One equation"),
                        Patch(facecolor=C_PROP, edgecolor="black", label="Two or more equations")],
               fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_dataset")


# ----------------------------------------------------------------------
# split_balance() — ported from generate_figures_v6.py fig_split_balance() L586-635（本文では非表示）
# ----------------------------------------------------------------------
def split_balance(seed=42):
    import sys
    sys.path.insert(0, str(ROOT))
    from set_aware_reranker import stratified_src_split
    from two_stage_query_conditioned import load_cases, case_src, in_vars, out_vars
    cases = load_cases()
    ex = _excluded_ids()
    keep = lambda v: v == "original" or v.startswith("multisource_") or v.startswith("dae_")
    sub = [c for c in cases if keep(c.get("variant_type", "")) and c["case_id"] not in ex]
    csrc = [case_src(c) for c in sub]
    nc = [len(c.get("correct_model_ids") or []) for c in sub]
    nin = [len(in_vars(c)) for c in sub]
    nout = [len(out_vars(c)) for c in sub]
    nsrc = [len({m.split("__", 1)[0] for m in (c.get("correct_model_ids") or [])}) for c in sub]
    feats = list(zip(nc, nin, nout, nsrc))
    sp = stratified_src_split(csrc, feats, seed)
    tr = [i for i, s in enumerate(csrc) if s in sp["train"]]
    te = [i for i, s in enumerate(csrc) if s in sp["test"]]

    panels = [("#Equations", nc), ("#Input variables", nin), ("#Output variables", nout), ("#Sources spanned", nsrc)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, (name, vals) in zip(axes, panels):
        allv = sorted(set(vals))
        if len(allv) > 12:
            bins = np.linspace(min(vals), max(vals), 10)
            trh, _ = np.histogram([vals[i] for i in tr], bins=bins, density=True)
            teh, _ = np.histogram([vals[i] for i in te], bins=bins, density=True)
            x = (bins[:-1] + bins[1:]) / 2
            w = (bins[1] - bins[0]) * 0.4
            ax.bar(x - w / 2, trh, w, color=C_PROP, alpha=0.8, label="Train")
            ax.bar(x + w / 2, teh, w, color="#d93025", alpha=0.8, label="Test")
        else:
            ctr = Counter(vals[i] for i in tr)
            cte = Counter(vals[i] for i in te)
            ntr = sum(ctr.values())
            nte = sum(cte.values())
            x = np.arange(len(allv))
            w = 0.38
            ax.bar(x - w / 2, [ctr[v] / ntr for v in allv], w, color=C_PROP, alpha=0.85, label="Train")
            ax.bar(x + w / 2, [cte[v] / nte for v in allv], w, color="#d93025", alpha=0.85, label="Test")
            ax.set_xticks(x)
            ax.set_xticklabels(allv, fontsize=8)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Value", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Fraction", fontsize=11)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "fig_split_balance")
    print(f"split_balance: train/test = {len(tr)}/{len(te)}")


# ----------------------------------------------------------------------
# method_comparison() — 全体比較（Table 3 と同じ値。main.json table）
# ----------------------------------------------------------------------
MODES_MAIN = [("baseline_tfidf", "Baseline (classical IR)", C_BASE),
              ("proposed_static", "Proposed, static reranking", C_PROP),
              ("proposed_train", "Proposed, learned greedy", C_TRAIN)]


def method_comparison():
    tab = json.load(open(K21 / "main.json"))["table"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    centers = np.array([0.0, 3.4])
    for ax, metric, tag in ((axes[0], "Recall@20", "(a)"), (axes[1], "MAP", "(b)")):
        for i, S in enumerate(("A", "B")):
            data = [[tab[S][mode][metric]["per_seed"][str(s)] for s in SEEDS] for mode, _, _ in MODES_MAIN]
            _box(ax, data, [centers[i] - 0.9, centers[i], centers[i] + 0.9], [c for _, _, c in MODES_MAIN],
                 widths=0.62)
        ax.set_xticks(centers)
        ax.set_xticklabels(["Setting A\n(1,731 cases)", "Setting B\n(synthesized, 1,000 cases)"], fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{tag} {metric}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    fig.legend(handles=[Patch(facecolor=c, alpha=0.85, label=lab) for _, lab, c in MODES_MAIN],
               fontsize=10, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), framealpha=1.0)
    fig.tight_layout()
    save(fig, "fig_method_comparison")


# ----------------------------------------------------------------------
# characteristic() — 正解式の数と入力変数の数で分けた値（設定 A、10 分割のテストを合わせた延べ 4,233 件。main.json buckets）
# ----------------------------------------------------------------------
IN_ORDER = ["<=5", "6-10", "11-15", "16-20", ">=21"]
IN_LABELS = ["≤5", "6–10", "11–15", "16–20", "≥21"]
MARK = {"baseline_tfidf": "o-", "proposed_static": "s-", "proposed_train": "^-"}


def characteristic():
    b = json.load(open(K21 / "main.json"))["buckets"]["A"]
    nc_order = [str(k) for k in range(1, 11)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharey="row")
    for col, (key, order, ticklabels, xlabel) in enumerate((
            ("n_correct", nc_order, nc_order, "Number of ground-truth equations"),
            ("n_input", IN_ORDER, IN_LABELS, "Number of input variables"))):
        for row, metric in enumerate(("Recall@20", "MAP")):
            ax = axes[row, col]
            x = np.arange(len(order))
            for mode, lab, color in MODES_MAIN:
                ax.plot(x, [b[key][k][f"{mode}|{metric}"] for k in order], MARK[mode], color=color,
                        lw=2.0, ms=6, label=lab)
            ax.set_xticks(x)
            ax.set_xticklabels(ticklabels, fontsize=10)
            ax.set_xlabel(xlabel, fontsize=11)
            if col == 0:
                ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f"({'abcd'[row * 2 + col]}) {metric} by {xlabel.lower()}", fontsize=11)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    save(fig, "fig_characteristic")


# ----------------------------------------------------------------------
# mechanism() — 集合の特徴量を 7 特徴量に足したときの MAP の上乗せ（mech.json lift_by_*）
# ----------------------------------------------------------------------
MODES_MECH = [("reranker-7_Comp", "+ Complementarity", "#1a73e8"),
              ("reranker-7_Coh", "+ Coherence", "#188038"),
              ("reranker-7_Dom", "+ Domain agreement", "#9aa0a6"),
              ("reranker-10S", "All three", "#d93025")]


def mechanism():
    me = json.load(open(K21 / "mech.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, key, xlabel, tag in ((axes[0], "lift_by_n_correct", "Number of ground-truth equations", "(a)"),
                                 (axes[1], "lift_by_n_input", "Number of input variables", "(b)")):
        bins = None
        for j, (mode, label, color) in enumerate(MODES_MECH):
            rows = me[key][mode]["MAP"]
            bins = [r["bin"] for r in rows]
            x = np.arange(len(rows)) + (j - 1.5) * 0.08
            ax.errorbar(x, [r["mean_delta"] for r in rows], yerr=[r["std_delta"] for r in rows],
                        marker="o", ms=4, lw=1.6, capsize=3, label=label, color=color)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels([("≥" + s[:-1]) if s.endswith("+") else s.replace("-", "–") for s in bins])
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(f"{tag} By {xlabel.lower()}", fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("MAP gain over the seven-feature reranker", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06),
               framealpha=1.0)   # 図の外（下）に置き、線を隠さない
    fig.tight_layout()
    save(fig, "fig_mechanism")


# ----------------------------------------------------------------------
# greedy_3way() — 一括の並べ直し・逐次選択（推論のみ）・逐次選択（訓練も逐次）の MAP（greedy.json）
# ----------------------------------------------------------------------
GREEDY_CONFIGS = [("static", "Static reranking", C_PROP), ("infer", "Greedy at inference", C_INFER),
                  ("train", "Learned greedy", C_TRAIN)]


def greedy_3way():
    g = json.load(open(K21 / "greedy.json"))["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    centers = np.array([0.0, 3.4])
    for ax, S, title in ((axes[0], "A", "(a) Setting A"), (axes[1], "B", "(b) Setting B (synthesized)")):
        for i, sub in enumerate(("all", "K>=8")):
            data = [[g[S][sub][mode]["MAP"]["per_seed"][str(s)] for s in SEEDS] for mode, _, _ in GREEDY_CONFIGS]
            _box(ax, data, [centers[i] - 0.9, centers[i], centers[i] + 0.9], [c for _, _, c in GREEDY_CONFIGS],
                 widths=0.62)
        ax.set_xticks(centers)
        ax.set_xticklabels(["All cases", "Cases with eight or more\nground-truth equations"], fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("MAP", fontsize=11)
    fig.legend(handles=[Patch(facecolor=c, alpha=0.85, label=lab) for _, lab, c in GREEDY_CONFIGS],
               fontsize=10, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), framealpha=1.0)
    fig.tight_layout()
    save(fig, "fig_greedy_3way")


# ----------------------------------------------------------------------
# pfi() — 日本語版 docs/figures/fig_pfi_recall20_ja_final_1021（analyze_kato1014_pfi.figure）の英語版。同じ構成・同じ値。
# 棒 = シャッフル前の平均 − 低下量の平均（10 分割、各 20 回）、誤差棒 = 低下量の分割間の標準偏差。
# ----------------------------------------------------------------------
EN_FEAT = {
    "text_sim": "Text similarity", "io_jaccard": "I/O Jaccard", "svd_sim": "Latent similarity",
    "input_cov": "Input coverage", "output_cov": "Output coverage", "specificity": "Specificity",
    "domain": "Domain match", "gComp": "Complementarity", "gCoh": "Coherence",
    "gDom": "Domain agreement",
}
VAR_FEATS = ["io_jaccard", "input_cov", "output_cov", "specificity", "gComp", "gCoh"]
TOPIC_FEATS = ["text_sim", "svd_sim", "domain", "gDom"]
C_VAR_G, C_VAR_M = "#08519c", "#6baed6"   # variable overlap: whole group / single feature
C_TOP_G, C_TOP_M = "#a63603", "#fdae6b"   # topic: whole group / single feature


def pfi():
    out = json.load(open(K21 / "pfi.json"))
    m = "Recall@20"
    base = out["base"][m]["mean"]
    res = out["targets"]
    rows = []
    for feats, gkey, c, cg, gname in ((VAR_FEATS, "GROUP_var", C_VAR_M, C_VAR_G, "All 6 variable-overlap features"),
                                      (TOPIC_FEATS, "GROUP_topic", C_TOP_M, C_TOP_G, "All 4 topic features")):
        gr = res[gkey][m]
        rows.append((gname, base - gr["drop_mean"], gr["drop_std"], cg, True))
        for f in sorted(feats, key=lambda f: -res[f][m]["drop_mean"]):
            r = res[f][m]
            rows.append(("  " + EN_FEAT[f], base - r["drop_mean"], r["drop_std"], c, False))
        rows.append(None)
    rows = rows[:-1]
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    y, ticks, labels = 0, [], []
    for row in rows:
        if row is None:
            y += 0.6
            continue
        lab, v, e, col, is_g = row
        ax.barh(y, v, xerr=e, color=col, height=0.72, capsize=3, error_kw=dict(ecolor="#333", lw=1),
                edgecolor="#222" if is_g else col, linewidth=1.2 if is_g else 0)
        ax.text(min(v + e, base) + 0.012 if v < base - 0.05 else base + 0.012, y, f"{v:.3f}",
                va="center", fontsize=11, fontweight="bold" if is_g else "normal")
        ticks.append(y)
        labels.append(lab)
        y += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=12)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if not lab.startswith("  "):
            t.set_fontweight("bold")
    ax.axvline(base, color="#555", ls=":", lw=1.5)
    ax.text(base - 0.01, -1.05, f"before permutation {base:.3f}", ha="right", va="center", fontsize=11,
            color="#333")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(y - 0.4, -1.6)
    ax.set_xlabel("Recall@20 after permutation (Setting A, mean over 10 splits)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[Patch(color=C_VAR_M, label="Variable-overlap features"),
                       Patch(color=C_TOP_M, label="Topic features")],
              fontsize=11, loc="upper left", bbox_to_anchor=(0.38, 0.90), framealpha=1.0)
    fig.tight_layout()
    save(fig, "fig_pfi")


# ----------------------------------------------------------------------
# n1_coverage() — 日本語版 docs/figures/fig_n1_full_coverage_ja_1020.png（analyze_stage1_n1.make_figure、
# marks = 50・200・500・1,000、chosen = 2,000）の英語版。2026-10-27 から fig_n1_full_coverage_ja_1021 と同じ値
# （experiments/kato1021/n1_coverage_w30-70.json）。
# ----------------------------------------------------------------------
def n1_coverage(marks=(50, 200, 500, 1000), chosen=2000):
    d = json.load(open(K21 / "n1_coverage_w30-70.json"))
    col = {"A": "#2a78d6", "B": "#eb6834"}
    lab = {"A": f"Setting A ({d['settings']['A']['n_cases']:,} cases)",
           "B": f"Setting B (synthesized, {d['settings']['B']['n_cases']:,} cases)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in ((axes[0], "full", "(a) Cases with every ground-truth equation among the candidates"),
                           (axes[1], "coverage", "(b) Candidate coverage")):
        for S in ("A", "B"):
            cur = d["settings"][S]["curve"]
            ax.plot(cur["n1"], cur[key], color=col[S], lw=2, label=lab[S])
        for x in marks:
            ax.axvline(x, color="#888888", ls=":", lw=1)
            ax.text(x, 0.705, f" {x:,}", fontsize=8.5, color="#52514e", va="bottom")
        ax.axvline(chosen, color="#333333", ls="-", lw=1.2)
        ax.text(chosen, 0.735, f" {chosen:,}\n (this study)", fontsize=8.5, color="#333333", va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(10, d["config"]["n_eq"])
        ax.set_ylim(0.70, 1.005)
        ax.set_xlabel("Number of first-stage candidates $N_1$ (log scale)", fontsize=11)
        ax.set_ylabel("Fraction", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    save(fig, "fig_n1_coverage")


# ---------------------------------------------------------------------------
# Fig. 1: AutoPMoB overview and the position of this study (schematic, no data)
# 仕様: docs/superpowers/specs/2026-09-07-w2-introduction-design.md §3
# ---------------------------------------------------------------------------
C_DOC = ["#1a73e8", "#e37400", "#188038"]   # document colors for the retrieved set
OVERVIEW_EDGE_LABELS = [r"$r_A$", r"$k$"]   # shared variables: ODE–rate law, rate law–Arrhenius


def _rbox(ax, x0, y0, w, h, text, fc="#f1f3f4", ec="#5f6368", lw=1.0, color="#202124",
          fontsize=7.5, weight="normal", aspect=1.0, rounding=1.5, ha="center", va="center",
          zorder=2):
    """角丸ボックス＋中央テキスト（データ座標）."""
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                         fc=fc, ec=ec, lw=lw, mutation_aspect=aspect, zorder=zorder)
    ax.add_patch(box)
    if text:
        tx = x0 + w / 2 if ha == "center" else x0 + 1.5
        ax.text(tx, y0 + h / 2, text, ha=ha, va=va, fontsize=fontsize, color=color,
                weight=weight, zorder=zorder + 1, linespacing=1.25)
    return box


def _arrow(ax, p, q, color="#5f6368", lw=1.0, scale=9, style="-|>", zorder=3):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=scale, color=color, lw=lw,
                        zorder=zorder, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    return a


def _aspect(fig, ax):
    """データ単位 1 の縦横比（mutation_aspect 用）: y 単位のインチ長 / x 単位のインチ長."""
    bb = ax.get_position()
    w_in = bb.width * fig.get_figwidth(); h_in = bb.height * fig.get_figheight()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (h_in / (y1 - y0)) / (w_in / (x1 - x0))


def overview():
    """Fig. 1: (a) AutoPMoB の 4 段階と本研究の段階、(b) 本研究の入出力（CSTR の 3 式）.

    2026-10-12: 「closed」「degrees of freedom 0」をやめた（自由度＝変数の数 − 式の数。この例は 10 − 3 = 7 で入力 7 個に等しい）。
    """
    fig = plt.figure(figsize=(7.2, 4.9))
    ax_a = fig.add_axes([0.01, 0.715, 0.98, 0.275])
    ax_b = fig.add_axes([0.01, 0.01, 0.98, 0.665])
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    asp_a, asp_b = _aspect(fig, ax_a), _aspect(fig, ax_b)

    # ---------- (a) AutoPMoB pipeline ----------
    ax_a.text(0.5, 97, "(a) AutoPMoB pipeline", fontsize=8.5, weight="bold", va="top", ha="left")
    stages = [
        ("Documents", "papers, textbooks"),
        ("Extraction", "equations, variables, data"),
        ("Equivalence\njudgment", "variables, equations"),
        ("Equation-set\nretrieval", "this study"),
        ("Physical\nmodel", "for simulation"),
    ]
    bw, bh, y0 = 17.5, 58, 12
    centers = [10.5, 30.25, 50, 69.75, 89.5]
    for i, (cx, (title, sub)) in enumerate(zip(centers, stages)):
        this = (i == 3)
        _rbox(ax_a, cx - bw / 2, y0, bw, bh, "",
              fc=C_PROP if this else "#f1f3f4", ec=C_PROP if this else "#5f6368",
              lw=1.6 if this else 1.0, aspect=asp_a, rounding=2.5)
        col = "white" if this else "#202124"
        ax_a.text(cx, y0 + bh * 0.66, title, ha="center", va="center", fontsize=7.4,
                  weight="bold" if this else "normal", color=col, zorder=4, linespacing=1.15)
        ax_a.text(cx, y0 + bh * 0.22, sub, ha="center", va="center", fontsize=6.8,
                  color=col, style="italic" if this else "normal", zorder=4)
    for cx0, cx1 in zip(centers[:-1], centers[1:]):
        _arrow(ax_a, (cx0 + bw / 2 + 0.3, y0 + bh / 2), (cx1 - bw / 2 - 0.3, y0 + bh / 2))

    # ---------- (b) This study ----------
    ax_b.text(0.5, 98.5, "(b) This study: presenting equations for the desired model", fontsize=8.5,
              weight="bold", va="top", ha="left")

    # Query card
    qx0, qy0, qw, qh = 1.5, 30, 24.5, 56
    _rbox(ax_b, qx0, qy0, qw, qh, "", fc="#ffffff", ec="#5f6368", aspect=asp_b, rounding=2)
    ax_b.text(qx0 + qw / 2, qy0 + qh - 6, "Query", ha="center", va="center", fontsize=8,
              weight="bold", zorder=4)
    qlines = [
        ("Description:", "bold"),
        ("concentration dynamics of a", "normal"),
        ("CSTR with a first-order reaction", "normal"),
        (r"Inputs: $F,\,V,\,C_{A0},\,A,\,E,\,R,\,T$", "normal"),
        (r"Output: $C_A$", "normal"),
    ]
    for j, (line, wt) in enumerate(qlines):
        ax_b.text(qx0 + 2.0, qy0 + qh - 15.5 - 8.4 * j, line, ha="left", va="center",
                  fontsize=6.6, weight=wt, zorder=4)

    # Equation database (stacked cards)
    dx0, dy0, dw, dh = 32.5, 33, 23.5, 50
    for off in (3.0, 1.5, 0.0):
        _rbox(ax_b, dx0 + off, dy0 - off * 1.6, dw, dh, "", fc="#f8f9fa", ec="#5f6368",
              aspect=asp_b, rounding=2, zorder=2 - off / 10)
    ax_b.text(dx0 + dw / 2, dy0 + dh - 6, "Equation database", ha="center", va="center",
              fontsize=8, weight="bold", zorder=5)
    ax_b.text(dx0 + dw / 2, dy0 + dh - 15, "11,146 equations", ha="center", va="center",
              fontsize=7.0, zorder=5)
    ax_b.text(dx0 + dw / 2, dy0 + dh - 22.5, "from 361 sources", ha="center", va="center",
              fontsize=7.0, zorder=5)
    ax_b.text(dx0 + dw / 2, dy0 + dh - 33, r"$Q = U A\,\Delta T$,  $P V = n R T$,",
              ha="center", va="center", fontsize=6.8, color="#80868b", zorder=5)
    ax_b.text(dx0 + dw / 2, dy0 + dh - 41, r"$\frac{\mathrm{d}h}{\mathrm{d}t} = \frac{F_{in} - F_{out}}{A_t}$,  ...",
              ha="center", va="center", fontsize=6.8, color="#80868b", zorder=5)

    # Retrieved set: three cards linked by shared variables
    sx0, sw, sh = 71, 27.5, 15.5
    ys = [77, 55, 33]
    eqs = [
        r"$\frac{\mathrm{d}C_A}{\mathrm{d}t} = \frac{F}{V}\,(C_{A0} - C_A) - r_A$",
        r"$r_A = k\,C_A$",
        r"$k = A\,\exp(-E/RT)$",
    ]
    names = ["mass balance", "reaction rate", "Arrhenius"]
    for i, (yc, eq, nm) in enumerate(zip(ys, eqs, names)):
        _rbox(ax_b, sx0, yc - sh / 2, sw, sh, "", fc="#ffffff", ec=C_DOC[i], lw=1.4,
              aspect=asp_b, rounding=2)
        ax_b.text(sx0 + 1.6, yc + sh / 2 - 3.2, f"Doc {i + 1}: {nm}", ha="left", va="center",
                  fontsize=6.4, color=C_DOC[i], zorder=4)
        ax_b.text(sx0 + sw / 2, yc - 2.2, eq, ha="center", va="center", fontsize=7.4, zorder=4)
    # shared-variable edges (ODE–rate law: r_A, rate law–Arrhenius: k)
    for (ya, yb), lab in zip(zip(ys[:-1], ys[1:]), OVERVIEW_EDGE_LABELS):
        xe = sx0 + sw / 2
        ax_b.plot([xe, xe], [ya - sh / 2, yb + sh / 2], color="#5f6368", lw=1.2, zorder=1)
        ax_b.text(xe + 1.8, (ya - sh / 2 + yb + sh / 2) / 2, lab, ha="left", va="center",
                  fontsize=7.2, zorder=4)
        ax_b.text(xe - 1.8, (ya - sh / 2 + yb + sh / 2) / 2, "shared", ha="right", va="center",
                  fontsize=6.2, color="#5f6368", zorder=4)
    ax_b.text(sx0 + sw / 2, 88.5, "Presented equations", ha="center", va="center",
              fontsize=8, weight="bold", zorder=4)
    ax_b.text(sx0 + sw / 2, 18.5, "Given the inputs, the 3 equations",
              ha="center", va="center", fontsize=6.8, zorder=4)
    ax_b.text(sx0 + sw / 2, 11.5, r"determine $C_A$, $r_A$, $k$, one each",
              ha="center", va="center", fontsize=6.8, zorder=4)

    # arrows between the three columns
    ym = 58
    _arrow(ax_b, (qx0 + qw + 0.5, ym), (dx0 - 0.5, ym))
    ax_b.text((qx0 + qw + dx0) / 2, ym + 4.5, "retrieve", ha="center", va="bottom",
              fontsize=6.8, color="#5f6368")
    _arrow(ax_b, (dx0 + dw + 3.5, ym), (sx0 - 0.5, ym))
    ax_b.text((dx0 + dw + 3.5 + sx0) / 2, ym + 4.5, "rank and select\nas a set", ha="center",
              va="bottom", fontsize=6.8, color="#5f6368", linespacing=1.1)

    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_autopmob_overview.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {FIG / 'fig_autopmob_overview'}.png/.pdf")
    return fig


if __name__ == "__main__":
    dataset()
    split_balance()
    method_comparison()
    characteristic()
    mechanism()
    greedy_3way()
    pfi()
    n1_coverage()
    plt.close(overview())
