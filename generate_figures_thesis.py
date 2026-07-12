#!/usr/bin/env python3
"""修士論文用の英語ラベル図を一括生成する.

出力: thesis/master_thesis/figures/*.{png,pdf}
データ: training_cases.json / experiments/strat_*.json / llm_set_*_equiv_results.json /
        experiments/xg/ xg_A/ xd_dof/（シード別生データ、箱ひげ用）/
        experiments/ablation_x_difficulty.json
方針: 10シード比較は箱ひげ図（G7）。LLM直接生成はシードなし単発評価のため点表示。
      日本語フォント設定は入れない（英語ラベルのみ）。suptitleは付けない。
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
FIG = ROOT / "thesis" / "master_thesis" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.axisbelow"] = True
matplotlib.rcParams["pdf.fonttype"] = 42

C_BASE = "#9aa0a6"   # baseline: gray
C_PROP = "#1a73e8"   # proposed: blue
C_LLM = "#d93025"    # LLM: red
C_INFER = "#f9ab00"  # inference greedy: amber
C_TRAIN = "#188038"  # trained greedy: green
C_DAE = "#16A085"    # DAE hard-case highlight (ported from generate_figures_strat.py)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {FIG / name}.png/.pdf")


def _box(ax, data, positions, colors, widths=0.55):
    """10シード値の箱ひげ＋ジッター散点（fig_v6_boxplot のスタイル踏襲）."""
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


def _per_seed(path, mode, metric="Recall@K_correct"):
    doc = json.load(open(path))
    return [p[metric] for p in doc["results"][mode]["per_seed"]]


# ----------------------------------------------------------------------
# dataset() — ported from generate_figures_strat.py L53-101
# ----------------------------------------------------------------------
def dataset():
    cases = json.load(open(ROOT / "training_cases.json"))
    vt = Counter(c.get("variant_type", "?") for c in cases)
    fam = {
        "Original": vt["original"],
        "Paraphrase (aug.)": vt["context_paraphrased"],
        "Random I/O (aug.)": vt["random_io_from_models"] + vt["swap_io"],
        "Multi-source": vt["multisource_original"] + vt["multisource_random_io"] + vt["multisource_v3"],
        "DAE (hard)": sum(v for k, v in vt.items() if k.startswith("dae_")),
    }
    labels, vals = list(fam.keys()), list(fam.values())
    # 青＋緑＝評価の設定 A（original+multisource+DAE）, 灰＝拡張のため設定 A から除外
    colors = [C_PROP, C_BASE, C_BASE, C_PROP, C_DAE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ypos = np.arange(len(labels))
    bars = ax1.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_yticks(ypos); ax1.set_yticklabels(labels, fontsize=10); ax1.invert_yaxis()
    for b, v in zip(bars, vals):
        ax1.text(v + 12, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of cases", fontsize=12)
    ax1.set_xlim(0, max(vals) * 1.18)
    ax1.set_title(f"(a) Composition of the 2,838 cases", fontsize=12)
    ax1.legend(handles=[Patch(facecolor=C_PROP, edgecolor="black", label="Setting A: evaluation target (1,823 cases)"),
                        Patch(facecolor=C_DAE, edgecolor="black", label="  of which DAE (1,000 cases)"),
                        Patch(facecolor=C_BASE, edgecolor="black", label="Augmentation: excluded from Setting A")],
               fontsize=9, loc="upper right")

    nc = Counter(len(c.get("correct_model_ids", [])) for c in cases)
    ks = sorted(k for k in nc if k >= 1); vs = [nc[k] for k in ks]
    multi_ratio = sum(v for k, v in zip(ks, vs) if k >= 2) / sum(vs)
    barcolors = [C_BASE if k == 1 else C_PROP for k in ks]
    bars = ax2.bar(ks, vs, color=barcolors, edgecolor="black", linewidth=0.8)
    for b, v in zip(bars, vs):
        if v >= 20:
            ax2.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}", ha="center", fontsize=8.5)
    ax2.set_xlabel("Number of equations in the true model", fontsize=12)
    ax2.set_ylabel("Number of cases", fontsize=12); ax2.set_xticks(ks)
    ax2.set_title(f"(b) Distribution of the number of correct equations", fontsize=12)
    ax2.legend(handles=[Patch(facecolor=C_BASE, edgecolor="black", label="Single-equation"),
                        Patch(facecolor=C_PROP, edgecolor="black", label="Multi-equation")],
               fontsize=10, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_dataset")


# ----------------------------------------------------------------------
# characteristic() — ported from generate_figures_strat.py L128-252
# (incl. bucket_stats L132-147)
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
    d = json.load(open(EXP / "strat_A.json"))
    pcm = {mode: d["results"][mode]["per_case"] for mode in d["results"]}

    # --- 数式数 n_correct: 1..10 ---
    ncs = list(range(1, 11))
    nc = bucket_stats(pcm, lambda e: e["n_correct"] if 1 <= e["n_correct"] <= 10 else (10 if e["n_correct"] > 10 else None), ncs)
    nc_base, nc_ours = [], []
    for k in ncs:
        b = nc[k]["baseline"][0]; r = nc[k]["reranker-10S"][0]
        nc_base.append(b); nc_ours.append(r)

    # --- ソース数 n_sources: 1,2,>=3 ---
    def srcbucket(e):
        s = e["n_sources"]
        if s == 1: return "1"
        if s == 2: return "2"
        return "3+"
    sorder = ["1", "2", "3+"]
    sc = bucket_stats(pcm, srcbucket, sorder)
    sc_base, sc_ours = [], []
    for k in sorder:
        b = sc[k]["baseline"][0]; r = sc[k]["reranker-10S"][0]
        sc_base.append(b); sc_ours.append(r)

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
    ic_base, ic_ours = [], []
    for k in iorder:
        b = ic[k]["baseline"][0]; r = ic[k]["reranker-10S"][0]
        ic_base.append(b); ic_ours.append(r)

    iorder_labels = ["≤5", "6–10", "11–15", "16–20", "≥21"]

    # ---------------- 図（3 パネル） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) 数式数
    ax = axes[0]
    ax.plot(ncs, nc_base, "o-", color=C_BASE, lw=2.2, ms=7, label="Baseline (classical IR)")
    ax.plot(ncs, nc_ours, "s-", color=C_PROP, lw=2.4, ms=7, label="Proposed (reranker-10S)")
    ax.fill_between(ncs, nc_base, nc_ours, color=C_PROP, alpha=0.15)
    ax.set_xlabel("Number of equations in the true model", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_xticks(ncs)
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) By number of equations", fontsize=12)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) ソース数
    ax = axes[1]
    x = np.arange(len(sorder)); w = 0.38
    ax.bar(x - w / 2, sc_base, w, color=C_BASE, edgecolor="black", linewidth=0.8, label="Baseline (classical IR)")
    ax.bar(x + w / 2, sc_ours, w, color=C_PROP, edgecolor="black", linewidth=0.8, label="Proposed (reranker-10S)")
    for i in range(len(sorder)):
        ax.text(x[i] - w / 2, sc_base[i] + 0.012, f"{sc_base[i]:.2f}", ha="center", fontsize=9)
        ax.text(x[i] + w / 2, sc_ours[i] + 0.012, f"{sc_ours[i]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["1", "2", "≥3"], fontsize=11)
    ax.set_xlabel("Number of source documents", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) By number of sources", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    # (c) 入力変数数
    ax = axes[2]
    x = np.arange(len(iorder))
    ax.plot(x, ic_base, "o-", color=C_BASE, lw=2.2, ms=7, label="Baseline (classical IR)")
    ax.plot(x, ic_ours, "s-", color=C_PROP, lw=2.4, ms=7, label="Proposed (reranker-10S)")
    ax.fill_between(x, ic_base, ic_ours, color=C_PROP, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(iorder_labels, fontsize=11)
    ax.set_xlabel("Number of input variables", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.set_title("(c) By number of input variables", fontsize=12)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save(fig, "fig_characteristic")


# ----------------------------------------------------------------------
# split_balance() — ported from generate_figures_v6.py fig_split_balance() L586-635
# ----------------------------------------------------------------------
def split_balance(seed=42):
    import sys
    sys.path.insert(0, str(ROOT))
    from set_aware_reranker import stratified_src_split
    from two_stage_query_conditioned import load_cases, case_src, in_vars, out_vars
    cases = load_cases()
    keep = lambda v: v == "original" or v.startswith("multisource_") or v.startswith("dae_")
    sub = [c for c in cases if keep(c.get("variant_type", ""))]
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
        # 連続的な変数はビン化
        if len(allv) > 12:
            import numpy as _np
            bins = _np.linspace(min(vals), max(vals), 10)
            trh, _ = _np.histogram([vals[i] for i in tr], bins=bins, density=True)
            teh, _ = _np.histogram([vals[i] for i in te], bins=bins, density=True)
            x = (bins[:-1] + bins[1:]) / 2
            w = (bins[1] - bins[0]) * 0.4
            ax.bar(x - w/2, trh, w, color=C_PROP, alpha=0.8, label="Train")
            ax.bar(x + w/2, teh, w, color=C_LLM, alpha=0.8, label="Test")
        else:
            from collections import Counter as _Counter
            ctr = _Counter(vals[i] for i in tr); cte = _Counter(vals[i] for i in te)
            ntr = sum(ctr.values()); nte = sum(cte.values())
            x = np.arange(len(allv)); w = 0.38
            ax.bar(x - w/2, [ctr[v]/ntr for v in allv], w, color=C_PROP, alpha=0.85, label="Train")
            ax.bar(x + w/2, [cte[v]/nte for v in allv], w, color=C_LLM, alpha=0.85, label="Test")
            ax.set_xticks(x); ax.set_xticklabels(allv, fontsize=8)
        ax.set_title(name, fontsize=12); ax.set_xlabel("Value", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Fraction", fontsize=11); axes[0].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "fig_split_balance")
    print(f"split_balance: train/test = {len(tr)}/{len(te)}")


def method_comparison():
    """3設定（全データ/設定A/設定B）× baseline・本手法の箱ひげ＋LLM点."""
    settings = [
        ("strat_full.json", "llm_set_full_equiv_results.json", "All cases\n(incl. augmentation, 2,838)"),
        ("strat_A.json", "llm_set_A_equiv_results.json", "Setting A\n(1,823)"),
        ("strat_B.json", "llm_set_B_equiv_results.json", "Setting B\n(DAE only, 1,000)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    centers = np.arange(len(settings)) * 2.6
    for i, (sf, lf, label) in enumerate(settings):
        base = _per_seed(EXP / sf, "baseline")
        prop = _per_seed(EXP / sf, "reranker-10S")
        _box(ax, [base, prop], [centers[i] - 0.45, centers[i] + 0.45], [C_BASE, C_PROP])
        llm = json.load(open(EXP / lf))
        ax.scatter([centers[i] + 1.05], [llm["Recall@K_correct"]], marker="D", s=55,
                   color=C_LLM, zorder=4)
        ax.hlines(llm["coverage"], centers[i] + 0.85, centers[i] + 1.25,
                  colors=C_LLM, linestyles="--", lw=1.4)
    ax.set_xticks(centers)
    ax.set_xticklabels([s[2] for s in settings])
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_BASE, alpha=0.85, label="Baseline (classical IR)"),
        plt.Rectangle((0, 0), 1, 1, fc=C_PROP, alpha=0.85, label="Proposed (reranker-10S)"),
        plt.Line2D([], [], marker="D", ls="", color=C_LLM, label="LLM direct generation (single run, n≈50)"),
        plt.Line2D([], [], ls="--", color=C_LLM, label="LLM coverage (rank-free)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, ncol=2)
    save(fig, "fig_method_comparison")


# ----------------------------------------------------------------------
# mechanism() — feature x difficulty cross (per Task 5 brief Step 1)
# ----------------------------------------------------------------------
MODES_MECH = [
    ("reranker-7+Comp", "+Complementarity (gComp)", "#1a73e8"),
    ("reranker-7+Coh", "+Coherence (gCoh)", "#188038"),
    ("reranker-7+Dom", "+Domain (gDom)", "#9aa0a6"),
    ("reranker-10S", "All three (10S)", "#d93025"),
]
BASE_MECH = "reranker-7"


def _seed_bucket_means(per_case, bucket_fn):
    """per_case を (seed, bucket) 別に平均 → {bucket: {seed: mean}}."""
    acc = {}
    for rec in per_case:
        b = bucket_fn(rec)
        if b is None:
            continue
        acc.setdefault(b, {}).setdefault(rec["seed"], []).append(rec["Recall@K_correct"])
    return {b: {s: float(np.mean(v)) for s, v in d.items()} for b, d in acc.items()}


def _lift_sem(base_by, mode_by):
    """シード対応リフトの mean±SEM を bucket ごとに返す."""
    out = {}
    for b in sorted(set(base_by) & set(mode_by)):
        seeds = sorted(set(base_by[b]) & set(mode_by[b]))
        d = np.array([mode_by[b][s] - base_by[b][s] for s in seeds])
        if len(d) >= 2:
            out[b] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))))
    return out


def mechanism():
    doc = json.load(open(EXP / "ablation_x_difficulty.json"))
    res = doc["results"]
    nin_edges = [(1, 3, "1–3"), (4, 6, "4–6"), (7, 10, "7–10"), (11, 15, "11–15"), (16, 10**9, "≥16")]

    def by_x(rec):
        return rec["n_correct"] if 1 <= rec["n_correct"] <= 10 else None

    def by_nin(rec):
        for lo, hi, lab in nin_edges:
            if lo <= rec["n_input"] <= hi:
                return lab
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for bucket_fn, ax, xlabel, order in [
        (by_x, axes[0], "Number of equations in the true model", list(range(1, 11))),
        (by_nin, axes[1], "Number of input variables", [e[2] for e in nin_edges]),
    ]:
        base_by = _seed_bucket_means(res[BASE_MECH]["per_case"], bucket_fn)
        for mode, label, color in MODES_MECH:
            ls = _lift_sem(base_by, _seed_bucket_means(res[mode]["per_case"], bucket_fn))
            xs = [b for b in order if b in ls]
            ax.errorbar(range(len(xs)), [ls[b][0] for b in xs], yerr=[ls[b][1] for b in xs],
                        marker="o", ms=4, lw=1.6, capsize=3, label=label, color=color)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len([b for b in order if b in ls])))
        ax.set_xticklabels([str(b) for b in order if b in ls])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Recall@K lift over 7-feature reranker")
    axes[0].set_title("(a) Stratified by number of equations")
    axes[1].set_title("(b) Stratified by number of input variables")
    axes[0].legend(fontsize=8.5)
    save(fig, "fig_mechanism")


# ----------------------------------------------------------------------
# greedy_3way() — static / inference-greedy / trained-greedy comparison
# ----------------------------------------------------------------------
def _greedy_seed_means(dirname, config):
    vals = {}
    for f in sorted((EXP / dirname).glob(f"{config}__*.json")):
        doc = json.load(open(f))
        for mode_res in doc["results"].values():
            for rec in mode_res.get("per_case", []):
                vals.setdefault(rec["seed"], []).append(rec["Recall@K_correct"])
    return [float(np.mean(v)) for _, v in sorted(vals.items())]


GREEDY_CONFIGS = [("static", "Static", C_BASE), ("infer", "Greedy\n(inference)", C_INFER),
                  ("train", "Greedy\n(trained)", C_TRAIN)]


def greedy_3way():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, dirname, title in [(axes[0], "xg", "(a) Setting B (DAE only)"),
                               (axes[1], "xg_A", "(b) Setting A")]:
        data = [_greedy_seed_means(dirname, c) for c, _, _ in GREEDY_CONFIGS]
        _box(ax, data, [0, 1, 2], [c for _, _, c in GREEDY_CONFIGS])
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([lab for _, lab, _ in GREEDY_CONFIGS], fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Recall@K (seed mean)")
    # (c) DAE の正解式数別（シード別平均の mean±SEM）
    ax = axes[2]
    for config, label, color in GREEDY_CONFIGS:
        acc = {}
        for f in sorted((EXP / "xg").glob(f"{config}__*.json")):
            doc = json.load(open(f))
            for mode_res in doc["results"].values():
                for rec in mode_res.get("per_case", []):
                    acc.setdefault(rec["n_correct"], {}).setdefault(rec["seed"], []).append(
                        rec["Recall@K_correct"])
        xs = sorted(x for x in acc if 1 <= x <= 10)
        means, sems = [], []
        for x in xs:
            sm = np.array([np.mean(v) for v in acc[x].values()])
            means.append(sm.mean())
            sems.append(sm.std(ddof=1) / np.sqrt(len(sm)))
        ax.errorbar(xs, means, yerr=sems, marker="o", ms=4, lw=1.6, capsize=3,
                    label=label.replace("\n", " "), color=color)
    ax.set_xlabel("Number of equations in the true model (DAE)")
    ax.set_title("(c) By difficulty (Setting B)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    save(fig, "fig_greedy_3way")


# ----------------------------------------------------------------------
# dof_stop() — DoF-based self-stopping vs. oracle-K
# ----------------------------------------------------------------------
def _dof_seed_means(setting, key):
    vals = {}
    for f in sorted((EXP / "xd_dof").glob(f"{setting}__*.json")):
        doc = json.load(open(f))
        for mode_res in doc["results"].values():
            for rec in mode_res.get("per_case", []):
                vals.setdefault(rec["seed"], []).append(rec[key])
    return [float(np.mean(v)) for _, v in sorted(vals.items())]


def dof_stop():
    stats_doc = json.load(open(EXP / "dof_stop_stats.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    groups = [("dae", "Setting B (DAE)"), ("A", "Setting A")]
    for i, (setting, glabel) in enumerate(groups):
        oracle = _dof_seed_means(setting, "set_f1_oracleK")
        dof = _dof_seed_means(setting, "set_f1_dof")
        _box(ax, [oracle, dof], [i * 2.4 - 0.45, i * 2.4 + 0.45], [C_BASE, C_PROP])
        dof_closed = stats_doc[setting]["closed_dof"]["mean"]
        oracle_closed = stats_doc[setting]["closed_oracleK"]["mean"]
        ax.text(i * 2.4, 1.02, f"closure {dof_closed:.2f} vs {oracle_closed:.2f} (oracle)",
                ha="center", fontsize=7.5)
    ax.set_ylim(top=1.10)
    ax.set_xticks([0, 2.4])
    ax.set_xticklabels([g[1] for g in groups])
    ax.set_ylabel("Set F1 (seed mean)")
    ax.set_title("(a) Oracle-K vs. DoF-stop")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=C_BASE, alpha=0.85, label="Oracle-K (K given)"),
               plt.Rectangle((0, 0), 1, 1, fc=C_PROP, alpha=0.85, label="DoF-stop (K unknown)")]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax = axes[1]
    series = [("set_f1_oracleK", "Set F1, oracle-K", C_BASE), ("set_f1_dof", "Set F1, DoF-stop", C_PROP),
              ("set_exact_dof", "Exact match, DoF-stop", C_TRAIN)]
    for key, label, color in series:
        acc = {}
        for f in sorted((EXP / "xd_dof").glob("dae__*.json")):
            doc = json.load(open(f))
            for mode_res in doc["results"].values():
                for rec in mode_res.get("per_case", []):
                    acc.setdefault(rec["n_correct"], {}).setdefault(rec["seed"], []).append(rec[key])
        xs = sorted(x for x in acc if 1 <= x <= 10)
        sm = [np.array([np.mean(v) for v in acc[x].values()]) for x in xs]
        ax.errorbar(xs, [m.mean() for m in sm], yerr=[m.std(ddof=1) / np.sqrt(len(m)) for m in sm],
                    marker="o", ms=4, lw=1.6, capsize=3, label=label, color=color)
    ax.set_xlabel("Number of equations in the true model (DAE)")
    ax.set_title("(b) By difficulty (Setting B)")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    save(fig, "fig_dof_stop")


if __name__ == "__main__":
    dataset()
    split_balance()
    method_comparison()
    characteristic()
    mechanism()
    greedy_3way()
    dof_stop()
