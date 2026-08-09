"""PFIケース別依存プロファイル：置換で正解→不正解に転じるケースを属性で特徴づける.

入力 : experiments/pfi_per_case.json（analyze_pfi.py が出力）
出力 : experiments/pfi_profile_stats.json, docs/figures/fig_pfi_dependence_profile.png/.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).parent
SETAWARE = [("gComp", "＋補完性", "#1f77b4"),
            ("gCoh", "＋一貫性", "#2ca02c"),
            ("gDom", "＋ドメイン一致", "#d62728")]
ATTRS = ["n_correct", "n_input", "n_output", "n_sources", "sep"]


def load_per_case(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["records"], d.get("config", {})


def split_dependent_robust(records, feature):
    solved = [r for r in records if r["feature"] == feature and r["base_R"] == 1.0]
    dep = [r for r in solved if r["flip"] == 1]
    rob = [r for r in solved if r["flip"] == 0]
    return dep, rob


def attr_array(recs, attr):
    return np.array([r[attr] for r in recs if r.get(attr) is not None], dtype=float)


def mannwhitney_effect(dep, rob):
    res = {"n_dep": int(dep.size), "n_rob": int(rob.size),
           "mean_dep": float(dep.mean()) if dep.size else float("nan"),
           "sem_dep": float(dep.std(ddof=1) / np.sqrt(dep.size)) if dep.size > 1 else 0.0,
           "mean_rob": float(rob.mean()) if rob.size else float("nan"),
           "sem_rob": float(rob.std(ddof=1) / np.sqrt(rob.size)) if rob.size > 1 else 0.0,
           "mannwhitney_p": float("nan"), "rank_biserial_r": float("nan")}
    if dep.size and rob.size:
        U, p = stats.mannwhitneyu(dep, rob, alternative="two-sided")
        res["mannwhitney_p"] = float(p)
        res["rank_biserial_r"] = float(2.0 * U / (dep.size * rob.size) - 1.0)
    return res


def standardized_mean_diff(dep, rob):
    n1, n2 = dep.size, rob.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = np.sqrt(((n1 - 1) * dep.var(ddof=1) + (n2 - 1) * rob.var(ddof=1)) / (n1 + n2 - 2))
    return float((dep.mean() - rob.mean()) / sp) if sp > 0 else 0.0


def smd_stderr(dep, rob):
    """標準化平均差（Cohen's d）の閉形式標準誤差.

    SE(d) = sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2)))
    いずれかの群が 2 未満なら nan。
    """
    n1, n2 = dep.size, rob.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    d = standardized_mean_diff(dep, rob)
    if d != d:  # nan
        return float("nan")
    return float(np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2.0 * (n1 + n2))))


def spearman_drop_sep(records, feature):
    """drop と sep の Spearman 順位相関.

    仕様に従い baseline 解済み（base_R == 1.0）の記録のみを対象とする。
    base_R < 1.0 のケースは drop が床効果で頭打ちになり相関を歪めるため除外する。
    """
    recs = [r for r in records
            if r["feature"] == feature and r.get("sep") is not None
            and r["base_R"] == 1.0]
    if len(recs) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(recs)}
    rho, p = stats.spearmanr([r["sep"] for r in recs], [r["drop"] for r in recs])
    return {"rho": float(rho), "p": float(p), "n": len(recs)}


def sep_profile(records, feature, n_bins=5):
    """sep の分位ビンごとの平均 drop ±SEM（全ケース対象）."""
    recs = [r for r in records if r["feature"] == feature and r.get("sep") is not None]
    if len(recs) < n_bins:
        return [], [], []
    seps = np.array([r["sep"] for r in recs])
    drops = np.array([r["drop"] for r in recs])
    edges = np.quantile(seps, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    xs, ys, es = [], [], []
    for b in range(n_bins):
        m = (seps >= edges[b]) & (seps < edges[b + 1])
        if m.sum() == 0:
            continue
        xs.append(float(0.5 * (edges[b] + edges[b + 1])))
        ys.append(float(drops[m].mean()))
        es.append(float(drops[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else 0.0)
    return xs, ys, es


def build_stats(records):
    feats = {}
    all_feats = sorted(set(r["feature"] for r in records))
    for name in all_feats:
        dep, rob = split_dependent_robust(records, name)
        solved = len(dep) + len(rob)
        entry = {"n_dependent": len(dep), "n_robust": len(rob),
                 "flip_fraction_of_solved": (len(dep) / solved) if solved else 0.0,
                 "spearman_drop_sep": spearman_drop_sep(records, name), "attrs": {}}
        for attr in ATTRS:
            entry["attrs"][attr] = mannwhitney_effect(attr_array(dep, attr), attr_array(rob, attr))
        feats[name] = entry
    return {"metric": "Recall@K_correct", "n_features": len(all_feats), "features": feats}


def _make_figure(records, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # (a) 分離度ビン × 平均drop
    for name, label, col in SETAWARE:
        xs, ys, es = sep_profile(records, name, n_bins=5)
        if xs:
            axes[0].errorbar(xs, ys, yerr=es, marker="o", lw=2.2, capsize=3, color=col, label=label)
    axes[0].axhline(0, color="#bbbbbb", lw=1, zorder=0)
    axes[0].set_title("(a) 分離度別の期待Recall低下")
    axes[0].set_xlabel("分離度 sep（正解式−不正解式の特徴値差）")
    axes[0].set_ylabel("置換によるRecall@K_correctの期待低下")
    axes[0].grid(alpha=0.25); axes[0].legend(fontsize=9, loc="upper left")

    # (b) gComp/gCoh の 依存−頑健 標準化平均差（属性別・横棒）
    bar_feats = [("gComp", "#1f77b4"), ("gCoh", "#2ca02c")]
    y = np.arange(len(ATTRS)); h = 0.38
    for bi, (name, col) in enumerate(bar_feats):
        dep, rob = split_dependent_robust(records, name)
        smds = [standardized_mean_diff(attr_array(dep, a), attr_array(rob, a)) for a in ATTRS]
        ses = [smd_stderr(attr_array(dep, a), attr_array(rob, a)) for a in ATTRS]
        smds = [0.0 if (s != s) else s for s in smds]  # nan→0
        ses = [0.0 if (e != e) else e for e in ses]    # nan→0
        axes[1].barh(y + (bi - 0.5) * h, smds, height=h, xerr=ses,
                     error_kw=dict(capsize=3, lw=1), color=col, label=name)
    axes[1].axvline(0, color="#888888", lw=1)
    axes[1].set_yticks(y); axes[1].set_yticklabels(ATTRS)
    axes[1].set_title("(b) 依存ケースを特徴づける属性")
    axes[1].set_xlabel("標準化平均差（依存−頑健）")
    axes[1].grid(alpha=0.25, axis="x"); axes[1].legend(fontsize=9)

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="experiments/pfi_per_case.json")
    ap.add_argument("--out-json", default="experiments/pfi_profile_stats.json")
    ap.add_argument("--out-fig", default="docs/figures/fig_pfi_dependence_profile.png")
    a = ap.parse_args()
    records, config = load_per_case(ROOT / a.input)
    stats_out = build_stats(records)
    stats_out["config"] = config
    json.dump(stats_out, open(ROOT / a.out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved stats: {a.out_json}")
    _make_figure(records, ROOT / a.out_fig)
    print(f"Saved figure: {a.out_fig}")
    # コンソール要約
    for name, label, _ in SETAWARE:
        g = stats_out["features"].get(name, {})
        s = g.get("spearman_drop_sep", {})
        print(f"{name:6s} 依存={g.get('n_dependent')}/{g.get('n_dependent',0)+g.get('n_robust',0)} "
              f"Spearman(drop,sep) rho={s.get('rho')} p={s.get('p')}")


if __name__ == "__main__":
    main()
