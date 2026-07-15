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


def spearman_drop_sep(records, feature):
    recs = [r for r in records if r["feature"] == feature and r.get("sep") is not None]
    if len(recs) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(recs)}
    rho, p = stats.spearmanr([r["sep"] for r in recs], [r["drop"] for r in recs])
    return {"rho": float(rho), "p": float(p), "n": len(recs)}
