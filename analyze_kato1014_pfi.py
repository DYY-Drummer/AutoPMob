"""§5・§8 の PFI の集計と図（2026-10-14 の指示 5）.

入力: --rundir/pfi/pfi_A__{seed}.json（kato1014_pfi.py の出力、10 分割）
出力: experiments/kato1014/pfi_<tag>.json
      docs/figures/fig_pfi_recall20_ja_<tag>.{png,pdf}（旧 fig_pfi_recall_ja と同じ形。報告書の本文は変えないので、
      旧図は上書きしない）
集計（analyze_pfi.py・analyze_pfi_profile.py と同じ数え方）:
  - 分割ごとの低下量 = シャッフル前の平均 − シャッフル後の平均（20 回の平均）。10 分割の平均 ± 標準偏差と、
    低下量が 0 かの 1 標本 t 検定。
  - x10（分野ラベルの一致度（集合））の打ち消し合い: ケースごとに、現れた分割での低下量と sep（候補の中で
    正解式の平均 − 不正解式の平均）を平均して 1 ケース 1 値にし、sep > 0 と sep < 0 の 2 群を Mann–Whitney の
    U 検定で比べる（analyze_pfi_profile.cancellation_check(solved_only=False) と同じ）。
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from kato1014_common import ROOT, dump

METRICS = ("Recall@20", "MAP", "MAP@20", "Recall@K")
JP = {"text_sim": "文章類似度", "io_jaccard": "変数の一致度", "svd_sim": "潜在意味類似度",
      "input_cov": "入力変数の被覆率", "output_cov": "出力変数の被覆率", "specificity": "式の特化度",
      "domain": "分野ラベルの一致", "gComp": "補完性", "gCoh": "一貫性", "gDom": "分野ラベルの一致度（集合）"}
VAR = ["io_jaccard", "input_cov", "output_cov", "specificity", "gComp", "gCoh"]
TOPIC = ["text_sim", "svd_sim", "domain", "gDom"]


def cancellation(per_seed_files, feature, metric):
    by = defaultdict(lambda: {"sep": [], "drop": []})
    for d in per_seed_files:
        tgt = d["pfi"]["targets"][feature]["case_perm_mean"][metric]
        for c, perm in zip(d["pfi"]["cases"], tgt):
            s = c["sep"].get(feature)
            if s is None or s != s:
                continue
            by[c["case_id"]]["sep"].append(s)
            by[c["case_id"]]["drop"].append(c["base"][metric] - perm)
    cases = [{"case_id": k, "sep": float(np.mean(v["sep"])), "drop": float(np.mean(v["drop"])),
              "n_obs": len(v["drop"])} for k, v in by.items()]
    pos = np.array([c["drop"] for c in cases if c["sep"] > 0])
    neg = np.array([c["drop"] for c in cases if c["sep"] < 0])
    zero = sum(1 for c in cases if c["sep"] == 0)
    res = {"n_cases": len(cases), "n_records": int(sum(c["n_obs"] for c in cases)),
           "n_sep_pos": int(pos.size), "n_sep_neg": int(neg.size), "n_sep_zero": zero,
           "mean_drop_sep_pos": float(pos.mean()) if pos.size else None,
           "mean_drop_sep_neg": float(neg.mean()) if neg.size else None,
           "mannwhitney_p": float(stats.mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
           if pos.size and neg.size else None}
    if pos.size and neg.size:
        res["weighted_mean_drop_pos_neg"] = float((pos.sum() + neg.sum()) / (pos.size + neg.size))
    return res


def figure(out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    m = "Recall@20"
    base = out["base"][m]["mean"]
    res = out["targets"]
    C_VAR, C_VAR_G, C_TOP, C_TOP_G = "#6baed6", "#08519c", "#fdae6b", "#a63603"
    rows = []
    for feats, gkey, c, cg, gname in ((VAR, "GROUP_var", C_VAR, C_VAR_G, "変数の重複率を測る 6 特徴量をまとめて"),
                                      (TOPIC, "GROUP_topic", C_TOP, C_TOP_G, "話題の近さを測る 4 特徴量をまとめて")):
        g = res[gkey][m]
        rows.append((gname, base - g["drop_mean"], g["drop_std"], cg, True))
        for f in sorted(feats, key=lambda f: -res[f][m]["drop_mean"]):
            r = res[f][m]
            rows.append(("  " + JP[f], base - r["drop_mean"], r["drop_std"], c, False))
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
        ticks.append(y); labels.append(lab); y += 1
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=12)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if not lab.startswith("  "):
            t.set_fontweight("bold")
    ax.axvline(base, color="#555", ls=":", lw=1.5)
    ax.text(base - 0.01, -1.05, f"シャッフル前 {base:.3f}", ha="right", va="center", fontsize=11, color="#333")
    ax.set_xlim(0, 1.12)
    ax.set_ylim(y - 0.4, -1.6)
    ax.set_xlabel("シャッフル後の Recall@20（設定 A、乱数 10 通りの平均）", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[Patch(color=C_VAR, label="変数の重複率を測る特徴量"),
                       Patch(color=C_TOP, label="話題の近さを測る特徴量")],
              fontsize=11, loc="upper left", bbox_to_anchor=(0.2, 0.93), framealpha=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = ROOT / "docs" / "figures" / f"fig_pfi_recall20_ja_{tag}.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        print(f"Saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    files = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(f"{a.rundir}/pfi/pfi_A__*.json"))]
    seeds = [d["pfi"]["seed"] for d in files]
    out = {"rundir": a.rundir, "seeds": seeds, "n_perm": files[0]["config"]["n_perm"],
           "top_k": files[0]["config"]["top_k"], "base": {}, "targets": {}, "cancellation": {}}
    for m in METRICS:
        b = [d["pfi"]["base"][m] for d in files]
        out["base"][m] = {"mean": float(np.mean(b)), "std": float(np.std(b, ddof=1)),
                          "per_seed": dict(zip(map(str, seeds), b))}
    for lab in files[0]["pfi"]["targets"]:
        out["targets"][lab] = {}
        for m in METRICS:
            drops = np.array([d["pfi"]["base"][m] - d["pfi"]["targets"][lab]["agg_perm_mean"][m] for d in files])
            t = stats.ttest_1samp(drops, 0.0)
            out["targets"][lab][m] = {"drop_mean": float(drops.mean()), "drop_std": float(drops.std(ddof=1)),
                                      "shuffled_mean": float(np.mean([d["pfi"]["targets"][lab]["agg_perm_mean"][m]
                                                                      for d in files])),
                                      "t": float(t.statistic), "p": float(t.pvalue)}
    for f in ("gDom", "domain", "svd_sim", "text_sim"):
        out["cancellation"][f] = {m: cancellation(files, f, m) for m in ("Recall@20", "MAP")}
    dump(out, f"pfi_{a.tag}.json")
    figure(out, a.tag)
    m = "Recall@20"
    print(f"base {m} {out['base'][m]['mean']:.4f}±{out['base'][m]['std']:.4f}")
    for lab, v in sorted(out["targets"].items(), key=lambda kv: -kv[1][m]["drop_mean"]):
        print(f"  {lab:14s} drop {v[m]['drop_mean']:+.4f}±{v[m]['drop_std']:.4f} → {v[m]['shuffled_mean']:.4f}  p={v[m]['p']:.2g}"
              f" | MAP drop {v['MAP']['drop_mean']:+.4f} p={v['MAP']['p']:.2g}")
    c = out["cancellation"]["gDom"][m]
    print(f"  x10 打ち消し: sep>0 {c['n_sep_pos']} 件 drop {c['mean_drop_sep_pos']}, sep<0 {c['n_sep_neg']} 件 "
          f"drop {c['mean_drop_sep_neg']}, MW p={c['mannwhitney_p']}")


if __name__ == "__main__":
    main()
