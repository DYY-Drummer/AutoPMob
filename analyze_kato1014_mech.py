"""D（2026-10-14 の指示 5）：§5 の向上の内訳と §8 の話題の特徴量を、新しい設定と Recall@20・MAP でやり直す集計.

入力（--rundir、run_kato1014.sh の出力。設定 A）:
  reranker-2_A, reranker-7_A, reranker-2_Spec_A, reranker-2_Svd_A, reranker-2_Dom_A,
  reranker-7_Comp_A, reranker-7_Coh_A, reranker-7_Dom_A, static_A（= reranker-10S）の {name}__{seed}.json
  分野ラベルをまとめた reranker-2+Dom: --rundir/domain_merge_v2/{orig,L1,K100,K40}__{seed}.json（kato1014_domain_merge.py。
    x7 は群で数える。旧版と同じく群の名前で比べると K100・K40 で x7 がほぼ 0 になり、合併の効果を測れない）
  baseline（同じ重み 0.3/0.7）: experiments/kato1014/raw/baselines_A.json の tfidf__none__w30-70
旧版との対応:
  §5 (i)(ii)(iii) と x6 だけを足した場合 … 旧版は集計スクリプトなし（xd/ の per-seed から計算されていた）
  集合の特徴量を 1 個ずつ足す、正解式の数・入力変数の数で分けた上乗せ … analyze_feature_x_difficulty.py
    （上乗せ = 層ごとに分割内で平均した値の差を、分割で平均 ± 標準誤差。入力変数の層は 1-3, 4-6, 7-10, 11-15, 16+）
  §8 の x7・x3・x10 … 対応のある t 検定（旧版の本文の p 値）。ここでは Wilcoxon も併記する。
  分野ラベルの合併 … run_domain_merge_ablation.py
出力: experiments/kato1014/mech_<tag>.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats

from kato1014_common import EXP, N_EQ, dump, load_records, paired, seed_means, seed_summary

METRICS = ("Recall@20", "MAP", "MAP@20")
NIN_BINS = [("1-3", lambda v: v <= 3), ("4-6", lambda v: 4 <= v <= 6), ("7-10", lambda v: 7 <= v <= 10),
            ("11-15", lambda v: 11 <= v <= 15), ("16+", lambda v: v >= 16)]
NC_BINS = [(str(k), (lambda v, k=k: v == k)) for k in range(1, 11)]


def with_t(a: dict, b: dict) -> dict:
    """paired（Wilcoxon）に対応のある t 検定の p を足す."""
    r = paired(a, b)
    seeds = sorted(set(a) & set(b))
    x = np.array([a[s] for s in seeds]); y = np.array([b[s] for s in seeds])
    r["p_ttest"] = float(stats.ttest_rel(x, y).pvalue) if len(seeds) > 1 and np.any(x != y) else None
    return r


def lift_by_bin(mode_recs, n_m, base_recs, n_b, key, bins, metric):
    rows = []
    for lbl, pred in bins:
        sub = (lambda r, pred=pred: pred(r[key]))
        m = seed_means(mode_recs, metric, n_m, sub)
        b = seed_means(base_recs, metric, n_b, sub)
        seeds = sorted(set(m) & set(b))
        if not seeds:
            rows.append({"bin": lbl, "n_records": 0}); continue
        d = np.array([m[s] - b[s] for s in seeds])
        rows.append({"bin": lbl, "n_records": sum(1 for r in mode_recs if pred(r[key])),
                     "lift_mean": float(d.mean()),
                     "lift_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                     "mode_mean": float(np.mean([m[s] for s in seeds])),
                     "base_mean": float(np.mean([b[s] for s in seeds]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    G = {}
    for name in ("reranker-2", "reranker-7", "reranker-2_Spec", "reranker-2_Svd", "reranker-2_Dom",
                 "reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom", "static"):
        recs, n = load_records(f"{a.rundir}/{name}_A__*.json")
        if recs:
            G["reranker-10S" if name == "static" else name] = (recs, n)
    braw = json.load(open(EXP / "kato1014" / "raw" / "baselines_A.json", encoding="utf-8"))["results"]
    G["baseline_w30-70"] = (braw["tfidf__none__w30-70"]["per_case"], N_EQ)
    for mg in ("orig", "L1", "K100", "K40"):
        recs, n = load_records(f"{a.rundir}/domain_merge_v2/{mg}__*.json")
        if recs:
            G[f"reranker-2_Dom@{mg}"] = (recs, n)

    SM = {g: {m: seed_means(r, m, n) for m in METRICS} for g, (r, n) in G.items()}
    out = {"rundir": a.rundir, "summary": {g: {m: seed_summary(SM[g][m]) for m in METRICS} for g in SM},
           "steps": {}, "single_set_additions": {}, "topic_additions": {}, "domain_merge": {},
           "lift_by_n_correct": {}, "lift_by_n_input": {}}

    def comp(x, y):
        if x in SM and y in SM:
            return {m: with_t(SM[x][m], SM[y][m]) for m in METRICS}
        return None

    out["steps"] = {
        "(i) baseline_w30-70 -> reranker-2": comp("reranker-2", "baseline_w30-70"),
        "(ii) reranker-2 -> reranker-7": comp("reranker-7", "reranker-2"),
        "(ii') reranker-2 -> reranker-2+Spec": comp("reranker-2_Spec", "reranker-2"),
        "(iii) reranker-7 -> reranker-10S": comp("reranker-10S", "reranker-7"),
        "total baseline_w30-70 -> reranker-10S": comp("reranker-10S", "baseline_w30-70"),
    }
    for x in ("reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom"):
        out["single_set_additions"][x] = comp(x, "reranker-7")
    out["topic_additions"] = {"x7: reranker-2+Dom vs reranker-2": comp("reranker-2_Dom", "reranker-2"),
                              "x3: reranker-2+Svd vs reranker-2": comp("reranker-2_Svd", "reranker-2"),
                              "x10: reranker-7+Dom vs reranker-7": comp("reranker-7_Dom", "reranker-7")}
    for mg in ("L1", "K100", "K40"):
        out["domain_merge"][mg] = comp(f"reranker-2_Dom@{mg}", "reranker-2")
    # 群で数える x7 は、合併しない（orig）とき元の x7 と同じ値になるはず → per-case の順位が一致するかを数える
    if "reranker-2_Dom@orig" in G and "reranker-2_Dom" in G:
        key = lambda r: (r["seed"], r["case_id"])
        A_ = {key(r): r["ranks"] for r in G["reranker-2_Dom@orig"][0]}
        B_ = {key(r): r["ranks"] for r in G["reranker-2_Dom"][0]}
        out["domain_merge"]["orig_equals_reranker-2_Dom"] = {
            "n": len(A_), "identical_ranks": sum(1 for k in A_ if B_.get(k) == A_[k])}
    if "reranker-7" in G:
        for mode in ("reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom", "reranker-10S"):
            if mode not in G:
                continue
            out["lift_by_n_correct"][mode] = {m: lift_by_bin(*G[mode], *G["reranker-7"], "n_correct", NC_BINS, m)
                                              for m in METRICS}
            out["lift_by_n_input"][mode] = {m: lift_by_bin(*G[mode], *G["reranker-7"], "n_input", NIN_BINS, m)
                                            for m in METRICS}
    for g, v in out["summary"].items():
        print(f"  {g:24s} R@20 {v['Recall@20']['mean']:.4f}±{v['Recall@20']['std']:.4f}  "
              f"MAP {v['MAP']['mean']:.4f}±{v['MAP']['std']:.4f}")
    for sect in ("steps", "single_set_additions", "topic_additions", "domain_merge"):
        for k, v in out[sect].items():
            if v and "Recall@20" in v:
                print(f"  [{sect}] {k:42s} " + "  ".join(
                    f"{m} {v[m]['mean_delta']:+.4f} (W p={v[m]['p_wilcoxon']}, t p={v[m]['p_ttest']})" for m in METRICS))
    dump(out, f"mech_{a.tag}.json")


if __name__ == "__main__":
    main()
