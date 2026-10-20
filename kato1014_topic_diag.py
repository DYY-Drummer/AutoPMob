"""§8（2026-10-14 の指示 5）：話題の特徴量の AUC 表と候補内の相関を、新しい設定でやり直す.

旧版: diagnose_topic_features.py（AUC 表、1,823 ケースの平均 ± 標準誤差、SVD の乱数 42）と
      analyze_case_outcomes.py（候補内の相関 corr(x1, x2)、乱数の種 42 の分割のテスト 423 ケース）。
      どちらも第 1 段の重みは 0.7/0.3、候補は 50 件に固定されていた。
新版: 第 1 段の重み 0.3/0.7、候補数 --top-k。特徴量は stage2_fast.FastBank で全式について計算できるので、
      全 DB の AUC も 10 特徴量すべてで出す（旧版は x1, x2, x3 だけ）。x8〜x10 の参照集合は第 1 段の上位 5 件。
訓練はしない（特徴量の値だけで正解式と不正解式を分けられる度合いを見る）。

AUC = P(正解式の値 > 不正解式の値) + 0.5 P(同じ値)。ケースごとに求め、ケースで平均する。
出力: experiments/kato1014/topic_diag_<tag>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_topic_features import _mean_sem
from kato1014_common import OUT

FEATS = ["x1_text_sim", "x2_io_jaccard", "x3_svd_sim", "x4_input_cov", "x5_output_cov",
         "x6_specificity", "x7_domain", "x8_comp", "x9_coh", "x10_dom_set"]


def auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    """auc_from_groups と同じ値を、順位（同順位は平均順位）から求める（大きな不正解の集合向け）."""
    from scipy.stats import rankdata
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    u = r[:pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def within_case_corr(x, y) -> float:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--svd-seed", type=int, default=42)
    ap.add_argument("--w-text", type=float, default=0.3)
    ap.add_argument("--w-var", type=float, default=0.7)
    ap.add_argument("--exclude", default=None, help="外すケースの一覧の JSON（2026-10-20）")
    ap.add_argument("--out-dir", default=None, help="出力先（既定 experiments/kato1014/）")
    a = ap.parse_args()

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from analyze_stage1_n1 import load_setting_A, split_for
    from stage2_fast import FastBank
    from two_stage_query_conditioned import case_text

    D = load_setting_A(a.exclude)
    cases, ek, et, ev, ed, cl, cs = (D[k] for k in ("cases", "ek", "et", "ev", "ed", "cl", "cs"))
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    svd = TruncatedSVD(n_components=256, random_state=a.svd_seed)
    E = svd.fit_transform(X_eq); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    bank = FastBank(X_ctx, X_eq, cases, ev, ed, Q @ E.T)
    n_eq = len(ek)
    mask = ("Comp", "Coh", "Dom")

    auc_full = {f: [] for f in FEATS}
    auc_cand = {f: [] for f in FEATS}
    corr_ts_vs = {}
    for ci in range(len(cases)):
        corr = set(cl[ci])
        if not corr:
            continue
        order = bank.stage1(ci, n_eq, a.w_text, a.w_var)   # 全式を s1 の順に（参照集合 = 先頭 5 件）
        F = bank.features(ci, order, mask)
        pos = np.array([j in corr for j in order])
        cand = np.zeros(n_eq, dtype=bool); cand[:a.top_k] = True
        for k, f in enumerate(FEATS):
            auc_full[f].append(auc_rank(F[pos, k], F[~pos, k]))
            pc, nc = F[pos & cand, k], F[~pos & cand, k]
            auc_cand[f].append(auc_rank(pc, nc) if pc.size and nc.size else float("nan"))
        corr_ts_vs[ci] = within_case_corr(F[cand, 0], F[cand, 1])

    _, _, te42 = split_for(cases, cl, ek, cs, 42)
    out = {"config": {**vars(a), "n_cases": sum(1 for c in cl if c), "n_eq": n_eq,
                      "stage1": f"argsort({a.w_text}*text_sim + {a.w_var}*io_jaccard)[:{a.top_k}]",
                      "reference_set": "第 1 段の上位 5 件（候補自身は除く）"},
           "auc_full_db": {f: _mean_sem(v) for f, v in auc_full.items()},
           "auc_within_candidates": {f: _mean_sem(v) for f, v in auc_cand.items()},
           "corr_text_jaccard_within_candidates_seed42_test": _mean_sem([corr_ts_vs[ci] for ci in te42]),
           "n_test_seed42": len(te42)}
    out_dir = Path(a.out_dir) if a.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"topic_diag_{a.tag}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {p}")
    for f in FEATS:
        print(f"  {f:16s} 全DB {out['auc_full_db'][f]['mean']:.3f}±{out['auc_full_db'][f]['sem']:.3f}"
              f"  候補{a.top_k}件 {out['auc_within_candidates'][f]['mean']:.3f}±{out['auc_within_candidates'][f]['sem']:.3f}")
    c = out["corr_text_jaccard_within_candidates_seed42_test"]
    print(f"  候補内 corr(x1, x2)（seed 42 のテスト {len(te42)} 件）: {c['mean']:+.3f}")


if __name__ == "__main__":
    main()
