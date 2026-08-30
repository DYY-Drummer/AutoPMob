"""第1段の混合重み（文章 w_text ＋ 変数 w_var）ごとの被覆率スイープ.

A1 実験（第1段の変数寄り再設計）の第 1 部。訓練を伴わない被覆率＝第 2 段の
Recall@K_correct の上限を、混合重み × 候補件数 k で測る。修論 §6.7 の予測
「変数寄りの第 1 段は上限を引き上げる（k=50 で 0.920→0.961）」の一般形を与え、
第 2 部（再ランク再訓練, run_stage1_weights.sh）の条件選定に使う。

端点の検算: (0.7,0.3) と (0,1) の k=50 被覆率は既存の実測
  experiments/topic_feature_diagnosis.json E_stage1_coverage
  （blend 0.9203±0.004 / io_jaccard_only 0.9610±0.002）と一致するはず。

出力: experiments/stage1_weight_coverage.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
WEIGHTS = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]
KS = [10, 25, 50, 100, 200, 400]


def wlabel(w: tuple) -> str:
    """重み組を実験ラベルへ（例 (0.3, 0.7) → "w30-70"）."""
    return f"w{int(round(w[0] * 100)):02d}-{int(round(w[1] * 100)):02d}"


def weight_orders(ts: np.ndarray, vs: np.ndarray, weights) -> dict:
    """各重みの混合スコアで全式を降順に並べた index リストを返す."""
    return {wlabel(w): np.argsort(-(w[0] * ts + w[1] * vs)).tolist()
            for w in weights}


def run(out_json="experiments/stage1_weight_coverage.json"):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
        case_text, io_vars, jaccard,
    )
    from analyze_pfi import keep_setting_A
    from analyze_stage1_coverage import coverage, mean_sem

    eqs = load_equations()
    cases = [c for c in load_cases()
             if keep_setting_A(str(c.get("variant_type", "")))]
    ek, et, ev = [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki]
          for c in cases]
    keep = [i for i in range(len(cases)) if cl[i]]
    cases = [cases[i] for i in keep]; cl = [cl[i] for i in keep]
    n_eq = len(ek)
    print(f"設定A ケース {len(cases)} 件 / 式 {n_eq} 件 / 重み {len(WEIGHTS)} 通り")

    tfidf = TfidfVectorizer(lowercase=True, max_features=50000,
                            ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    ios = [io_vars(c) for c in cases]

    labels = [wlabel(w) for w in WEIGHTS]
    cov = {m: {k: [] for k in KS} for m in labels}
    full = {m: {k: [] for k in KS} for m in labels}
    for ci in range(len(cases)):
        corr = set(cl[ci])
        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
        vs = np.array([jaccard(ios[ci], ev[j]) for j in range(n_eq)],
                      dtype=np.float32)
        for m, order in weight_orders(ts, vs, WEIGHTS).items():
            for k in KS:
                h = coverage(order, corr, k)
                cov[m][k].append(h)
                full[m][k].append(1.0 if h == 1.0 else 0.0)

    out = {
        "config": {"setting": "A", "n_cases": len(cases), "n_eq": n_eq,
                   "weights": [list(w) for w in WEIGHTS], "ks": KS},
        "note": "被覆率は第2段 Recall@K_correct の上限。w70-30 が現行、w00-100 が変数のみ。",
        "coverage": {m: {str(k): mean_sem(cov[m][k]) for k in KS} for m in labels},
        "full_coverage_rate": {m: {str(k): float(np.mean(full[m][k])) for k in KS}
                               for m in labels},
    }
    (ROOT / out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    for m in labels:
        row = "  ".join(f"k={k}:{out['coverage'][m][str(k)]['mean']:.4f}"
                        for k in KS)
        print(f"  {m:<8s} {row}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="experiments/stage1_weight_coverage.json")
    a = ap.parse_args()
    run(out_json=a.out_json)
