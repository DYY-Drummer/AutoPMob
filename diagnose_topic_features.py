"""話題特徴（text_sim / svd_sim / domain）が再ランクに寄与しない原因の診断.

PFI で話題群の重要度が 0 だった結果を受け、モデルを訓練せずデータと
stage1 の性質だけから原因を切り分ける：

  A. ケース間ベクトル類似   : TF-IDF→SVD のクエリベクトルがケース間で
                              どれだけ離れているか（先生仮説の直接検証）。
                              生 TF-IDF 空間と SVD 空間の両方で測る。
  B. 範囲制限               : stage1（0.7*text_sim + 0.3*jaccard の上位50件）
                              が候補内の svd_sim / text_sim の分散をどれだけ
                              潰しているか（std 比）。
  C. 候補内識別力           : 候補50件の中で各特徴が正解行と不正解行を
                              分離できるか（ケース別 AUC と sep）。
  D. 全DB識別力             : 候補に絞る前の全11,146式に対する AUC。
                              C との差が「信号は stage1 で使い切られた」量。
  E. stage1 被覆分解        : 候補50件に正解式が入る割合（被覆率）を
                              blend / text のみ / jaccard のみ / svd のみで比較。

出力: experiments/topic_feature_diagnosis.json,
      docs/figures/fig_topic_diagnosis.png / .pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 純関数（tests/test_topic_diagnosis.py で検証）
# ---------------------------------------------------------------------------

def mean_pairwise_cos(V: np.ndarray) -> float:
    """L2正規化済み行ベクトル集合の非対角ペア平均コサイン（閉形式・厳密）.

    mean = (||Σv||^2 - n) / (n(n-1))
    """
    n = V.shape[0]
    s = np.asarray(V.sum(axis=0)).ravel()
    return float((s @ s - n) / (n * (n - 1)))


def sample_pairwise_cos(V: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    """異なる行のランダムペア n_pairs 組のコサイン値（分位点推定用）."""
    rng = np.random.RandomState(seed)
    n = V.shape[0]
    i = rng.randint(0, n, size=n_pairs)
    j = rng.randint(0, n - 1, size=n_pairs)
    j = np.where(j >= i, j + 1, j)  # i==j を回避
    if hasattr(V, "multiply"):  # scipy sparse
        vals = np.asarray(V[i].multiply(V[j]).sum(axis=1)).ravel()
    else:
        vals = np.einsum("ij,ij->i", V[i], V[j])
    return vals


def auc_from_groups(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U に基づく AUC = P(pos > neg) + 0.5*P(同値)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (pos.size * neg.size))


def sep_groups(pos: np.ndarray, neg: np.ndarray) -> float:
    """分離度 sep = mean(正解行) - mean(不正解行)（片側が空なら nan）."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(pos.mean() - neg.mean())


def within_candidate_spread(full_vals: np.ndarray, cand_vals: np.ndarray) -> dict:
    """全DB上と候補内の値のばらつき比較。ratio = std_cand / std_full."""
    sf = float(full_vals.std())
    sc = float(cand_vals.std())
    return {"std_full": sf, "std_cand": sc,
            "ratio": (sc / sf) if sf > 0 else float("nan"),
            "mean_full": float(full_vals.mean()), "mean_cand": float(cand_vals.mean())}


def stage1_coverage(cands, corr: set) -> float:
    """候補リストに正解式が入っている割合 |corr ∩ cands| / |corr|."""
    if not corr:
        return float("nan")
    return len(corr & set(cands)) / len(corr)


def top_k_by_score(scores: np.ndarray, k: int) -> list:
    return np.argsort(-scores)[:k].tolist()


def rank_auc_full_db(scores: np.ndarray, corr: set) -> float:
    """全DBスコア列に対する AUC（正解 vs それ以外）."""
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[list(corr)] = True
    return auc_from_groups(scores[mask], scores[~mask])


def _mean_sem(arr) -> dict:
    a = np.asarray([x for x in arr if x == x], dtype=float)  # nan除外
    if a.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "n": int(a.size)}


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

def run(seed: int = 42, n_pairs: int = 200_000,
        out_json: str = "experiments/topic_feature_diagnosis.json",
        out_fig: str = "docs/figures/fig_topic_diagnosis.png") -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize

    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
        case_text, io_vars, in_vars, out_vars, jaccard, stage1,
    )
    from set_aware_reranker import compute_features_with_set
    from analyze_pfi import FEATURES, SET_MASK, TOP_K, keep_setting_A

    eqs = load_equations()
    cases_all = load_cases()
    cases = [c for c in cases_all if keep_setting_A(str(c.get("variant_type", "")))]
    ek, et, ev, ed = [], [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or ""))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki]
          for c in cases]
    keep = [i for i in range(len(cases)) if cl[i]]
    cases = [cases[i] for i in keep]
    cl = [cl[i] for i in keep]
    n_eq = len(ek)
    print(f"設定A ケース {len(cases)} 件 / 式 {n_eq} 件")

    # --- analyze_pfi.train_and_cache と同一の TF-IDF → SVD 構成 ---
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])          # stage1 / text_sim 用
    X_ctx_io = tfidf.transform([case_text(c, io=True) for c in cases])  # svd クエリ用
    svd = TruncatedSVD(n_components=256, random_state=seed)
    E = svd.fit_transform(X_eq)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    Q = svd.transform(X_ctx_io)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    svd_sim = (Q @ E.T).astype(np.float32)
    ts_all = cosine_similarity(X_ctx, X_eq).astype(np.float32)

    ios = [io_vars(c) for c in cases]

    # --- A. ケース間ベクトル類似 -------------------------------------------
    Xn = normalize(X_ctx_io)  # 生 TF-IDF（L2 正規化）
    def _quant(vals):
        qs = np.percentile(vals, [5, 25, 50, 75, 95])
        return {"p5": float(qs[0]), "p25": float(qs[1]), "median": float(qs[2]),
                "p75": float(qs[3]), "p95": float(qs[4]),
                "share_gt_0.5": float(np.mean(vals > 0.5)),
                "share_gt_0.8": float(np.mean(vals > 0.8))}
    pairs_svd = sample_pairwise_cos(Q, n_pairs, seed=0)
    pairs_raw = sample_pairwise_cos(Xn, n_pairs, seed=0)
    sec_A = {
        "query_svd_space": {"mean_exact": mean_pairwise_cos(Q), **_quant(pairs_svd)},
        "query_raw_tfidf": {"mean_exact": mean_pairwise_cos(Xn),  # 疎のまま閉形式
                            **_quant(pairs_raw)},
        "equation_svd_space": {"mean_exact": mean_pairwise_cos(E),
                               **_quant(sample_pairwise_cos(E, n_pairs, seed=0))},
        "n_pairs_sampled": n_pairs,
    }

    # --- ケース別ループ（B〜E をまとめて計算） ------------------------------
    n_feat_names = [name for _, name, _ in FEATURES]
    feat_groups = {name: g for _, name, g in FEATURES}
    auc_cand = {name: [] for name in n_feat_names}
    sep_cand = {name: [] for name in n_feat_names}
    spread_svd, spread_ts = [], []
    cov_blend, cov_ts, cov_vs, cov_svd = [], [], [], []
    auc_full_svd, auc_full_ts, auc_full_vs = [], [], []
    n_no_pos_in_cand = 0

    for ci, c in enumerate(cases):
        corr = set(cl[ci])
        ts = ts_all[ci]
        sv = svd_sim[ci]
        vs = np.array([jaccard(ios[ci], ev[j]) for j in range(n_eq)], dtype=np.float32)

        # E. stage1 被覆分解
        cands = stage1(ci, X_ctx, X_eq, ios[ci], ev, TOP_K)
        cov_blend.append(stage1_coverage(cands, corr))
        cov_ts.append(stage1_coverage(top_k_by_score(ts, TOP_K), corr))
        cov_vs.append(stage1_coverage(top_k_by_score(vs, TOP_K), corr))
        cov_svd.append(stage1_coverage(top_k_by_score(sv, TOP_K), corr))

        # D. 全DB識別力
        auc_full_svd.append(rank_auc_full_db(sv, corr))
        auc_full_ts.append(rank_auc_full_db(ts, corr))
        auc_full_vs.append(rank_auc_full_db(vs, corr))

        # B. 範囲制限（svd_sim / text_sim）
        spread_svd.append(within_candidate_spread(sv, sv[cands]))
        spread_ts.append(within_candidate_spread(ts, ts[cands]))

        # C. 候補内識別力（10特徴）
        pos_in = [j for j in cands if j in corr]
        neg_in = [j for j in cands if j not in corr]
        if not pos_in or not neg_in:
            n_no_pos_in_cand += 1
            continue
        feats = compute_features_with_set(
            cands, ts, ios[ci], in_vars(c), out_vars(c), sv, ev, ed,
            case_text(c), None, ios[ci], False, SET_MASK)
        pos_mask = np.array([j in corr for j in cands], dtype=bool)
        for col, name, _g in FEATURES:
            pv, nv = feats[pos_mask, col], feats[~pos_mask, col]
            auc_cand[name].append(auc_from_groups(pv, nv))
            sep_cand[name].append(sep_groups(pv, nv))

    out = {
        "config": {"setting": "A", "n_cases": len(cases), "n_eq": n_eq,
                   "top_k": TOP_K, "svd_components": 256, "svd_seed": seed,
                   "stage1": "argsort(0.7*text_sim + 0.3*io_jaccard)[:50]"},
        "A_case_vector_similarity": sec_A,
        "B_range_restriction": {
            "svd_sim": {k: _mean_sem([d[k] for d in spread_svd])
                        for k in ("std_full", "std_cand", "ratio", "mean_full", "mean_cand")},
            "text_sim": {k: _mean_sem([d[k] for d in spread_ts])
                         for k in ("std_full", "std_cand", "ratio", "mean_full", "mean_cand")},
        },
        "C_within_candidate_auc": {
            name: {"group": feat_groups[name], "auc": _mean_sem(auc_cand[name]),
                   "sep": _mean_sem(sep_cand[name]),
                   "share_auc_gt_0.5": float(np.mean(np.asarray(
                       [a for a in auc_cand[name] if a == a]) > 0.5))}
            for name in n_feat_names
        },
        "C_n_cases_excluded_no_pos_or_neg": n_no_pos_in_cand,
        "D_full_db_auc": {
            "svd_sim": _mean_sem(auc_full_svd),
            "text_sim": _mean_sem(auc_full_ts),
            "io_jaccard": _mean_sem(auc_full_vs),
        },
        "E_stage1_coverage": {
            "blend_0.7ts_0.3vs": _mean_sem(cov_blend),
            "text_sim_only": _mean_sem(cov_ts),
            "io_jaccard_only": _mean_sem(cov_vs),
            "svd_sim_only": _mean_sem(cov_svd),
            "share_full_coverage_blend": float(np.mean(np.asarray(cov_blend) == 1.0)),
            "share_zero_coverage_blend": float(np.mean(np.asarray(cov_blend) == 0.0)),
        },
    }

    Path(ROOT / out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    _make_figure(out, pairs_svd, pairs_raw, ROOT / out_fig)
    print(f"Saved figure: {out_fig}")
    _print_summary(out)
    return out


def _print_summary(out):
    A = out["A_case_vector_similarity"]
    print("\n=== A. ケース間クエリベクトル類似（コサイン） ===")
    for k, lab in (("query_svd_space", "SVD空間(256次元)"),
                   ("query_raw_tfidf", "生TF-IDF空間"),
                   ("equation_svd_space", "式ベクトル(SVD空間)")):
        d = A[k]
        print(f"  {lab:<18s} mean={d['mean_exact']:.3f} median={d['median']:.3f} "
              f"p95={d['p95']:.3f} share>0.5={d['share_gt_0.5']:.1%}")
    B = out["B_range_restriction"]
    print("=== B. stage1候補内の範囲制限（std比 候補内/全DB） ===")
    for k in ("svd_sim", "text_sim"):
        print(f"  {k:<9s} ratio={B[k]['ratio']['mean']:.3f}±{B[k]['ratio']['sem']:.3f} "
              f"(全DB std={B[k]['std_full']['mean']:.4f} → 候補内 {B[k]['std_cand']['mean']:.4f})")
    print("=== C. 候補内AUC（正解行 vs 不正解行、ケース平均±SEM） ===")
    C = out["C_within_candidate_auc"]
    for name, d in sorted(C.items(), key=lambda x: -(x[1]["auc"]["mean"] or 0)):
        print(f"  {name:<12s} [{d['group']:<5s}] AUC={d['auc']['mean']:.3f}±{d['auc']['sem']:.3f} "
              f"sep={d['sep']['mean']:+.4f}")
    D = out["D_full_db_auc"]
    print("=== D. 全DB AUC（候補に絞る前） ===")
    for k, d in D.items():
        print(f"  {k:<12s} AUC={d['mean']:.3f}±{d['sem']:.3f}")
    E = out["E_stage1_coverage"]
    print("=== E. stage1被覆率（正解式が候補50件に入る割合） ===")
    for k in ("blend_0.7ts_0.3vs", "text_sim_only", "io_jaccard_only", "svd_sim_only"):
        d = E[k]
        print(f"  {k:<18s} {d['mean']:.3f}±{d['sem']:.3f}")
    print(f"  完全被覆ケース率(blend) = {E['share_full_coverage_blend']:.1%}, "
          f"被覆ゼロ = {E['share_zero_coverage_blend']:.1%}")



# 図の日本語ラベル（作文ルール A6：プログラム変数名は文書・論文で使わない）
JA_LABEL = {
    "text_sim": "文章類似度", "io_jaccard": "変数の一致度", "svd_sim": "意味の近さ",
    "input_cov": "入力変数の被覆", "output_cov": "出力変数の被覆",
    "specificity": "式の特化度", "domain": "分野の一致",
    "gComp": "補完性", "gCoh": "一貫性", "gDom": "分野の一致（集合版）",
}


def _make_figure(out, pairs_svd, pairs_raw, out_png):
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

    # (a) ケース間クエリベクトルのコサイン類似分布
    bins = np.linspace(-0.1, 1.0, 56)
    axes[0].hist(pairs_raw, bins=bins, alpha=0.55, color="#9467bd",
                 label="生TF-IDF空間", density=True)
    axes[0].hist(pairs_svd, bins=bins, alpha=0.55, color="#d62728",
                 label="SVD 空間（意味の近さが使う 256 次元）", density=True)
    med_raw = float(np.median(pairs_raw)); med_svd = float(np.median(pairs_svd))
    axes[0].axvline(med_raw, color="#9467bd", ls="--", lw=1.5)
    axes[0].axvline(med_svd, color="#d62728", ls="--", lw=1.5)
    axes[0].set_title("(a) ケースどうしのベクトルのコサイン類似度の分布")
    axes[0].set_xlabel("ケースの組のコサイン類似度（破線は中央値）")
    axes[0].set_ylabel("確率密度")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.25)

    # (b) 候補内AUC（mean±SEM）
    C = out["C_within_candidate_auc"]
    names = sorted(C.keys(), key=lambda n: C[n]["auc"]["mean"])
    y = np.arange(len(names))
    vals = [C[n]["auc"]["mean"] for n in names]
    errs = [C[n]["auc"]["sem"] for n in names]
    cols = ["#1f77b4" if C[n]["group"] == "var" else "#d62728" for n in names]
    axes[1].barh(y, vals, xerr=errs, color=cols,
                 error_kw=dict(capsize=3, lw=1))
    axes[1].axvline(0.5, color="#555555", lw=1.2, ls="--")
    axes[1].text(0.502, len(names) - 0.4, "AUC=0.5（識別力なし）", fontsize=8, color="#555555")
    axes[1].set_yticks(y); axes[1].set_yticklabels([JA_LABEL.get(n, n) for n in names])
    axes[1].set_xlim(0.3, 1.0)
    axes[1].set_title("(b) 第 1 段が選んだ候補 50 件の中での識別力")
    axes[1].set_xlabel("候補の中での AUC（ケース平均±標準誤差。青＝変数の重なり、赤＝話題の近さ）")
    axes[1].grid(alpha=0.25, axis="x")

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png).rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pairs", type=int, default=200_000)
    ap.add_argument("--out-json", default="experiments/topic_feature_diagnosis.json")
    ap.add_argument("--out-fig", default="docs/figures/fig_topic_diagnosis.png")
    a = ap.parse_args()
    run(seed=a.seed, n_pairs=a.n_pairs, out_json=a.out_json, out_fig=a.out_fig)
