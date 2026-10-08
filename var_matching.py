"""変数の照合と特徴量（説明版と記号版を同じコードで計算する）.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.4
- 「同じ変数」= 類似度 ≥ τ。記号版は完全一致の 0/1 行列と τ = SYMBOL_TAU で同じ関数を通す。
- 1 対 1 の対応は二部グラフの最大マッチング（Hopcroft–Karp）。
- 第 1 段（全式）は |M| の上限 min(対応先のある Q の数, 対応先のある V の数) で近似する。
- case_features は set_aware_reranker.compute_features_with_set（reranker-10S）と同じ 10 列を返す。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

SYMBOL_TAU = 1.0  # 記号版：一致 = 1.0 / 不一致 = 0.0 の行列と組で使う
K_REF = 5         # set_aware_reranker.K_REF と同じ（参照集合 = 第 1 段の上位 5 件）


def max_matching(adj: np.ndarray) -> np.ndarray:
    """bool 行列 adj（照会の変数 × 式の変数）の最大マッチング。各行の対応列（なければ -1）."""
    nq, nv = adj.shape
    if nq == 0 or nv == 0 or not adj.any():
        return np.full(nq, -1, dtype=np.int64)
    m = maximum_bipartite_matching(csr_matrix(adj.astype(np.int8)), perm_type="column")
    return np.asarray(m, dtype=np.int64)


def occurrence_incidence(eq_ptr: np.ndarray) -> csr_matrix:
    """変数の出現（式ごとに連続して並ぶ）→ 式 の 0/1 行列 (n_occ × n_eq)。変数のない式は空の列."""
    eq_ptr = np.asarray(eq_ptr, dtype=np.int64)
    n_eq, n_occ = len(eq_ptr) - 1, int(eq_ptr[-1])
    cols = np.repeat(np.arange(n_eq), np.diff(eq_ptr))
    return csr_matrix((np.ones(n_occ, dtype=np.float32), (np.arange(n_occ), cols)),
                      shape=(n_occ, n_eq))


def stage1_match_counts(hit: np.ndarray, inc: csr_matrix) -> np.ndarray:
    """第 1 段の近似 |M|（全式）。hit: bool (|Q| × n_occ)、inc: occurrence_incidence.

    min(対応先のある Q の数, 対応先のある V の数) は最大マッチングの大きさの上限で、
    完全一致の照合なら |Q ∩ V| と等しい。
    """
    n_eq = inc.shape[1]
    if hit.shape[0] == 0:
        return np.zeros(n_eq, dtype=np.int64)
    per_q = np.asarray(inc.T @ hit.T.astype(np.float32)).T          # (|Q| × n_eq)
    n_q = (per_q > 0).sum(axis=0)
    n_v = np.asarray(inc.T @ hit.any(axis=0).astype(np.float32)).ravel()
    return np.minimum(n_q, n_v).astype(np.int64)


def jaccard_from_counts(m, nq: int, n_vars) -> np.ndarray:
    """m / (|Q| + |V| − m)（分母 0 は 0）を float32 で。two_stage_query_conditioned.jaccard と同じ値."""
    m = np.asarray(m, dtype=np.float64)
    union = nq + np.asarray(n_vars, dtype=np.float64) - m
    out = np.zeros(len(m), dtype=np.float64)
    np.divide(m, union, out=out, where=union > 0)
    return out.astype(np.float32)


def youden_tau(sims, labels) -> float:
    """正例 (1) と負例 (0) を最もよく分ける τ（Youden の J = 真陽性率 − 偽陽性率 が最大）。sim ≥ τ を正と判定."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(np.asarray(labels), np.asarray(sims, dtype=np.float64))
    return float(thr[int(np.argmax(tpr - fpr))])


def domain_flags(cands: list, eq_domains: list, ctx: str) -> np.ndarray:
    """f6（分野の一致）。compute_features_with_set と同じ規則（小文字の部分文字列）."""
    dl = ctx.lower()
    out = np.zeros(len(cands), dtype=np.float32)
    for k, j in enumerate(cands):
        d = eq_domains[j].lower()
        out[k] = 1.0 if (d and d in dl) or (dl and dl in d) else 0.0
    return out


def case_features(ts_c, svd_c, f6_c, q_in, q_out, QC, CR, cptr, cand_doms, tau, k_ref=K_REF):
    """reranker-10S の 10 列 [f0, f1, f2, f3, f4, f5, f6, gComp, gCoh, gDom] と各候補の |M|.

    ts_c, svd_c, f6_c: 候補ごとの f0, f2, f6
    q_in, q_out: 照会の各変数が入力か・出力か（bool, |Q|）
    QC: 照会の変数 × 候補の変数（候補の順に連結）の類似度 (|Q| × cptr[-1])
    CR: 候補の変数 × 参照集合（上位 k_ref 件）の変数の類似度 (cptr[-1] × cptr[k_ref])
    cptr: 候補 k の変数は列 cptr[k]:cptr[k+1]
    cand_doms: 候補の分野（gDom 用、正規化済み）
    """
    n = len(cptr) - 1
    nq = QC.shape[0]
    q_in = np.asarray(q_in, dtype=bool)
    q_out = np.asarray(q_out, dtype=bool)
    n_in, n_out = int(q_in.sum()), int(q_out.sum())
    kr = min(k_ref, n)
    ref_end = int(cptr[kr])
    QR = QC[:, :ref_end] >= tau
    CRb = CR >= tau
    feats = np.zeros((n, 10), dtype=np.float32)
    m_exact = np.zeros(n, dtype=np.int64)
    for k in range(n):
        a, b = int(cptr[k]), int(cptr[k + 1])
        nv = b - a
        matched = max_matching(QC[:, a:b] >= tau) >= 0
        m = int(matched.sum())
        m_exact[k] = m
        union = nq + nv - m
        feats[k, 0] = ts_c[k]
        feats[k, 1] = m / union if union else 0.0
        feats[k, 2] = svd_c[k]
        feats[k, 3] = int((matched & q_in).sum()) / n_in if n_in else 0.0
        feats[k, 4] = int((matched & q_out).sum()) / n_out if n_out else 0.0
        feats[k, 5] = m / nv if nv else 0.0
        feats[k, 6] = f6_c[k]
        keep = np.ones(ref_end, dtype=bool)       # 参照集合の変数の列（自分自身を除く）
        if k < kr:
            keep[a:b] = False
            ref = [r for r in range(kr) if r != k]
        else:
            ref = list(range(kr))
        q_cov = QR[:, keep].any(axis=1)
        feats[k, 7] = int((matched & ~q_cov).sum()) / nq if nq else 0.0
        if nv:
            feats[k, 8] = int(CRb[a:b][:, keep].any(axis=1).sum()) / nv
        if ref:
            same = sum(1 for r in ref if cand_doms[r] == cand_doms[k] and cand_doms[r])
            feats[k, 9] = same / len(ref)
    return feats, m_exact
