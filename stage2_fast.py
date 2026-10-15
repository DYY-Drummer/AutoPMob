"""第 2 段の特徴量をまとめて（ベクトル化して）計算する高速版（set_aware_reranker.py --fast 用）.

加藤先生の指示 3（2026-10-14）で第 1 段の候補数 N1 を数千〜DB 全体（11,146 式）まで広げると、
既定の計算（候補 1 件ずつ Python の集合演算、エポックごとに第 1 段を全式で計算し直す）では
時間がかかりすぎる。ここでは同じ値を、式×変数の疎行列とケースごとの使い回しで計算する。

既定の計算とビット単位で同じ値を返すことを tests/test_stage2_fast.py で確かめている:
  - stage1()                      → FastBank.stage1
  - compute_features_with_set()  → FastBank.features（既存グラフ特徴量には未対応）
  - compute_set_aware_features() → FastBank.set_features
値の作り方（float64 で割ってから float32 に入れる、など）も既定の計算に合わせてある。
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from two_stage_query_conditioned import case_text, in_vars, io_vars, out_vars

K_REF = 5  # set_aware_reranker.K_REF と同じ（一括の並べ直しの参照集合 = 第 1 段の上位 5 件）


class FastBank:
    """1 つの乱数の種（= 1 つの SVD）について、全ケース×全式の特徴量の材料を持つ."""

    def __init__(self, X_ctx, X_eq, cases, eq_vars_list, eq_domains, svd_sim):
        from sklearn.metrics.pairwise import cosine_similarity
        self.cases = cases
        self.ev = eq_vars_list
        self.ed = eq_domains
        self.svd_sim = svd_sim
        # 文章類似度：既定の計算はケース 1 行ずつ cosine_similarity(X_ctx[ci], X_eq) を取る。
        # 行ごとの計算は独立なので、まとめて計算しても同じ値になる。
        self.TS = cosine_similarity(X_ctx, X_eq)
        self.n_eq = len(eq_vars_list)

        ios = [io_vars(c) for c in cases]
        vocab: dict[str, int] = {}
        for s in list(eq_vars_list) + ios + [in_vars(c) for c in cases] + [out_vars(c) for c in cases]:
            for v in s:
                vocab.setdefault(v, len(vocab))
        self.vocab = vocab
        rows, cols = [], []
        for i, s in enumerate(eq_vars_list):
            for v in s:
                rows.append(i); cols.append(vocab[v])
        self.A = sparse.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, cols)),
                                   shape=(self.n_eq, max(1, len(vocab))))
        self.nv = np.array([len(s) for s in eq_vars_list], dtype=np.int64)

        # 分野ラベル：x7 は小文字にして部分文字列で比べる、x10 は文字列の完全一致
        low = [d.lower() for d in eq_domains]
        self._low_u, self._low_id = np.unique(np.array(low, dtype=object), return_inverse=True)
        self._dom_u, self._dom_id = np.unique(np.array(eq_domains, dtype=object), return_inverse=True)
        self._dom_index = {d: i for i, d in enumerate(self._dom_u)}

        self._J: dict[int, np.ndarray] = {}
        self._order: dict[tuple, np.ndarray] = {}

    # --- 小道具 ---------------------------------------------------------------
    def _vec(self, s) -> np.ndarray:
        x = np.zeros(self.A.shape[1], dtype=np.int64)
        for v in s:
            x[self.vocab[v]] = 1
        return x

    def jaccard_row(self, ci) -> np.ndarray:
        """jaccard(io_vars(case), V(e)) を全式について float32 で（stage1() の vs と同じ値）."""
        J = self._J.get(ci)
        if J is None:
            io = io_vars(self.cases[ci])
            inter = self.A @ self._vec(io)
            union = len(io) + self.nv - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                J = np.where(union > 0, inter / np.maximum(union, 1), 0.0).astype(np.float32)
            self._J[ci] = J
        return J

    # --- 第 1 段 --------------------------------------------------------------
    def stage1(self, ci, k, w_text=0.7, w_var=0.3) -> list[int]:
        """stage1(ci, ...) と同じ上位 k 件（同じ dtype の点数を同じ argsort で並べる）."""
        key = (ci, w_text, w_var)
        order = self._order.get(key)
        if order is None:
            s = w_text * self.TS[ci] + w_var * self.jaccard_row(ci)
            order = np.argsort(-s).astype(np.int32)
            self._order[key] = order
        return order[:k].tolist()

    # --- 第 2 段の特徴量 ------------------------------------------------------
    def features(self, ci, cands, set_mask=()) -> np.ndarray:
        """compute_features_with_set(..., use_existing_graph=False, set_aware_mask) と同じ値."""
        c = self.cases[ci]
        inv, outv, io = in_vars(c), out_vars(c), io_vars(c)
        idx = np.asarray(cands, dtype=np.int64)
        n = len(idx)
        F = np.zeros((n, 7 + len(set_mask)), dtype=np.float32)
        if n == 0:
            return F
        Ac = self.A[idx]
        nv = self.nv[idx]
        F[:, 0] = self.TS[ci, idx]
        F[:, 1] = self.jaccard_row(ci)[idx]
        F[:, 2] = self.svd_sim[ci][idx]
        if inv:
            F[:, 3] = (Ac @ self._vec(inv)) / len(inv)
        if outv:
            F[:, 4] = (Ac @ self._vec(outv)) / len(outv)
        io_cnt = Ac @ self._vec(io)
        with np.errstate(divide="ignore", invalid="ignore"):
            F[:, 5] = np.where(nv > 0, io_cnt / np.maximum(nv, 1), 0.0)
        dl = case_text(c).lower()
        m = np.array([1.0 if (d and d in dl) or (dl and dl in d) else 0.0 for d in self._low_u])
        F[:, 6] = m[self._low_id[idx]]
        if set_mask:
            S = self.set_features(ci, cands, None)
            col = {"Comp": 0, "Coh": 1, "Dom": 2, "OutNov": 3}
            for i, name in enumerate(set_mask):
                if name in col:
                    F[:, 7 + i] = S[:, col[name]]
        return F

    def set_features(self, ci, cands, ref=None, k_ref=K_REF) -> np.ndarray:
        """compute_set_aware_features(cands, io_vars, ..., output_vars, ref_indices=ref) と同じ値.

        列: 0 = 補完性、1 = 一貫性、2 = 分野ラベルの一致度（集合）、3 = 出力補完性。
        参照集合 = ref（逐次選択の既選択集合）か、無ければ cands の先頭 k_ref 件。
        参照集合に入っている候補は、自分を除いた参照集合と比べる（その行だけ 1 件ずつ計算）。
        """
        c = self.cases[ci]
        qv, outv = io_vars(c), out_vars(c)
        idx = np.asarray(cands, dtype=np.int64)
        n = len(idx)
        F = np.zeros((n, 4), dtype=np.float32)
        if n == 0:
            return F
        ref_set = list(ref) if ref is not None else list(cands[:k_ref])
        U = set()
        for r in ref_set:
            U |= self.ev[r]
        u = self._vec(U)
        Ac = self.A[idx]
        nv = self.nv[idx]
        if qv:
            F[:, 0] = (Ac @ (self._vec(qv) * (1 - u))) / len(qv)
        with np.errstate(divide="ignore", invalid="ignore"):
            F[:, 1] = np.where(nv > 0, (Ac @ u) / np.maximum(nv, 1), 0.0)
        L = len(ref_set)
        if L:
            cnt = Counter(self.ed[r] for r in ref_set if self.ed[r])
            per_dom = np.zeros(len(self._dom_u), dtype=np.int64)
            for d, v in cnt.items():
                per_dom[self._dom_index[d]] = v
            F[:, 2] = per_dom[self._dom_id[idx]] / L
        if outv:
            F[:, 3] = (Ac @ (self._vec(outv) * (1 - u))) / len(outv)

        # 参照集合に入っている候補：自分を除いた参照集合で計算し直す（既定の計算と同じ式）
        refs = set(ref_set)
        for k, j in enumerate(cands):
            if j not in refs:
                continue
            local_union = set()
            for r in ref_set:
                if r != j:
                    local_union |= self.ev[r]
            local_domains = [self.ed[r] for r in ref_set if r != j]
            cv = self.ev[j]
            F[k, 0] = len((cv & qv) - local_union) / len(qv) if qv else 0.0
            F[k, 1] = len(cv & local_union) / len(cv) if cv else 0.0
            if local_domains:
                same = sum(1 for d in local_domains if d == self.ed[j] and d)
                F[k, 2] = same / len(local_domains)
            else:
                F[k, 2] = 0.0
            F[k, 3] = len((cv & outv) - local_union) / len(outv) if outv else 0.0
        return F
