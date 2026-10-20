"""kato1020_baselines.py のテスト：全式の順位（逆順列）から求める「同じ式を正解とみなした順位」が、
set_aware_reranker.equiv_ranks を全式の並びに当てた結果と同じになる."""
import numpy as np

from kato1020_baselines import equiv_from_pos, members_by_key
from set_aware_reranker import equiv_ranks


def test_equiv_from_pos_matches_equiv_ranks_on_full_order():
    rng = np.random.default_rng(0)
    n = 60
    ids = [f"k{int(x)}" for x in rng.integers(0, 15, size=n)]   # 同じ式が多い
    src = [f"s{int(x)}" for x in rng.integers(0, 4, size=n)]
    members = members_by_key(ids)
    for trial in range(200):
        order = rng.permutation(n)
        pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(1, n + 1)
        corr = set(rng.choice(n, size=int(rng.integers(1, 5)), replace=False).tolist())
        want = equiv_ranks(order.tolist(), corr, ids, miss_rank=10 ** 9, eq_src=src)
        got = equiv_from_pos(pos, corr, ids, src, members)
        assert got == want, trial
