"""kato1014_pfi.py のテスト（§5 の PFI を新しい設定でやり直す）.

シャッフルの手順は analyze_pfi.permute_feats（scope="case"、乱数 seed*1000+i）と同じであること、
並べ方は set_aware_reranker の一括の並べ直し（点数の降順、同点は候補の順）と同じであることを確かめる。
"""
import numpy as np

from analyze_pfi import permute_feats
from kato1014_pfi import case_metrics_from_scores, permute_case


def test_permute_case_matches_analyze_pfi_case_scope():
    rng = np.random.default_rng(0)
    cache = [{"feats": rng.random((n, 10)).astype(np.float32)} for n in (5, 9, 3)]
    for cols in ([5], [1, 3, 4, 5, 7, 8]):
        want = permute_feats(cache, cols, perm_seed=42 * 1000 + 3, scope="case")
        got = permute_case([c["feats"] for c in cache], cols, perm_seed=42 * 1000 + 3)
        for a, b in zip(got, want):
            assert np.array_equal(a, b)


def test_case_metrics_from_scores_uses_stable_descending_order():
    cands = [10, 11, 12, 13]
    scores = np.array([0.5, 0.9, 0.5, 0.1], dtype=np.float32)
    # 並び: 11, 10, 12, 13（0.5 の同点は候補の順）
    m = case_metrics_from_scores(cands, scores, corr={12, 13})
    assert m["ranks"] == [3, 4]
    assert m["Recall@20"] == 1.0
    assert abs(m["MAP"] - (1 / 3 + 2 / 4) / 2) < 1e-12
    assert m["Recall@K"] == 0.0
    assert abs(m["MAP@20"] - (1 / 3 + 2 / 4) / 2) < 1e-12  # 20 位までに全部あるので MAP と同じ
