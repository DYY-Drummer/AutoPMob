"""analyze_judge_agreement.py（B1: 第 2 判定器との一致度）の単体テスト.

仕様: docs/superpowers/specs/2026-09-07-b1-second-judge-design.md §2
- Cohen's κ が sklearn の cohen_kappa_score と一致する
- ペアの整列は case_id と K の一致を要求する
- 集計: 一致率・κ・番号一致・両判定器の Recall@K / coverage
"""
import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from analyze_judge_agreement import cohen_kappa, pair_labels, agreement


def test_cohen_kappa_matches_sklearn():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 2, 200).tolist()
    b = [x if rng.rand() < 0.8 else 1 - x for x in a]
    assert abs(cohen_kappa(a, b) - cohen_kappa_score(a, b)) < 1e-12
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert math.isnan(cohen_kappa([1, 1, 1], [1, 1, 1]))  # 分散なしは定義不能


def test_pair_labels():
    case = {"case_id": "c1", "n_correct": 3, "match_index": [2, 0, 5]}
    lab = pair_labels(case)
    assert lab["matched"] == [1, 0, 1]
    assert lab["topk"] == [1, 0, 0]          # K=3 なので 5 は上位 K 外
    assert lab["index"] == [2, 0, 5]


def test_agreement_aligns_and_aggregates():
    ref = [{"case_id": "c1", "n_correct": 2, "match_index": [1, 0]},
           {"case_id": "c2", "n_correct": 3, "match_index": [0, 2, 4]}]
    new = [{"case_id": "c1", "n_correct": 2, "match_index": [1, 0]},
           {"case_id": "c2", "n_correct": 3, "match_index": [0, 3, 0]}]
    out = agreement(ref, new)
    assert out["n_cases"] == 2 and out["n_pairs"] == 5
    assert out["matched"]["agree_rate"] == 0.8          # 4/5 pairs agree on matched/unmatched
    assert out["topk"]["agree_rate"] == 1.0             # topk labels: ref [1,0,0,1,0], new [1,0,0,1,0]
    assert out["index_agree_rate"] == 0.6                # exact index: c1 both, c2 first only
    assert abs(out["ref"]["Recall@K_correct"] - np.mean([0.5, 1 / 3])) < 1e-12
    assert abs(out["new"]["coverage"] - np.mean([0.5, 1 / 3])) < 1e-12


def test_agreement_rejects_misaligned_cases():
    ref = [{"case_id": "c1", "n_correct": 2, "match_index": [1, 0]}]
    new = [{"case_id": "c1", "n_correct": 3, "match_index": [1, 0, 0]}]
    with pytest.raises(ValueError):
        agreement(ref, new)
