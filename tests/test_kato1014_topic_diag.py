"""kato1014_topic_diag.py のテスト：順位から求める AUC が diagnose_topic_features の定義と同じ値になる."""
import numpy as np
import pytest

from diagnose_topic_features import auc_from_groups
from kato1014_topic_diag import auc_rank


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_auc_rank_equals_pairwise_definition_with_ties(seed):
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, 5, size=7).astype(float)   # 同じ値を多く含める
    neg = rng.integers(0, 5, size=40).astype(float)
    assert auc_rank(pos, neg) == pytest.approx(auc_from_groups(pos, neg))


def test_auc_rank_empty_group_is_nan():
    assert np.isnan(auc_rank(np.array([]), np.array([1.0])))
