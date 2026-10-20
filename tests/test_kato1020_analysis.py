"""kato1020_analysis.py のテスト：検定の代わりに、分割で対にして上回った・下回った数を数える."""
import pytest

from kato1020_analysis import counts


def test_counts_wins_losses_ties_and_mean_delta():
    a = {1: 0.5, 2: 0.7, 3: 0.6}
    b = {1: 0.4, 2: 0.8, 3: 0.6}
    c = counts(a, b)
    assert (c["wins"], c["losses"], c["ties"], c["n"]) == (1, 1, 1, 3)
    assert c["mean_delta"] == pytest.approx(0.0)
    assert "p_wilcoxon" not in c


def test_counts_uses_only_shared_splits():
    assert counts({1: 1.0, 2: 1.0}, {2: 0.5})["n"] == 1
