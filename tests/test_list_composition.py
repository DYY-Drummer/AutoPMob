"""analyze_list_composition.py（上位 20 件の内訳と、クエリの変数にすべて含まれる式の数）の単体テスト."""
import pytest

from analyze_list_composition import composition, covered, group_stats, input_bin, k_bin, mean_sd


def test_covered_requires_every_variable_in_the_query_and_at_least_one_variable():
    assert covered({"x", "y"}, {"x", "y", "z"})
    assert not covered({"x", "w"}, {"x", "y", "z"})
    assert not covered(set(), {"x"})          # 変数の記録がない式は数えない


def test_composition_sorts_the_entries_of_one_list():
    cov = {"a": True, "b": True, "c": False, "g": True}.get
    out = composition(["g", "a", "c", None, "b"], correct={"g"}, is_covered=cov, top_n=20)
    assert out == {"correct": 1, "covered": 2, "outside": 1, "invalid": 1, "empty": 15}
    assert sum(out.values()) == 20


def test_composition_reads_only_the_top_n_entries():
    out = composition(["a", "b", "c"], correct={"c"}, is_covered=lambda k: True, top_n=2)
    assert out == {"correct": 0, "covered": 2, "outside": 0, "invalid": 0, "empty": 0}


def test_bins_follow_the_groups_of_the_tables():
    assert [k_bin(k) for k in (1, 2, 3, 4, 6, 7, 10)] == ["1", "2-3", "2-3", "4-6", "4-6", "7-10", "7-10"]
    assert [input_bin(n) for n in (1, 5, 6, 10, 11, 15, 16, 20, 21, 60)] == \
        ["<=5", "<=5", "6-10", "6-10", "11-15", "11-15", "16-20", "16-20", ">=21", ">=21"]


def test_group_stats_gives_recall_complete_share_and_missed_equations():
    recs = [{"recall": 1.0, "complete": 1.0, "n_gt": 2, "missed": 0.0},
            {"recall": 0.5, "complete": 0.0, "n_gt": 4, "missed": 2.0}]
    assert group_stats(recs) == {"n": 2, "recall": 0.75, "complete": 0.5, "missed": 2.0, "n_gt": 6}
    assert group_stats([]) == {"n": 0, "recall": None, "complete": None, "missed": 0.0, "n_gt": 0}


def test_mean_sd_skips_missing_values_and_reports_how_many_it_used():
    out = mean_sd([0.5, None, 1.0])
    assert out["mean"] == pytest.approx(0.75) and out["n"] == 2
    assert out["sd"] == pytest.approx(0.3535533906)
    assert mean_sd([None]) == {"mean": None, "sd": None, "n": 0}
    assert mean_sd([0.5]) == {"mean": 0.5, "sd": None, "n": 1}
