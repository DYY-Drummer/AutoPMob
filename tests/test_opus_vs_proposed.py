"""analyze_opus_vs_proposed.copies_summary：Opus が上回ったケースで、正解式と同じ表記の別の ID の式が上位 20 件に入っていた件数."""
import pytest

from analyze_opus_vs_proposed import copies_summary


def row(prop, opus):
    return {"wrong_prop": {"twin": prop}, "wrong_opus": {"twin": opus}}


def test_copies_summary_counts_cases_where_both_methods_listed_copies():
    out = copies_summary([row(2.0, 1.0), row(0.0, 3.0), row(0.5, 0.0), row(4.0, 4.0)])
    assert out["n_cases"] == 4
    assert out["n_with_copies_in_both"] == 2
    assert out["copies_per_list_prop"] == pytest.approx((2.0 + 0.0 + 0.5 + 4.0) / 4)
    assert out["copies_per_list_opus"] == pytest.approx((1.0 + 3.0 + 0.0 + 4.0) / 4)


def test_copies_summary_of_no_cases():
    assert copies_summary([]) == {"n_cases": 0, "n_with_copies_in_both": 0,
                                  "copies_per_list_prop": None, "copies_per_list_opus": None}
