"""llm_id_summary.py（LLM との比較を式の ID で判定した値の集計）の単体テスト."""
import pytest

from llm_id_summary import runs_agreement, split_means


PER_SEED = {
    "42": {"a": {"recall@20": 1.0, "AP@20": 0.5}, "b": {"recall@20": 0.5, "AP@20": 0.25}},
    "123": {"a": {"recall@20": 0.0, "AP@20": 0.0}, "c": {"recall@20": 1.0, "AP@20": 1.0}},
}
KIND = {"a": "single", "b": "synth", "c": "synth"}


def test_split_means_averages_the_test_cases_of_each_split():
    assert split_means(PER_SEED, "recall@20") == {"42": pytest.approx(0.75), "123": pytest.approx(0.5)}


def test_split_means_keeps_only_cases_of_one_kind():
    assert split_means(PER_SEED, "AP@20", kind_of=KIND, kind="synth") == {"42": 0.25, "123": 1.0}


def test_split_means_skips_splits_without_cases_of_that_kind():
    assert split_means(PER_SEED, "recall@20", kind_of=KIND, kind="single") == {"42": 1.0, "123": 0.0}
    assert split_means({"42": {"b": {"recall@20": 0.5}}}, "recall@20", kind_of=KIND, kind="single") == {}


def test_runs_agreement_reports_mean_sd_and_share_of_identical_runs():
    runs = {"x": [1.0, 1.0, 1.0], "y": [0.0, 0.5, 1.0]}
    out = runs_agreement(runs)
    assert out["recall@20_sd_mean"] == pytest.approx((0.0 + 0.5) / 2)
    assert out["cases_all_runs_same"] == pytest.approx(0.5)
    assert out["n_cases"] == 2
