"""llm_id_summary.py（LLM との比較を式の ID で判定した値の集計）の単体テスト."""
import pytest

from llm_id_summary import answer_format, cost_breakdown, runs_agreement, split_means, time_range


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


def test_cost_breakdown_separates_the_cached_database_reads():
    # Opus 5.5: input 4, cache read 0.20, output 20 dollars per million tokens (run_llm_db_baseline.PRICES)
    usages = [{"input_tokens": 1000, "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0,
               "output_tokens": 1000}] * 2
    out = cost_breakdown(usages, "claude-opus-5-5")
    assert out["total"] == pytest.approx(2 * 0.224)
    assert out["read"] == pytest.approx(2 * 0.2)
    assert out["read_share"] == pytest.approx(0.2 / 0.224)
    assert out["n_calls"] == 2


def test_time_range_gives_min_and_max_and_ignores_missing_values():
    assert time_range([67.05, None, 93.0, 80.0]) == {"min": 67.05, "max": 93.0, "n": 3}


def test_answer_format_counts_each_way_an_answer_departs_from_the_instruction():
    """指示は「20 個の ID を、どれも 1 回ずつ、データベースにあるものだけ」。数え方が重ならないこと（2026-11-16 の査読の指摘：
    旧 answers_with_20 は「重複を除いて 20 個以上」で、20 個を超えた答えも含んでいた）."""
    def row(n_raw, n_dup=0, n_invalid=0, n_truncated=0, stop="end_turn"):
        return {"n_raw": n_raw, "n_duplicates": n_dup, "n_invalid": n_invalid, "n_truncated": n_truncated,
                "ranked_ids": [1] * min(n_raw - n_dup, 20), "stop_reason": stop}
    rows = [row(20), row(20),                     # 指示どおり
            row(20, n_invalid=1),                 # 20 個だが 1 個はデータベースにない
            row(21, n_dup=1),                     # 重複を除くと 20 個
            row(30, n_truncated=10),              # 20 個を超える
            row(22, n_dup=5),                     # 重複を除くと 17 個
            row(12),                              # 20 個に足りない
            row(0, stop="refusal")]               # 止められて空
    assert answer_format(rows) == {
        "n": 8, "as_instructed": 2, "exactly_20_distinct": 4, "more_than_20_distinct": 1,
        "fewer_than_20_distinct": 3, "empty": 1, "with_repeats": 2, "with_absent_id": 1, "refusals": 1}


def test_answer_format_partitions_the_answers():
    rows = [{"n_raw": n, "n_duplicates": 0, "n_invalid": 0, "n_truncated": max(0, n - 20),
             "ranked_ids": [1] * min(n, 20), "stop_reason": "end_turn"} for n in (0, 5, 19, 20, 20, 21, 40)]
    out = answer_format(rows)
    assert out["exactly_20_distinct"] + out["more_than_20_distinct"] + out["fewer_than_20_distinct"] == out["n"] == 7
