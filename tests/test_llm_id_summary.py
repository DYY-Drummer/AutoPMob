"""llm_id_summary.py（LLM との比較を式の ID で判定した値の集計）の単体テスト."""
import pytest

from llm_id_summary import (answer_format, by_overlap, cost_breakdown, overlap_groups, runs_agreement, split_means,
                            subset_means, time_range)


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


# ---- 訓練用のケースと正解式を共有するテスト用のケース、共有しないテスト用のケースに分けた集計 ---------------

OV_GT = {"t1": ["s1__a", "s2__b"], "t2": ["s5__k"], "v": ["s6__m"],
         "x": ["s3__c"],              # 共有なし（出典も共有しない）
         "y": ["s2__b", "s4__d"],     # 訓練用のケース t1 と式 s2__b を共有
         "z": ["s2__e"],              # 式は共有しないが、出典 s2 は訓練用の正解式の出典でもある
         "u": ["s6__m"]}              # 検証用のケース v とだけ式を共有


def test_overlap_groups_split_the_test_cases_of_one_split():
    g = overlap_groups({"train": ["t1", "t2"], "val": ["v"], "test": ["x", "y", "z", "u"]}, OV_GT)
    assert g["shared_equation"] == {"y"}
    assert g["no_shared_equation"] == {"x", "z", "u"}
    assert g["no_shared_equation_train_val"] == {"x", "z"}
    assert g["no_shared_source"] == {"x", "u"}
    assert g["shared_equation"] | g["no_shared_equation"] == {"x", "y", "z", "u"}


def test_subset_means_averages_only_the_given_cases_and_skips_empty_splits():
    assert subset_means(PER_SEED, "recall@20", {"42": {"b"}, "123": {"a", "c"}}) == {"42": 0.5, "123": pytest.approx(0.5)}
    assert subset_means(PER_SEED, "AP@20", {"42": set(), "123": {"c"}}) == {"123": 1.0}


def test_by_overlap_scores_every_method_on_each_group_of_test_cases():
    def m(r, a):
        return {"recall@20": r, "AP@20": a}
    splits = {"1": {"train": ["t1", "t2"], "val": ["v"], "test": ["x", "y"]},
              "2": {"train": ["t1", "t2"], "val": ["v"], "test": ["x", "y", "z"]}}
    kind = {"x": "synth", "y": "cross", "z": "synth"}
    psc = {"proposed": {"1": {"x": m(1.0, 0.8), "y": m(1.0, 1.0)}, "2": {"x": m(0.5, 0.5), "y": m(1.0, 0.5), "z": m(1.0, 1.0)}},
           "baseline": {"1": {"x": m(0.5, 0.4), "y": m(1.0, 1.0)}, "2": {"x": m(0.5, 0.5), "y": m(0.0, 0.0), "z": m(0.5, 0.5)}}}
    out = by_overlap(psc, splits, OV_GT, kind)
    assert set(out) == {"shared_equation", "no_shared_equation", "no_shared_equation_train_val", "no_shared_source"}
    no = out["no_shared_equation"]
    assert no["n_test"] == {"1": 1, "2": 2}
    assert no["n_test_by_kind"] == {"single": 0, "cross": 0, "synth": 3}
    assert no["methods"]["proposed"]["recall@20"]["per_seed"] == {"1": 1.0, "2": 0.75}
    assert no["methods"]["proposed"]["recall@20"]["mean"] == pytest.approx(0.875)
    assert no["methods"]["baseline"]["AP@20"]["per_seed"] == {"1": 0.4, "2": 0.5}
    # 提案手法が上回った分割の数：種 1 は 1.0 > 0.5、種 2 は 0.75 > 0.5
    assert no["compare"]["proposed vs baseline"]["recall@20"]["wins"] == 2
    assert no["compare"]["proposed vs baseline"]["recall@20"]["n"] == 2
    sh = out["shared_equation"]
    assert sh["n_test"] == {"1": 1, "2": 1} and sh["n_test_by_kind"]["cross"] == 2
    assert sh["compare"]["proposed vs baseline"]["recall@20"] == {"n": 2, "mean_diff": 0.5, "wins": 1, "ties": 1, "losses": 0}
    # 種類ごとの値は、その種類のケースがある分割が 2 つ以上あるときだけ出す
    assert sh["methods"]["proposed"]["by_kind"]["cross"]["AP@20"]["per_seed"] == {"1": 1.0, "2": 0.5}
    assert "synth" not in sh["methods"]["proposed"]["by_kind"]
    assert out["no_shared_source"]["n_test"] == {"1": 1, "2": 1}
