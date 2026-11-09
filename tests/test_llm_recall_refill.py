"""llm_recall_dedup.refill_ranks（同じ式を 1 本にまとめ、順位 21 以降から 20 本になるまで補充した上位 20 件の採点）の単体テスト."""
import pytest

from llm_recall_dedup import refill_ranks, refill_top

IDENT = {"a1": "A", "a2": "A", "b1": "B", "c1": "C", "d1": "D", "e1": "E"}


def test_refill_top_collapses_copies_and_fills_from_below():
    # a1 と a2 は同じ式。上位 3 件に収めると、a2 を詰めて d1 を繰り上げる
    assert refill_top(["a1", "a2", "b1", "d1", "e1"], IDENT, n=3) == ["a1", "b1", "d1"]


def test_refill_top_keeps_unknown_ids_as_slots():
    assert refill_top([None, "a1", "a2", "b1"], IDENT, n=3) == [None, "a1", "b1"]


def test_refill_top_returns_shorter_list_when_nothing_is_left_to_fill():
    # LLM の答えが 20 本きりで、同じ式が 2 本あれば、詰めた後は 19 本のまま
    assert refill_top(["a1", "a2", "b1"], IDENT, n=3) == ["a1", "b1"]


def test_refill_ranks_scores_the_refilled_list():
    # 正解 B と D。a2 は a1 の写しなので詰められ、d1 が 3 位に繰り上がって上位 3 件に入る
    ranks, k = refill_ranks(["a1", "a2", "b1", "d1"], correct_keys=["b1", "d1"], identity=IDENT, n=3)
    assert (ranks, k) == ([2, 3], 2)


def test_refill_ranks_counts_copies_in_the_correct_set_once():
    ranks, k = refill_ranks(["a2", "c1"], correct_keys=["a1", "a2"], identity=IDENT, n=20)
    assert (ranks, k) == ([1], 1)


def test_refill_ranks_marks_equations_outside_the_top_n_as_missed():
    ranks, k = refill_ranks(["c1", "d1", "e1", "b1"], correct_keys=["b1"], identity=IDENT, n=3)
    assert k == 1 and ranks[0] > 20
