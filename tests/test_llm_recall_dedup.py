"""llm_recall_dedup.py（同じ式を 1 本と数える＝延べなしの Recall@20・MAP@20）の単体テスト.

規則：式の同一性は表記（latex_key）で決める。正解集合でも並べた答えでも、同じ式は 1 本と数える。
"""
import pytest

from llm_recall_dedup import dedup_ranks, dedup_metrics, parse_weight_label

IDENT = {"a1": "A", "a2": "A", "b1": "B", "c1": "C", "d1": "D"}   # a1 と a2 は同じ式（別の ID）


def test_copy_of_a_correct_equation_counts_as_found():
    ranks, k = dedup_ranks(["a2", "d1"], correct_keys=["a1"], identity=IDENT)
    assert (ranks, k) == ([1], 1)


def test_two_copies_in_the_correct_set_count_as_one_equation():
    ranks, k = dedup_ranks(["a1", "b1"], correct_keys=["a1", "a2", "b1"], identity=IDENT)
    assert k == 2
    assert ranks == [1, 2]


def test_second_copy_in_the_answer_is_collapsed_and_later_items_move_up():
    # a1 と a2 は同じ式なので答えの中では 1 本：b1 は 3 位でなく 2 位
    ranks, k = dedup_ranks(["a1", "a2", "b1"], correct_keys=["b1"], identity=IDENT)
    assert (ranks, k) == ([2], 1)


def test_unknown_ids_keep_their_slots_as_wrong_items():
    ranks, k = dedup_ranks([None, None, "c1"], correct_keys=["c1"], identity=IDENT)
    assert (ranks, k) == ([3], 1)


def test_missing_correct_equation_gets_the_miss_rank():
    ranks, k = dedup_ranks(["d1"], correct_keys=["a1", "b1"], identity=IDENT)
    assert k == 2
    assert len(ranks) == 2 and all(r > 20 for r in ranks)


def test_key_without_identity_is_its_own_equation():
    ranks, k = dedup_ranks(["x9"], correct_keys=["x9"], identity=IDENT)
    assert (ranks, k) == ([1], 1)


@pytest.mark.parametrize("label, weights", [("w40-60", (0.4, 0.6)), ("w30-70", (0.3, 0.7)), ("w05-95", (0.05, 0.95)),
                                            ("w00-100", (0.0, 1.0)), ("w100-0", (1.0, 0.0))])
def test_parse_weight_label_gives_text_and_variable_weights(label, weights):
    # baseline の重みの名前（kato1020_baselines.wlabel の逆）：文章類似度の重みと変数の重複率の重み
    assert parse_weight_label(label) == pytest.approx(weights)


def test_dedup_metrics_recall_and_ap_over_distinct_equations():
    m = dedup_metrics(["a2", "d1", "b1"], correct_keys=["a1", "a2", "b1", "c1"], identity=IDENT)
    # 正解は A, B, C の 3 本。A は 1 位、B は 3 位、C はなし → Recall 2/3、AP = (1/1 + 2/3)/3
    assert m["K"] == 3
    assert m["recall@20"] == pytest.approx(2 / 3)
    assert m["AP@20"] == pytest.approx((1 + 2 / 3) / 3)
