"""kato1020_n1.py のテスト：N1 の決め方（訓練と検証のケースで全正解式が入る値を、千の位に切り上げる）."""
from kato1020_n1 import round_up_thousand


def test_round_up_to_thousand():
    assert round_up_thousand(1880) == 2000
    assert round_up_thousand(2000) == 2000
    assert round_up_thousand(1) == 1000
    assert round_up_thousand(5537) == 6000


def test_round_up_is_capped_at_db_size():
    assert round_up_thousand(11_100, n_eq=11_146) == 11_146   # 12,000 は DB の式数を超える
    assert round_up_thousand(10_500, n_eq=11_146) == 11_000
