"""kato1020_select.py のテスト：検証用のケースの Recall@20 が最も高い設定を選ぶ（同じなら MAP、それでも同じなら並びの先）."""
from kato1020_select import choose


def test_choose_highest_val_recall20():
    cand = [("w70-30", "random", {"Recall@20": 0.95, "MAP": 0.90}),
            ("w30-70", "hard", {"Recall@20": 0.97, "MAP": 0.80})]
    assert choose(cand)[:2] == ("w30-70", "hard")


def test_tie_on_recall20_broken_by_map_then_order():
    cand = [("w70-30", "random", {"Recall@20": 0.97, "MAP": 0.80}),
            ("w30-70", "random", {"Recall@20": 0.97, "MAP": 0.85}),
            ("w00-100", "hard", {"Recall@20": 0.97, "MAP": 0.85})]
    assert choose(cand)[:2] == ("w30-70", "random")
