"""kato1020_exclusions.py のテスト：未知の変数のそれぞれに、それを決める式を 1 本ずつ割り当てられるか.

判定は analyze_kato_0831_followup.gt_checks と同じ：正解式と未知の変数（正解式の変数のうち入力変数でないもの）の
二部グラフで完全マッチングがあり、式の数 = 未知の変数の数のとき「決まる」。
"""
from kato1020_exclusions import all_vars_are_inputs, solvable


def test_chain_is_solvable():
    # y = f(x), z = g(y)：x を与えると y、z が 1 本ずつ決まる
    assert solvable([{"x", "y"}, {"y", "z"}], {"x"})


def test_equation_with_only_inputs_is_not_solvable():
    # 2 本目の式の変数がすべて入力変数 → その式が決める未知の変数がない
    assert not solvable([{"x", "y"}, {"x"}], {"x"})
    assert all_vars_are_inputs([{"x", "y"}, {"x"}], {"x"})


def test_two_equations_for_one_unknown_is_not_solvable():
    # 未知の変数 1 個（y）を 2 本の式が決めようとし、z を決める式がない
    assert not solvable([{"x", "y"}, {"y", "w"}, {"z", "y"}], {"x", "w"})
    assert not all_vars_are_inputs([{"x", "y"}, {"y", "w"}, {"z", "y"}], {"x", "w"})
