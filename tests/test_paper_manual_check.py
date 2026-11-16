"""paper/cace/make_manual_check_sheet.py（人手の確認のための標本とチェックシート）の単体テスト."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "cace"))
import make_manual_check_sheet as mc  # noqa: E402


# ---- 標本：種類の件数の割合に合わせて無作為に選ぶ ---------------------------------------------------

def test_allocate_follows_the_sizes_of_the_families_and_sums_to_n():
    assert mc.allocate({"single": 91, "cross": 640, "synth": 1000}, 50) == {"single": 3, "cross": 18, "synth": 29}
    assert mc.allocate({"a": 2, "b": 98}, 50) == {"a": 1, "b": 49}
    assert sum(mc.allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2


def test_allocate_refuses_a_sample_larger_than_the_population():
    with pytest.raises(ValueError, match="population"):
        mc.allocate({"a": 2, "b": 3}, 50)


POP = {"single": [f"s{i}" for i in range(91)], "cross": [f"c{i}" for i in range(640)],
       "synth": [f"d{i}" for i in range(1000)]}


def test_draw_sample_is_reproducible_and_keeps_the_allocation():
    a = mc.draw_sample(POP, 50, seed=1)
    assert a == mc.draw_sample(POP, 50, seed=1)
    assert len(a) == 50 and len(set(a)) == 50
    assert [sum(x.startswith(p) for x in a) for p in "scd"] == [3, 18, 29]
    assert mc.draw_sample(POP, 50, seed=2) != a


def test_draw_sample_does_not_depend_on_the_order_of_the_population():
    shuffled = {k: list(reversed(v)) for k, v in reversed(list(POP.items()))}
    assert mc.draw_sample(shuffled, 50, seed=1) == mc.draw_sample(POP, 50, seed=1)


# ---- 1 件分の材料 -------------------------------------------------------------------------------

EQS = [{"id": "s1__a", "variables": {"T": "temperature [K]", "k": "rate constant"}},
       {"id": "s2__b", "variables": {"T": "dimensionless temperature", "q": "heat flow"}},
       {"id": "s3__c", "variables": {"q": "heat flow"}}]


def test_shared_symbols_lists_every_definition_of_a_symbol_that_two_or_more_equations_hold():
    assert mc.shared_symbols(EQS) == {"T": [("s1__a", "temperature [K]"), ("s2__b", "dimensionless temperature")],
                                      "q": [("s2__b", "heat flow"), ("s3__c", "heat flow")]}
    assert mc.shared_symbols(EQS[:1]) == {}


def test_other_variables_are_those_that_are_neither_inputs_nor_outputs():
    assert mc.other_variables(EQS, inputs=["T"], outputs=["q"]) == ["k"]


# ---- チェックシート ------------------------------------------------------------------------------

def _record(case_id="ms_x<1>", two_equations=False):
    eqs = [{"id": "s__e", "latex": "y_{1} = x < 2", "source": "s", "domain": "reactor", "context": "ctx",
            "variables": {"x": "input <x>", "y_{1}": "output"}}]
    if two_equations:
        eqs.append({"id": "s__f", "latex": "z = x", "source": "s", "domain": "reactor", "context": "ctx",
                    "variables": {"x": "another input", "z": "other"}})
    return {"case_id": case_id, "kind": "cross", "description": "heat & mass", "inputs": ["x"], "outputs": ["y_{1}"],
            "equations": eqs, "shared": mc.shared_symbols(eqs), "flagged": ["x"], "others": mc.other_variables(eqs, ["x"], ["y_{1}"])}


def test_render_html_has_one_block_per_case_with_the_four_criteria():
    html = mc.render_html([_record("a"), _record("b", two_equations=True)], seed=7)
    assert html.count('<section class="case"') == 2
    for i in (0, 1):
        for c in ("C1", "C2", "C3", "C4"):
            assert f'name="case{i}_{c}"' in html
    # 共通の記号がないケースは、C3 が最初から「対象外」
    assert 'name="case0_C3" value="NA" checked' in html
    assert 'name="case1_C3" value="NA" checked' not in html
    assert 'manual_check_7' in html                       # 途中経過の保存先は種ごと
    # 言語モデルが意味の違いを判定した記号には、その印が付く（2 本以上の式に現れる記号だけに出す）
    assert html.count("言語モデルが「式によって別の量」と判定") == 1
    assert "入力でも出力でもない変数" in html           # 2 件目の z


def test_render_html_escapes_the_text_of_the_data():
    html = mc.render_html([_record()], seed=7)
    assert 'data-case="ms_x&lt;1&gt;"' in html
    assert "heat &amp; mass" in html
    assert 'data-tex="y_{1} = x &lt; 2"' in html
    assert "input &lt;x&gt;" in html
    assert "<x>" not in html


def test_results_csv_header_names_the_criteria():
    assert mc.RESULT_COLUMNS == ["case_id", "C1", "C2", "C3", "C4", "note"]
    assert [c[0] for c in mc.CRITERIA] == ["C1", "C2", "C3", "C4"]


# ---- 実データ：記録した標本が、作り直した標本と同じであること ------------------------------------------

def test_recorded_sample_matches_a_fresh_draw():
    rec = json.loads(mc.SAMPLE_JSON.read_text(encoding="utf-8"))
    fresh = mc.build_sample()
    assert rec == fresh
    assert rec["n"] == 50 and len(rec["cases"]) == 50
    assert rec["population"] == {"single": 91, "cross": 640, "synth": 1000}
    assert rec["allocation"] == {"single": 3, "cross": 18, "synth": 29}
