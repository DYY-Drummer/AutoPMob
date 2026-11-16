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


# ---- 確認の結果の読み込み、集計、2 人の判定の一致（paper/cace/manual_check.py） ----------------------------

import manual_check as chk  # noqa: E402

CSV_HEAD = "case_id,C1,C2,C3,C4,note\n"


def _write(tmp_path, body, name="r.csv"):
    f = tmp_path / name
    f.write_text(CSV_HEAD + body, encoding="utf-8")
    return f


def test_read_results_returns_the_judgments_by_case(tmp_path):
    f = _write(tmp_path, 'a,Y,N,NA,Y,"理由, その 1"\nb,N,Y,Y,N,\n')
    out = chk.read_results(f, ["a", "b"])
    assert out["a"] == {"C1": "Y", "C2": "N", "C3": "NA", "C4": "Y", "note": "理由, その 1"}
    assert list(out) == ["a", "b"]


@pytest.mark.parametrize("body, message", [
    ("a,Y,N,NA,Y,\n", "missing"),                       # b がない
    ("a,Y,N,NA,Y,\nb,N,Y,Y,N,\nc,Y,Y,Y,Y,\n", "not in the sample"),
    ("a,Y,,NA,Y,\nb,N,Y,Y,N,\n", "C2"),                 # 空欄
    ("a,Y,N,NA,NA,\nb,N,Y,Y,N,\n", "C4"),               # 対象外は C3 だけ
    ("a,Y,N,NA,Y,\na,N,Y,Y,N,\n", "twice"),
])
def test_read_results_rejects_incomplete_or_invalid_files(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        chk.read_results(_write(tmp_path, body), ["a", "b"])


def test_read_results_rejects_other_columns(tmp_path):
    f = tmp_path / "r.csv"
    f.write_text("case_id,C1,C2,C3,note\na,Y,N,NA,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        chk.read_results(f, ["a"])


RES = {"a": {"C1": "Y", "C2": "Y", "C3": "NA", "C4": "Y", "note": ""},
       "b": {"C1": "N", "C2": "Y", "C3": "Y", "C4": "N", "note": ""},
       "c": {"C1": "Y", "C2": "N", "C3": "N", "C4": "Y", "note": ""}}
KIND_OF = {"a": "single", "b": "cross", "c": "cross"}


def test_summarize_counts_each_criterion_overall_and_by_family():
    s = chk.summarize(RES, KIND_OF)
    assert s["n"] == 3
    assert s["criteria"]["C1"] == {"Y": 2, "N": 1, "NA": 0}
    assert s["criteria"]["C3"] == {"Y": 1, "N": 1, "NA": 1}
    assert s["by_kind"]["cross"]["C2"] == {"Y": 1, "N": 1, "NA": 0}
    assert s["all_met"] == {"n": 1, "by_kind": {"single": 1, "cross": 0, "synth": 0}, "cases": ["a"]}
    assert s["n_by_kind"] == {"single": 1, "cross": 2, "synth": 0}


def test_cohen_kappa():
    assert chk.cohen_kappa([("Y", "Y"), ("N", "N"), ("Y", "Y"), ("N", "N")]) == pytest.approx(1.0)
    assert chk.cohen_kappa([("Y", "Y"), ("Y", "N"), ("N", "Y"), ("N", "N")]) == pytest.approx(0.0)
    # 観測の一致 0.75、偶然の一致 0.5*0.25 + 0.5*0.75 = 0.5 → (0.75 - 0.5) / 0.5
    assert chk.cohen_kappa([("Y", "Y"), ("Y", "N"), ("N", "N"), ("N", "N")]) == pytest.approx(0.5)
    assert chk.cohen_kappa([("Y", "Y"), ("Y", "Y")]) is None      # どちらも 1 種類の答えだけ：定義できない


def test_agreement_counts_matches_and_lists_the_cases_that_differ():
    other = {"a": dict(RES["a"]), "b": dict(RES["b"], C4="Y", note="求まる"), "c": dict(RES["c"], C3="Y")}
    out = chk.agreement(RES, other)
    assert out["criteria"]["C1"]["n"] == 3 and out["criteria"]["C1"]["agree"] == 3
    assert out["criteria"]["C4"]["agree"] == 2
    assert out["criteria"]["C3"]["agree"] == 2
    assert out["cases_all_four_agree"] == 1
    assert out["judgments"] == {"n": 12, "agree": 10}
    assert out["differences"] == [{"case_id": "b", "criterion": "C4", "a": "N", "b": "Y", "note_a": "", "note_b": "求まる"},
                                  {"case_id": "c", "criterion": "C3", "a": "N", "b": "Y", "note_a": "", "note_b": ""}]


def test_agreement_needs_the_same_cases():
    with pytest.raises(ValueError, match="same cases"):
        chk.agreement(RES, {"a": RES["a"]})


def test_recorded_judgments_of_the_language_model_are_complete():
    """Claude の判定（著者の結果と照合する相手）が、標本の 50 件をすべて含み、値が正しい形であること."""
    sample = json.loads(mc.SAMPLE_JSON.read_text(encoding="utf-8"))
    res = chk.read_results(chk.RESULTS_CLAUDE, sample["cases"])
    s = chk.summarize(res, {c: ("synth" if c.startswith("dae_") else "single" if c.startswith("core_") else "cross") for c in res})
    assert s["criteria"] == {"C1": {"Y": 38, "N": 12, "NA": 0}, "C2": {"Y": 23, "N": 27, "NA": 0},
                             "C3": {"Y": 25, "N": 16, "NA": 9}, "C4": {"Y": 28, "N": 22, "NA": 0}}
    assert s["all_met"]["n"] == 15 and s["all_met"]["by_kind"] == {"single": 3, "cross": 5, "synth": 7}
