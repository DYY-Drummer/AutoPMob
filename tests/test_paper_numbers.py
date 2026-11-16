"""paper/cace/make_numbers.py（論文の数値を結果の JSON から TeX のマクロにする）の単体テスト."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "cace"))
import make_numbers as mn  # noqa: E402


# ---- 数値の書式 ----------------------------------------------------------------------------

def test_fmt_fixed_decimals():
    assert mn.fmt(0.98099) == "0.981"
    assert mn.fmt(1.0) == "1.000"
    assert mn.fmt(5.4979, 2) == "5.50"


def test_fmt_no_negative_zero():
    assert mn.fmt(-0.0004) == "0.000"


def test_diff_uses_raw_values():
    # 0.981 - 0.906 と丸めた値どうしで引いても 0.075 だが、丸める前の差を丸めることを確かめる（0.0749 → 0.075）
    assert mn.diff(0.98099, 0.90612) == "+0.075"
    assert mn.diff(0.9044, 0.9066) == "-0.002"


def test_diff_unsigned_gives_the_magnitude():
    assert mn.diff(0.98099, 0.92422, signed=False) == "0.057"
    assert mn.diff(0.92422, 0.98099, signed=False) == "0.057"


def test_diff_that_rounds_to_zero_has_no_sign():
    assert mn.diff(0.5001, 0.5003) == "0.000"


def test_pct():
    assert mn.pct(0.9361) == "93.6"
    assert mn.pct(0.0287, 1) == "2.9"


def test_count_thousands():
    assert mn.count(11146) == "11{,}146"
    assert mn.count(91) == "91"
    assert mn.count(1000) == "1{,}000"


def test_money():
    assert mn.money(1058.93) == "1{,}058.93"
    assert mn.money(255.2301) == "255.23"


# ---- マクロの登録 ---------------------------------------------------------------------------

def test_registry_renders_sorted_newcommands():
    r = mn.Registry()
    r.add("RecB", "0.2")
    r.add("RecA", "0.1")
    assert r.render().splitlines()[-2:] == [r"\newcommand{\RecA}{0.1}", r"\newcommand{\RecB}{0.2}"]


def test_registry_rejects_duplicate():
    r = mn.Registry()
    r.add("RecAProp", "1")
    with pytest.raises(ValueError, match="RecAProp"):
        r.add("RecAProp", "2")


@pytest.mark.parametrize("name", ["Rec20A", "Rec_A", "Rec A", "", "rec@A"])
def test_registry_rejects_names_that_are_not_letters_only(name):
    # TeX は \Rec20A を \Rec のあとに 20A が続くものとして読むので、数字や記号を含む名前は受け付けない
    with pytest.raises(ValueError):
        mn.Registry().add(name, "1")


def test_get_names_file_and_path_when_a_key_is_missing():
    with pytest.raises(KeyError) as e:
        mn.get({"a": {"x": 1}}, "a.b", "x.json")
    assert "x.json" in str(e.value) and "a.b" in str(e.value)
    assert mn.get({"a": {"b": 3}}, "a.b", "x.json") == 3


# ---- 実データからの組み立て -----------------------------------------------------------------

@pytest.fixture(scope="module")
def built():
    return mn.build().values


@pytest.mark.parametrize("name, value", [
    ("NEq", "11{,}146"), ("NSources", "361"), ("NCases", "1{,}731"),        # データ
    ("RecABase", "0.906"), ("RecAProp", "0.981"), ("MapAProp", "0.862"),    # kato1021/main.json
    ("RecAOpus", "0.924"), ("MapTwAOpus", "0.761"), ("RecAHaiku", "0.497"),  # llm_db_baseline/id_summary.json
    ("NSymbols", "13{,}220"), ("NSymbolsConflict", "4{,}208"), ("NLabels", "1{,}998"),  # unified_equations.json
    ("NSynthOneSource", "959"), ("NOpenAlex", "290"), ("NEarly", "39"),                 # cases, papers.csv
    ("NTestAMin", "298"), ("NTestAMax", "527"), ("NValAMin", "216"),                    # llm_db_baseline/splits.json
    ("NOne", "2{,}000"), ("WText", "0.3"), ("WVar", "0.7"), ("RefSize", "5"),           # kato1021/selected.json, n1_rule.json
    ("ValRecAMin", "0.980"), ("ValRecAMax", "0.983"), ("LearnRate", "10^{-3}"),
    ("NSymbolCases", "1{,}299"), ("NMeaningCases", "596"),                              # kato1021/data_counts.json
    ("NDbTokens", "952{,}144"), ("NTaskTokens", "303"),                                 # llm_db_baseline/token_counts_v4.json
    ("NFields", "32"), ("MeanSrcSynth", "1.10"),                                        # paper_collection metadata, data_counts.json
    ("NTestDistinct", "1{,}673"), ("NCalls", "5{,}019"),                                # llm_db_baseline/id_summary.json, results
    ("CostOpus", "1{,}058.93"), ("CostHaiku", "255.23"), ("CostTotal", "1{,}314.16"),   # id_summary.json models.*.cost
    ("ReadShareOpus", "90.2"), ("ReadShareHaiku", "93.6"), ("HaikuRefusals", "144"),
    ("ShareHaikuRefusals", "2.9"), ("OpusWithTwenty", "4{,}838"), ("ShareOpusWithTwenty", "96.4"),
    ("RunSdOpus", "0.042"), ("RunSameOpus", "72.0"), ("JaccardHaiku", "0.324"), ("SameListOpus", "0.06"),
    ("TrainMinAMax", "8.2"), ("TrainMinAMin", "2.4"), ("EvalSecAMax", "9"),              # id_summary.json proposed_time_sec
])
def test_build_has_known_values(built, name, value):
    assert built[name] == value


# ---- 古い numbers.tex の検出 -----------------------------------------------------------------

def test_check_detects_stale_file(tmp_path):
    out = tmp_path / "numbers.tex"
    assert mn.main(["--out", str(out)]) == 0
    assert mn.main(["--check", "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace("}{", "}{9", 1)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert mn.main(["--check", "--out", str(out)]) == 1
