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
    ("ShareHaikuRefusals", "2.9"), ("OpusAsInstructed", "4{,}788"), ("ShareOpusAsInstructed", "95.4"),
    ("HaikuAsInstructed", "3{,}746"), ("ShareHaikuAsInstructed", "74.6"),
    ("OpusOverTwenty", "31"), ("OpusUnderTwenty", "181"), ("HaikuOverTwenty", "645"), ("HaikuUnderTwenty", "463"),
    ("OpusRepeatedId", "75"), ("OpusInvalidId", "21"), ("HaikuRepeatedId", "626"),
    ("RunSdOpus", "0.042"), ("RunSameOpus", "72.0"), ("JaccardHaiku", "0.324"), ("SameListOpus", "0.06"),
    ("TimeRunA", "161"), ("TimeRunAFifty", "66"), ("RecAFiveHundredCand", "0.981"),      # kato1021/sweep.json
    ("TimeRatioA", "2.4"),
    ("BaseValRecATextOnly", "0.362"), ("BaseValRecALowMin", "0.889"),                   # kato1021/baselines.json by_weight
    ("BaseValRecALowMax", "0.906"),
    ("Dropout", "0.1"), ("BmKOne", "1.5"), ("BmB", "0.75"), ("SvdDim", "256"),          # constants in the code
    ("NSearchPdf", "306"), ("NPerFieldMin", "7"), ("NPerFieldMax", "10"),               # appendix_materials/search_terms.csv
    ("RecKOneProp", "1.000"), ("RecKOneBase", "1.000"),                                 # kato1021/main.json buckets
    ("NBasePairs", "12"), ("BasePairFirst", "(1.0, 0.0)"), ("BasePairLast", "(0.0, 1.0)"),  # kato1021/baselines.json weights
    ("StagePairs", "$(0.7, 0.3)$, $(0.5, 0.5)$, $(0.3, 0.7)$, and $(0.0, 1.0)$"),      # kato1021/n1_rule.json weights
    ("Seeds", "42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999"),                # llm_db_baseline/splits.json
    ("NTrainFirstSplit", "975"), ("NTestFirstSplit", "407"), ("FirstSeed", "42"),
    ("ShareSharedEqA", "37.1"), ("ShareSharedEqAMin", "29.1"), ("ShareSharedEqAMax", "43.5"),   # split_overlap.json
    ("ShareSharedEqCross", "69.9"), ("ShareSharedEqSynth", "17.3"), ("ShareSharedEqSingle", "4.7"),
    ("ShareSharedEqB", "8.4"),
    ("NSingleOneEq", "67"), ("NSingleRefOnly", "61"), ("NWithIntermediate", "7"),     # kato1021/data_counts.json
    ("NOnlyPropEq", "16"), ("NRefSetsOther", "31"),
    ("LeadRecSingle", "0.036"), ("LeadMapTwSingle", "0.031"), ("LeadRecCross", "0.050"),       # id_summary.json by_kind
    ("LeadMapTwCross", "0.130"), ("LeadRecSynth", "0.063"), ("LeadMapTwSynth", "0.079"),
    ("TerOpusMin", "0.769"), ("TerOpusMax", "0.780"),
    ("NRuleAdded", "277"),                                                              # fix_dataset_quality_report.json
    ("DiffRecAPropBase", "+0.075"), ("DiffMapBPropBase", "+0.205"),                     # kato1021/main.json compare
    ("DiffRecKTen", "+0.238"), ("MapKOneBase", "0.934"), ("NRecordsA", "4{,}229"),      # kato1021/main.json buckets
    ("RecARSeven", "0.975"), ("DiffMapStepThree", "+0.073"), ("WinsRecStepOne", "3"),   # kato1021/mech.json
    ("DiffRecLatent", "-0.021"), ("LiftMapKMax", "+0.095"),
    ("PfiRecVar", "0.018"), ("PfiRecTopic", "0.974"), ("PfiRecSpec", "0.089"),          # kato1021/pfi.json
    ("MapAEightGreedy", "0.822"), ("DiffMapAEight", "+0.032"), ("NRecEightA", "698"),   # kato1021/greedy.json
    ("AucCandLatent", "0.909"), ("AucFullIoJac", "0.999"),                              # kato1021/topic_diag_final.json
    ("NMissed", "84"), ("MeanKUnsolved", "7.81"), ("RescueSpec", "0.52"),               # kato1021/case_outcomes_final.json
    ("NOpusWinRec", "36"), ("NOpusWinCopies", "28"), ("NOpusWinSevenPlus", "26"),       # llm_db_baseline/opus_vs_proposed.json
    ("NOnlyOpusEq", "58"), ("WrongInIoProp", "9.2"), ("TerLowWin", "19.5"),
    ("NOneAllGt", "1{,}880"), ("CovATwoHundred", "0.992"),                              # kato1021/n1_coverage_w30-70.json
    ("CostPerCallOpus", "0.21"), ("CostPerCallHaiku", "0.05"),                          # id_summary.json models.*.cost
])
def test_build_has_known_values(built, name, value):
    assert built[name] == value


@pytest.mark.parametrize("name, value", [
    # llm_db_baseline/id_summary.json の by_overlap：テスト用のケースを、訓練用のケースとの正解式の共有で分けた値
    ("NTestShareEqA", "1{,}568"), ("NTestShareEqCrossA", "1{,}143"), ("NTestShareEqSingleA", "9"),
    ("PctNoShareEqA", "62.9"), ("PctNoShareEqTvA", "57.6"), ("PctNoShareSrcA", "21.7"),
    ("RecANoShareProp", "0.977"), ("MapTwANoShareProp", "0.853"),
    ("RecANoShareTvProp", "0.978"), ("MapTwANoShareTvProp", "0.861"),
    ("MapTwCrossShareProp", "0.884"), ("MapTwCrossNoShareProp", "0.899"),
    ("MapTwSynthShareProp", "0.824"), ("MapTwSynthNoShareProp", "0.834"),
])
def test_build_has_the_values_by_overlap_with_the_training_cases(built, name, value):
    assert built[name] == value


def test_overlap_claims_stop_the_build_when_they_no_longer_hold(monkeypatch):
    """「共有しないテスト用のケースでも 10 通りすべてで上回る」が成り立たなくなったら、止まること."""
    import copy
    real = mn.load

    def fake(path):
        d = real(path)
        if Path(path).name != "id_summary.json":
            return d
        d = copy.deepcopy(d)
        d["settings"]["A"]["by_overlap"]["no_shared_equation"]["compare"]["proposed vs Opus 5.5"]["AP@20"]["wins"] = 9
        return d
    monkeypatch.setattr(mn, "load", fake)
    with pytest.raises(ValueError, match="share no ground-truth equation"):
        mn._results_overlap(mn.Registry())


@pytest.mark.parametrize("name, value", [
    # paper/cace/data/manual_check_results_claude.csv：言語モデル（Claude Fable 5.2）が標本の 50 件を 4 つの基準で判定した結果
    ("NCheckCases", "50"), ("NCheckSingle", "3"), ("NCheckCross", "18"), ("NCheckSynth", "29"),
    ("NCheckProgram", "13"), ("NCheckWritten", "5"),
    ("NCheckOneYes", "38"), ("NCheckTwoYes", "23"), ("NCheckThreeYes", "25"), ("NCheckThreeApplicable", "41"),
    ("NCheckFourYes", "28"), ("NCheckAllMet", "15"), ("NCheckAllMetSynth", "7"),
    ("NCheckProgramOneNo", "10"), ("NCheckSynthTwoNo", "14"), ("NCheckSynthThreeNo", "11"), ("NCheckSynthFourNo", "17"),
    ("NCheckFlagged", "15"), ("NCheckFlagAgree", "40"),       # 以前の Opus 5.5 の判定（experiments/var_descriptions.json）との一致
])
def test_build_has_the_judgments_of_the_language_model_on_the_sample(built, name, value):
    assert built[name] == value


def test_sample_check_claims_stop_the_build_when_they_no_longer_hold(monkeypatch):
    """「人が書いた出典横断のケースは 4 つとも満たした」が成り立たなくなったら、止まること."""
    import manual_check as chk
    real = chk.read_results

    def fake(path, sample_cases):
        res = real(path, sample_cases)
        res["ms_cstr_heat_001"] = {**res["ms_cstr_heat_001"], "C1": "N", "note": "C1: changed by the test"}
        return res
    monkeypatch.setattr(chk, "read_results", fake)
    with pytest.raises(ValueError, match="written combination"):
        mn._sample_check(mn.Registry())


# ---- 古い numbers.tex の検出 -----------------------------------------------------------------

def test_check_detects_stale_file(tmp_path):
    out = tmp_path / "numbers.tex"
    assert mn.main(["--out", str(out)]) == 0
    assert mn.main(["--check", "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-1].replace("}{", "}{9", 1)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert mn.main(["--check", "--out", str(out)]) == 1


def test_require_stops_when_a_claim_of_the_text_no_longer_holds():
    mn.require(True, "x")
    with pytest.raises(ValueError, match="all 10 splits"):
        mn.require(False, "the method wins in all 10 splits")


def test_sdiff_formats_a_raw_difference_with_its_sign():
    assert mn.sdiff(0.0749) == "+0.075"
    assert mn.sdiff(-0.0209) == "-0.021"
    assert mn.sdiff(0.0001) == "0.000"


# ---- 付録の表（行を生成する） ----------------------------------------------------------------

def test_tables_have_the_expected_rows():
    t = mn.tables()
    assert set(t) == {"tab_baseline_weights.tex", "tab_stage1_grid.tex", "tab_bucket.tex", "tab_n1.tex",
                      "tab_search.tex", "tab_overlap.tex", "tab_check.tex"}
    rows = {k: [ln for ln in v.splitlines() if ln.rstrip().endswith("\\\\")] for k, v in t.items()}
    assert len(rows["tab_baseline_weights.tex"]) == 15       # 12 weight pairs + TF-IDF, BM25, E5 chosen per split
    assert len(rows["tab_stage1_grid.tex"]) == 16            # 2 settings x 4 weight pairs x 2 choices of incorrect candidates
    assert len(rows["tab_bucket.tex"]) == 15                 # 10 sizes + 5 bins of input variables
    assert len(rows["tab_n1.tex"]) == 10                     # 2 settings x 5 values of N1
    assert any(ln.startswith("$(0.3, 0.7)$ & 0.906 & $0.907 \\pm 0.020$ & $0.693 \\pm 0.061$")
               for ln in rows["tab_baseline_weights.tex"])
    assert rows["tab_baseline_weights.tex"][-2].startswith("BM25, per split & --- & $0.905 \\pm 0.019$")
    assert any("$(0.3, 0.7)$ & hard negatives & 2{,}000 & 0.982 & 0.981 & 0.862" in ln for ln in rows["tab_stage1_grid.tex"])
    assert any(ln.startswith("21 or more input variables & 996 & 0.757 & 0.943 & 0.947 & 0.573 & 0.819 & 0.846")
               for ln in rows["tab_bucket.tex"])
    assert any(ln.startswith("Setting A & 50 & 0.9644 & $0.959 \\pm 0.013$") for ln in rows["tab_n1.tex"])
    assert len(rows["tab_search.tex"]) == 33                 # 32 fields + the total
    assert "Optics & optics lens equation diffraction wave Maxwell & 5 & --- & --- & 2 & 3 & 10 & 10 \\\\" in rows["tab_search.tex"]
    assert rows["tab_search.tex"][-1] == "Total & & 124 & 92 & 36 & 51 & 3 & 306 & 290 \\\\"
    assert len(rows["tab_overlap.tex"]) == 24                # 4 groups of test cases x (heading + count + 4 methods)
    ov = rows["tab_overlap.tex"]
    assert ov[0] == "\\multicolumn{5}{@{}l}{All test cases} \\\\"
    assert ov[1] == "Test cases in the 10 splits & \\multicolumn{2}{c}{4{,}229} & \\multicolumn{2}{c}{2{,}432} \\\\"
    assert ov[4] == "Opus 5.5 & $0.924 \\pm 0.018$ & $0.761 \\pm 0.063$ & $0.899 \\pm 0.013$ & $0.744 \\pm 0.044$ \\\\"
    assert ov[6] == "\\multicolumn{5}{@{}l}{Share a ground-truth equation with a training case} \\\\"
    assert ov[12] == "\\multicolumn{5}{@{}l}{Share no ground-truth equation with a training case} \\\\"
    assert ov[13] == "Test cases in the 10 splits & \\multicolumn{2}{c}{2{,}661} & \\multicolumn{2}{c}{2{,}227} \\\\"
    assert ov[15] == "Proposed & $0.977 \\pm 0.011$ & $0.853 \\pm 0.054$ & $0.969 \\pm 0.013$ & $0.827 \\pm 0.042$ \\\\"
    assert ov[18] == "\\multicolumn{5}{@{}l}{Share no source of a ground-truth equation with a training case} \\\\"
    assert ov[19] == "Test cases in the 10 splits & \\multicolumn{2}{c}{916} & \\multicolumn{2}{c}{1{,}136} \\\\"
    # the judgments of the language model on the sample: met / cases to which the criterion applies
    assert rows["tab_check.tex"] == [
        "1. The equations describe the requested model & 3/3 & 3/13 & 5/5 & 27/29 & 38/50 \\\\",
        "2. The outputs are the requested quantities & 3/3 & 0/13 & 5/5 & 15/29 & 23/50 \\\\",
        "3. Shared symbols keep one meaning & 1/1 & 8/13 & 3/3 & 13/24 & 25/41 \\\\",
        "4. The equations determine the outputs & 3/3 & 8/13 & 5/5 & 12/29 & 28/50 \\\\",
        "All four & 3/3 & 0/13 & 5/5 & 7/29 & 15/50 \\\\"]


def test_code_constant_names_the_file_when_the_pattern_is_missing(tmp_path):
    f = tmp_path / "model.py"
    f.write_text("nn.Dropout(0.1)\n", encoding="utf-8")
    assert mn.code_constant(f, r"nn\.Dropout\(([0-9.]+)\)") == "0.1"
    with pytest.raises(ValueError, match="model.py"):
        mn.code_constant(f, r"k1: float = ([0-9.]+)")


def test_code_constant_rejects_two_different_values(tmp_path):
    f = tmp_path / "model.py"
    f.write_text("nn.Dropout(0.1)\nnn.Dropout(0.2)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model.py"):
        mn.code_constant(f, r"nn\.Dropout\(([0-9.]+)\)")


def test_check_covers_the_generated_tables(tmp_path):
    out = tmp_path / "numbers.tex"
    assert mn.main(["--out", str(out)]) == 0
    f = tmp_path / "generated" / "tab_bucket.tex"
    f.write_text(f.read_text(encoding="utf-8") + "x", encoding="utf-8")
    assert mn.main(["--check", "--out", str(out)]) == 1


# ---- 査読の指摘（2026-11-16）：欠けたキー、欠損値、追跡されていない入力 --------------------------------

def test_load_names_the_file_and_the_path_of_a_missing_key(tmp_path):
    """get() を通さない素の添字でも、欠けたキーはファイル名とキーの道筋つきで止まること."""
    import json
    f = tmp_path / "result.json"
    f.write_text(json.dumps({"table": {"A": {"MAP": {"mean": 0.5}}}, "rows": [{"x": 1}]}), encoding="utf-8")
    d = mn.load(f)
    assert d["table"]["A"]["MAP"]["mean"] == 0.5
    assert [k for k, _ in d["table"].items()] == ["A"] and d["table"]["A"].get("absent", 0) == 0
    with pytest.raises(KeyError, match=r"result\.json.*table\.A\.MAP\.mean_delta"):
        d["table"]["A"]["MAP"]["mean_delta"]
    with pytest.raises(KeyError, match=r"result\.json.*rows\[0\]\.y"):
        d["rows"][0]["y"]
    with pytest.raises(IndexError, match=r"result\.json.*rows\[3\]"):
        d["rows"][3]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.5", True])
def test_number_formats_reject_missing_and_non_finite_values(bad):
    for f in (mn.fmt, mn.pct, mn.money, mn.count):
        with pytest.raises(ValueError):
            f(bad)


@pytest.mark.parametrize("bad", ["None", "nan", "", "-nan", "inf", "0.5 and None"])
def test_registry_rejects_placeholders_for_missing_values(bad):
    with pytest.raises(ValueError, match="RecAProp"):
        mn.Registry().add("RecAProp", bad)


def test_registry_accepts_text_values_that_only_resemble_placeholders():
    r = mn.Registry()
    r.add("StagePairs", "$(0.7, 0.3)$, and $(0.0, 1.0)$")
    r.add("Seeds", "42, 123")


def test_database_counts_come_from_a_tracked_file():
    """unified_equations.json は git で追跡していないので、数は追跡してある db_counts.json から読むこと."""
    assert mn.DB_COUNTS.exists()
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(mn.DB_COUNTS)], cwd=mn.ROOT,
                             capture_output=True, text=True)
    assert tracked.returncode == 0, "paper/cace/data/db_counts.json must be tracked by git"


def test_build_does_not_need_the_untracked_database(monkeypatch, tmp_path):
    monkeypatch.setattr(mn, "DB_FILE", tmp_path / "absent.json")
    assert mn.build().values["NEq"] == "11{,}146"


def test_build_stops_when_the_database_changed_after_the_counts(monkeypatch, tmp_path):
    changed = tmp_path / "unified_equations.json"
    changed.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(mn, "DB_FILE", changed)
    with pytest.raises(ValueError, match="db_counts"):
        mn.build()
