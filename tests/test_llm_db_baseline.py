"""run_llm_db_baseline.py（LLM に数式データベースを渡して式の ID を選ばせる比較）の単体テスト.

加藤先生の指示（2026-10-14）による判定の規則:
- ID だけで判定し、等価性の判定はしない。
- DB にない ID や読めない出力は、その順位を占める不正解として数える。
- 同じ ID が重なったら、2 回目以降は除く。
- 指標は Recall@20 と MAP（標準の定義。上位 20 件に入らなかった正解式は P(e) = 0）。
"""
import copy
import json
import math
from pathlib import Path

import pytest

from run_llm_db_baseline import (
    MISS_RANK, KeyIndex, ap_full, build_params, compress_latex, cost_usd, db_text, examples_text,
    judge_ids, metrics_from_ranks, compare_splits, parse_output, pending_cases, price_for, ranks_in_list, ref_cases,
    shuffled_keys, split_cost, split_summary, summarize, task_text, split_test_records, v3_case_ids, v3_plan,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "experiments" / "llm_db_baseline" / "splits.json"


# ---- 応答の読み取りと ID の判定 ---------------------------------------------------

def test_parse_output_reads_equation_ids():
    assert parse_output('{"equation_ids": [3, 1, 2]}') == ([3, 1, 2], None)


@pytest.mark.parametrize("text", ["", "not json", '{"ids": [1]}', '{"equation_ids": "1,2"}', "[1, 2]"])
def test_parse_output_marks_unreadable(text):
    ids, err = parse_output(text)
    assert ids is None
    assert err


def test_judge_ids_dedups_and_keeps_invalid_ids_as_wrong_ranks():
    raw = [5, "7", 5, 99999, True, 3.0, "abc", 7, 2]
    got = judge_ids(raw, n_db=10)
    # 5 と "7" は有効。2 回目の 5 と 7 は除く。99999（範囲外）・True・3.0・"abc" は
    # DB にない ID として順位を占める（None）。最後の 2 は 7 番目の順位。
    assert got["ranked_ids"] == [5, 7, None, None, None, None, 2]
    assert got["n_raw"] == 9
    assert got["n_duplicates"] == 2
    assert got["n_invalid"] == 4
    assert got["n_truncated"] == 0


def test_judge_ids_keeps_only_top_20():
    got = judge_ids(list(range(1, 26)), n_db=100)
    assert got["ranked_ids"] == list(range(1, 21))
    assert got["n_truncated"] == 5


def test_judge_ids_of_unreadable_output_is_empty():
    got = judge_ids(None, n_db=10)
    assert got["ranked_ids"] == []
    assert got["n_raw"] == 0


# ---- 指標 ---------------------------------------------------------------------

def test_metrics_from_ranks_standard_ap_at_20():
    # 正解 3 本：1 位、3 位、上位 20 件の外。AP = (1/1 + 2/3) / 3 = 5/9
    m = metrics_from_ranks([1, 3, MISS_RANK])
    assert m["K"] == 3
    assert m["recall@20"] == pytest.approx(2 / 3)
    assert m["AP@20"] == pytest.approx(5 / 9)


def test_metrics_from_ranks_counts_rank_20_and_drops_rank_21():
    m = metrics_from_ranks([21, 20])
    assert m["recall@20"] == pytest.approx(0.5)
    assert m["AP@20"] == pytest.approx((1 / 20) / 2)


def test_ap_full_counts_ranks_beyond_20_but_not_misses():
    # 8/31 報告の MAP（並べた全体）: 1 位と 30 位、1 本は並べた中にない。AP = (1/1 + 2/30) / 3
    assert ap_full([1, 30, MISS_RANK]) == pytest.approx((1 + 2 / 30) / 3)
    assert metrics_from_ranks([1, 30, MISS_RANK])["AP@20"] == pytest.approx(1 / 3)


def test_ranks_in_list_counts_invalid_ids_as_occupied_ranks():
    # 2 番目は DB にない ID（None）。正解 "a" は 3 位で、2 位には繰り上がらない
    ranks = ranks_in_list([5, None, 3, 8], {3: "a", 5: "b", 8: "c"}, ["a", "z"])
    assert sorted(ranks) == [3, MISS_RANK]
    m = metrics_from_ranks(ranks)
    assert m["recall@20"] == pytest.approx(0.5)
    assert m["AP@20"] == pytest.approx((1 / 3) / 2)


def test_ranks_in_list_dedups_correct_keys():
    assert ranks_in_list([1], {1: "a"}, ["a", "a"]) == [1]


def test_summarize_means_per_split_and_sd_across_splits():
    recs = [
        {"setting": "A", "seed": 42, "recall@20": 1.0, "AP@20": 1.0},
        {"setting": "A", "seed": 42, "recall@20": 0.5, "AP@20": 0.25},
        {"setting": "A", "seed": 123, "recall@20": 0.5, "AP@20": 0.5},
        {"setting": "A", "seed": 123, "recall@20": 0.5, "AP@20": 0.25},
    ]
    s = summarize(recs)["A"]
    assert s["per_seed"]["42"] == {"n": 2, "recall@20": 0.75, "MAP@20": 0.625}
    assert s["per_seed"]["123"] == {"n": 2, "recall@20": 0.5, "MAP@20": 0.375}
    assert s["recall@20"]["mean"] == pytest.approx(0.625)
    assert s["recall@20"]["sd"] == pytest.approx(math.sqrt(2 * 0.125 ** 2))
    assert s["MAP@20"]["mean"] == pytest.approx(0.5)
    assert s["n_seeds"] == 2


# ---- 料金 ---------------------------------------------------------------------

def test_cost_opus_cache_write_standard_and_batch():
    usage = {"input_tokens": 1000, "cache_creation_input_tokens": 800_000,
             "cache_creation": {"ephemeral_5m_input_tokens": 800_000, "ephemeral_1h_input_tokens": 0},
             "cache_read_input_tokens": 0, "output_tokens": 2000}
    # 1000×4 + 800000×5 + 2000×20（USD / 100 万トークン）
    assert cost_usd(usage, "claude-opus-5-5") == pytest.approx(4.044)
    assert cost_usd(usage, "claude-opus-5-5", batch=True) == pytest.approx(2.022)


def test_cost_haiku_uses_long_prompt_prices_above_100k():
    long_usage = {"input_tokens": 500, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 900_000, "output_tokens": 1000}
    # 100,000 トークンを超えるプロンプト: 入力 0.50・読み出し 0.05・出力 2.50
    assert cost_usd(long_usage, "claude-haiku-5-5") == pytest.approx(0.04775)
    short_usage = {"input_tokens": 50_000, "cache_creation_input_tokens": 0,
                   "cache_read_input_tokens": 0, "output_tokens": 1000}
    assert cost_usd(short_usage, "claude-haiku-5-5") == pytest.approx(0.0055)


def test_cost_counts_one_hour_cache_writes_at_their_price():
    usage = {"input_tokens": 0, "cache_creation_input_tokens": 100_000,
             "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 100_000},
             "cache_read_input_tokens": 0, "output_tokens": 0}
    assert cost_usd(usage, "claude-sonnet-5-5") == pytest.approx(0.4)


# ---- プロンプトの組み立て --------------------------------------------------------

EQS = [
    {"source_id": "S1", "eq_id": "e1", "equation": "a = b\n+ c",
     "variables": {"a": "first", "b": "second", "T(z, t)": "temp"}, "context_text": "ctx one", "domain": "D1"},
    {"source_id": "S1", "eq_id": "e2", "equation": "x = y", "variables": {"x": "ex"},
     "context_text": "ctx two", "domain": "D1"},
    {"source_id": "S2", "eq_id": "e3", "equation": "p = q", "variables": {}, "context_text": "", "domain": "D2"},
]


def _lines(text):
    return text.splitlines()


def test_db_text_gives_one_line_per_equation_with_short_ids():
    lines = _lines(db_text(EQS, "eq", version="v1"))
    assert "1: a = b + c" in lines          # 式の中の改行は空白にする
    assert "2: x = y" in lines
    assert "3: p = q" in lines
    i_s1, i_s2 = lines.index("## Source: S1"), lines.index("## Source: S2")
    assert i_s1 < lines.index("1: a = b + c") < lines.index("2: x = y") < i_s2 < lines.index("3: p = q")
    assert lines.count("### Domain: D1") == 1  # 続く同じ分野ラベルは 1 回だけ書く
    assert lines.count("### Domain: D2") == 1
    text = db_text(EQS, "eq", version="v1")
    assert "first" not in text and "ctx one" not in text


def test_db_text_variable_options_keep_symbols_with_commas_intact():
    sym = db_text(EQS, "eq+sym", version="v1")
    assert "variables: a ; b ; T(z, t)" in sym
    assert "first" not in sym
    defs = db_text(EQS, "eq+def", version="v1")
    assert "variables: a = first ; b = second ; T(z, t) = temp" in defs
    ctx = db_text(EQS, "eq+ctx", version="v1")
    assert "context: ctx one" in ctx
    assert ctx.count("context:") == 2      # 文章のない式には context 行を書かない


def test_db_text_is_deterministic():
    assert db_text(EQS, "eqc+miss", version="v2") == db_text(copy.deepcopy(EQS), "eqc+miss", version="v2")
    assert db_text(EQS, "eqd", version="v3") == db_text(copy.deepcopy(EQS), "eqd", version="v3")


CASES = [
    {"case_id": "c1", "context": "Model a\nreactor.", "input_variables": ["T(z, t)", "b", " "],
     "output_variables": ["a"], "correct_model_ids": ["S1__e1", "S9__missing", "S1__e1", "S2__e3"]},
    {"case_id": "c2", "context": "Second.", "input_variables": ["y"], "output_variables": ["x"],
     "correct_model_ids": ["S1__e2"]},
]
KEY_TO_ID = {"S1__e1": 1, "S1__e2": 2, "S2__e3": 3}


def test_examples_text_lists_answer_ids_in_db_only_once():
    text = examples_text(CASES, KEY_TO_ID, version="v1")
    assert "Description: Model a reactor." in text
    assert "Input variables: T(z, t) ; b" in text
    assert "Output variables: a" in text
    assert "Equations: 1, 3" in text        # DB にない ID と重複は書かない
    assert "Equations: 2" in text
    assert text.count("Description:") == 2


def test_task_text_has_no_answer():
    text = task_text(CASES[0], version="v1")
    assert "Description: Model a reactor." in text
    assert "Input variables: T(z, t) ; b" in text
    assert "Equations:" not in text


def test_build_params_puts_cache_breakpoints_before_the_test_case():
    db, ex = db_text(EQS, "eqc+miss", version="v2"), examples_text(CASES, KEY_TO_ID, version="v2")
    p1 = build_params("claude-opus-5-5", "medium", db, ex, task_text(CASES[0], version="v2"), fmt="eqc+miss", version="v2")
    p2 = build_params("claude-opus-5-5", "medium", db, ex, task_text(CASES[1], version="v2"), fmt="eqc+miss", version="v2")
    blocks1, blocks2 = p1["messages"][0]["content"], p2["messages"][0]["content"]
    assert [b.get("cache_control") is not None for b in blocks1] == [True, True, False]
    assert blocks1[:2] == blocks2[:2]        # DB と参考ケースは同じ分割の全ケースで同じ
    assert blocks1[2] != blocks2[2]
    assert "Model a reactor." in blocks1[2]["text"]
    assert "Equations:" not in blocks1[2]["text"] and "E:" not in blocks1[2]["text"]
    assert p1["system"] == p2["system"]
    assert p1["output_config"]["effort"] == "medium"
    assert p1["output_config"]["format"]["type"] == "json_schema"
    json.dumps(p1)                            # そのまま API に送れる（JSON にできる）


# ---- 途中からの再開 ---------------------------------------------------------------

def test_pending_cases_skips_answered_and_retries_failed_calls():
    cfg = {"model": "m", "effort": "medium", "db_format": "eq", "examples": "all", "prompt_version": "v1"}
    recs = [
        {**cfg, "setting": "A", "seed": 42, "case_id": "c1", "error": None, "text": "{}"},
        {**cfg, "setting": "A", "seed": 42, "case_id": "c2", "error": "api: timeout", "text": None},
        {**cfg, "prompt_version": "v0", "setting": "A", "seed": 42, "case_id": "c3", "error": None, "text": "{}"},
        {**cfg, "setting": "B", "seed": 42, "case_id": "c4", "error": None, "text": "{}"},
        {**cfg, "examples": "300", "setting": "A", "seed": 42, "case_id": "c5", "error": None, "text": "{}"},
    ]
    plan = [("A", 42, "c1"), ("A", 42, "c2"), ("A", 42, "c3"), ("A", 42, "c4"), ("A", 42, "c5")]
    assert pending_cases(recs, plan, cfg) == [("A", 42, "c2"), ("A", 42, "c3"), ("A", 42, "c4"), ("A", 42, "c5")]


# ---- 固定した分割（データの検査） ------------------------------------------------------

@pytest.mark.skipif(not SPLITS.exists(), reason="splits.json がまだない")
def test_fixed_splits_are_disjoint_and_cover_each_setting():
    d = json.loads(SPLITS.read_text(encoding="utf-8"))
    excluded = set(d["provenance"].get("excluded_case_ids", []))
    # 2026-10-21：外す一覧 v3（92 件 = 未知の変数が決まらない出典横断の 91 件と、2 冊の教科書の式から成る単一出典の
    # core_019_v1）を外してから分割した（設定 A 1,823 → 1,731、B は 1,000 のまま）
    assert len(excluded) == 92 and "core_019_v1" in excluded
    assert d["n_cases"] == {"A": 1823 - 92, "B": 1000}
    for setting, by_seed in d["splits"].items():
        assert len(by_seed) == 10
        for seed, sp in by_seed.items():
            tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
            assert not (tr & va) and not (tr & te) and not (va & te)
            assert len(tr | va | te) == d["n_cases"][setting]
            assert not ((tr | va | te) & excluded)
            assert te


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.json がまだない")
def test_fixed_splits_match_the_recorded_per_case_outputs():
    # splits を作ったとき、--check で与えた per-case 出力のテストの並びと比べた結果。すべて一致していること
    d = json.loads(SPLITS.read_text(encoding="utf-8"))
    results = [ok for by_seed in d["check"].values() for checks in by_seed.values() for ok in checks.values()]
    assert results and all(results)
    assert {s for s, by_seed in d["check"].items() if any(by_seed.values())} >= {"B"}


# ---- 参考ケース ------------------------------------------------------------------

SPLIT = {"train": ["c1", "c2", "c3", "c4", "c5"], "val": ["c6"], "test": ["c7"]}
ALL = {f"c{i}": {"case_id": f"c{i}"} for i in range(1, 8)}


def test_ref_cases_are_train_and_val_in_file_order():
    assert [c["case_id"] for c in ref_cases(SPLIT, ALL)] == ["c1", "c2", "c3", "c4", "c5", "c6"]


def test_ref_cases_subset_is_fixed_by_seed_and_never_includes_test():
    a = [c["case_id"] for c in ref_cases(SPLIT, ALL, max_examples=3, seed=42)]
    b = [c["case_id"] for c in ref_cases(SPLIT, ALL, max_examples=3, seed=42)]
    assert a == b
    assert len(a) == 3 and "c7" not in a
    assert a == sorted(a, key=lambda x: int(x[1:]))   # ファイルの順を保つ


# ---- 式の表記の短縮（案 4。構造の部分だけを短くし、各式の変数記号の見え方を変えない） ----------------

@pytest.mark.parametrize("latex, symbols, want", [
    # \left・\right を除き、英数字どうしの間以外の空白を除く
    (r"\left( a + b \right)", ["a", "b"], "(a+b)"),
    (r"\left. \frac{dy}{dx} \right|_{x=0}", ["y", "x"], r"dy/dx|_{x=0}"),
    # 制御綴りの後の空白（\alpha x）と、明示の空白（\ ）は残す
    (r"\alpha x = \beta", ["x"], r"\alpha x=\beta"),
    (r"a \ b", ["a", "b"], r"a\ b"),
    # 空白を含む記号は、その記号の中の空白を残す
    (r"x = \Delta H_R + y", ["x", r"\Delta H_R", "y"], r"x=\Delta H_R+y"),
    (r"T(z, t) = 5", ["T(z, t)"], "T(z, t)=5"),
    # 2 文字以上の記号は、命令の中から始まっていても（\% MOR の % MOR）そのまま残す
    (r"\% MOR = a", ["% MOR", "a"], r"\% MOR=a"),
    # \frac は分子・分母が 1 つの記号なら括弧なし、そうでなければ括弧で囲む
    (r"\frac{dC_A}{dt} = \frac{F_{A0} - F_A}{V}", ["C_A", "t", "F_{A0}", "F_A", "V"],
     r"dC_A/dt=(F_{A0}-F_A)/V"),
    # 前後が掛け算・割り算になる分数は全体を括弧で囲む（意味を変えない）
    (r"2\frac{a}{b}c", ["a", "b", "c"], "2(a/b)c"),
    (r"x/\frac{a}{b}", ["x", "a", "b"], "x/(a/b)"),
    # 入れ子の分数
    (r"\frac{\frac{a}{b}}{c}", ["a", "b", "c"], "(a/b)/c"),
    # 1 文字だけの添字・指数の波括弧と \mathrm{1 文字}、細い空白
    (r"x^{2} + \mathrm{d}t\,y", ["x", "t", "y"], "x^2+dt y"),
])
def test_compress_latex_shortens_structure_only(latex, symbols, want):
    assert compress_latex(latex, symbols) == want


def test_compress_latex_keeps_symbol_strings_and_visibility():
    # C_{A} を C_A にすると、記号 C_A が新しく見えるようになる（見え方が変わる）ので、その規則は使わない
    assert compress_latex(r"C_{A} + x^{2}", ["C_A", "x"]) == "C_{A}+x^{2}"
    # 記号の中の \mathrm{} や波括弧は変えない
    assert compress_latex(r"\mathrm{Re} = \frac{\rho u L}{\mu}", [r"\mathrm{Re}", r"\rho", "u", "L", r"\mu"]) == \
        r"\mathrm{Re}=(\rho u L)/\mu"


# ---- 版 2 のプロンプト（短い見出し・短縮した式・記号のずれの補足） ------------------------------------

def test_db_text_v2_compresses_equations_and_adds_symbols_written_differently():
    lines = _lines(db_text(EQS, "eqc+miss", version="v2"))
    assert lines[:4] == ['<equation_database count="3">', "# S1", "## D1", "1: a=b+c"]
    # 式 1 の変数 T(z, t) は式の表記に現れないので、記号を補う
    assert lines[4] == "   symbols: T(z, t)"
    assert "2: x=y" in lines
    assert lines.index("# S2") < lines.index("## D2") < lines.index("3: p=q")
    assert "symbols:" not in db_text(EQS, "eqc", version="v2")


def test_examples_and_task_v2_use_short_labels_without_losing_fields():
    text = examples_text(CASES, KEY_TO_ID, version="v2")
    assert "D: Model a reactor.\nI: T(z, t) ; b\nO: a\nE: 1, 3" in text
    assert "D: Second.\nI: y\nO: x\nE: 2" in text
    task = task_text(CASES[0], version="v2")
    assert "<task>\nD: Model a reactor.\nI: T(z, t) ; b\nO: a\n</task>" in task


def test_key_index_counts_only_symbols_that_stand_as_symbols():
    idx = KeyIndex(["t", "x", "C_A", "d", "f"])
    # \left・\right・\frac の綴りの中の t・f は記号として現れていない
    assert idx.real(r"\left(x\right)") == {"x"}
    assert idx.real(r"\frac{dC_A}{dt}") == {"d", "C_A", "t"}
    assert idx.real("dC_A/dt") == {"d", "C_A", "t"}


def test_compress_latex_never_makes_another_equations_symbol_appear():
    # C_A がほかの式の記号なら、C_{A} を C_A にすると C_A が新しく現れるので、その規則は使わない
    idx = KeyIndex(["C_A", "x"])
    assert compress_latex(r"C_{A} + x^{2}", ["x"], idx) == "C_{A}+x^{2}"
    # 命令の綴りの中の 1 文字（\frac の f や \left の t）が消えるのは、記号の消失に数えない
    idx = KeyIndex(["f", "r", "a", "c", "b", "t"])
    assert compress_latex(r"\left( \frac{a}{b} \right)", ["a", "b"], idx) == "(a/b)"


# ---- 費用の見積もり（案 4・2・1。1 分割あたり） ----------------------------------------------

P = {"in": 1.0, "w5m": 2.0, "read": 0.5, "out": 10.0}   # USD / 100 万トークン（計算の確かめ用）


def test_split_cost_single_call_reads_prefix_and_writes_examples_once():
    # 1 回に全部（案 4・2）：1 件あたり (DB＋参考) の読み出し＋テストケース＋出力。書き込みは DB（最初の分割だけ）と参考ケース
    # 1 件: 150×0.5 + 10×1 + 5×10 = 135、書き込み: (100 + 50)×2 = 300 → (2×135 + 300) / 1e6
    assert split_cost(P, db=100, ex=50, task=10, n=2, out=5, first=True, option="single") == pytest.approx(570 / 1e6)
    assert split_cost(P, db=100, ex=50, task=10, n=2, out=5, first=False, option="single") == pytest.approx(370 / 1e6)


def test_split_cost_two_halves_and_merge():
    # 案 1：(DB の半分＋参考) を 2 回と、参考＋候補 40 式で 1 回。候補はキャッシュしない（cand トークン）
    # 1 件: (2×(50+50) + 50)×0.5 + (3×10 + 20)×1 + 3×5×10 = 125 + 50 + 150 = 325
    # 書き込み: DB 100（最初の分割）＋ 参考 50×3 → 250×2 = 500 → (2×325 + 500) / 1e6
    assert split_cost(P, db=100, ex=50, task=10, n=2, out=5, first=True, option="halves",
                      cand=20) == pytest.approx(1150 / 1e6)


def test_price_for_uses_long_prompt_prices_of_haiku():
    assert price_for("claude-haiku-5-5", 900_000)["read"] == 0.05
    assert price_for("claude-haiku-5-5", 50_000)["read"] == 0.01
    assert price_for("claude-opus-5-5", 900_000)["read"] == 0.20


# ---- 版 3（ゼロショット：DB だけ・出典なし・無作為な並び、全ケース × 3 回） ------------------------------

def test_shuffled_keys_is_a_fixed_permutation_independent_of_input_order():
    keys = [f"S{i % 3}__e{i}" for i in range(50)]
    a = shuffled_keys(keys, 7)
    assert sorted(a) == sorted(keys)
    assert a == shuffled_keys(list(reversed(keys)), 7)
    assert a != sorted(keys)                     # 出典ごとのまとまりが崩れている
    assert a != shuffled_keys(keys, 8)


def test_db_text_v3_gives_id_domain_equation_without_sources():
    text = db_text(EQS, "eqd", version="v3")
    lines = text.splitlines()
    assert lines[0] == '<equation_database count="3">'
    assert lines[1:4] == ["1\tD1\ta = b + c", "2\tD1\tx = y", "3\tD2\tp = q"]   # 式の表記は元のまま（改行だけ空白）
    assert lines[4] == "</equation_database>"
    assert "S1" not in text and "S2" not in text and "first" not in text and "ctx one" not in text


def test_build_params_v3_sends_db_and_one_task_only():
    db = db_text(EQS, "eqd", version="v3")
    p1 = build_params("claude-haiku-5-5", "high", db, None, task_text(CASES[0], version="v3"), fmt="eqd", version="v3")
    p2 = build_params("claude-haiku-5-5", "high", db, None, task_text(CASES[1], version="v3"), fmt="eqd", version="v3")
    b1, b2 = p1["messages"][0]["content"], p2["messages"][0]["content"]
    assert len(b1) == 2
    assert b1[0] == b2[0] and b1[0]["cache_control"]["type"] == "ephemeral"
    assert "cache_control" not in b1[1]
    assert "Description: Model a reactor." in b1[1]["text"] and "Input variables: T(z, t) ; b" in b1[1]["text"]
    assert "<examples" not in json.dumps(p1) and "Equations:" not in b1[1]["text"]
    assert p1["output_config"]["effort"] == "high"


def test_pending_cases_by_case_for_v3():
    cfg = {"model": "m", "effort": "high", "db_format": "eqd", "examples": "none", "prompt_version": "v3"}
    recs = [{**cfg, "case_id": "c1", "error": None},
            {**cfg, "case_id": "c2", "error": "api 529"},
            {**cfg, "effort": "medium", "case_id": "c3", "error": None}]
    plan = [("c1",), ("c2",), ("c3",)]
    assert pending_cases(recs, plan, cfg, fields=("case_id",)) == [("c2",), ("c3",)]


def test_v3_plan_is_the_distinct_test_cases_of_both_settings():
    # 例題を渡さないので、同じケースの答えはどの分割でも使い回す。呼び出しは異なるテストケースごとに 1 回
    splits = {"A": {"1": {"test": ["c2", "c1"]}, "2": {"test": ["c3"]}},
              "B": {"1": {"test": ["c1", "c4"]}, "2": {"test": ["c5"]}}}
    cases = {f"c{i}": {} for i in range(1, 6)}
    assert v3_plan(splits, [1], cases) == ["c1", "c2", "c4"]
    assert v3_plan(splits, [1, 2], cases) == ["c1", "c2", "c3", "c4", "c5"]


def test_split_summary_means_per_split_and_sd_across_splits():
    splits = {"A": {"42": {"test": ["c1", "c2"]}, "123": {"test": ["c2", "c3"]}}}
    per_case = {"c1": {"recall@20": 1.0, "AP@20": 1.0}, "c2": {"recall@20": 0.0, "AP@20": 0.0},
                "c3": {"recall@20": 0.5, "AP@20": 0.25}}
    s = split_summary(per_case, splits, "A", [42, 123])
    # 種 42 は (1.0 + 0.0)/2 = 0.5、種 123 は (0.0 + 0.5)/2 = 0.25（c2 の答えを両方の分割で使う）
    assert s["per_seed"]["42"] == {"n": 2, "recall@20": 0.5, "MAP@20": 0.5}
    assert s["per_seed"]["123"] == {"n": 2, "recall@20": 0.25, "MAP@20": 0.125}
    assert s["recall@20"]["mean"] == pytest.approx(0.375)
    assert s["recall@20"]["sd"] == pytest.approx(math.sqrt(2 * 0.125 ** 2))


def test_compare_splits_counts_wins_without_a_test():
    # 検定はしない（本人の決定、2026-10-20）。分割ごとの値を対にして、何通りで上回ったかを数える
    c = compare_splits({"1": 0.5, "2": 0.6, "3": 0.7, "4": 0.2}, {"1": 0.4, "2": 0.6, "3": 0.6, "4": 0.3})
    assert c == {"n": 4, "mean_diff": pytest.approx(0.025), "wins": 2, "ties": 1, "losses": 1}
    assert "wilcoxon_p" not in c


DB_IDS = ROOT / "experiments" / "llm_db_baseline" / "db_ids.json"
DB_IDS_V3 = ROOT / "experiments" / "llm_db_baseline" / "db_ids_v3.json"


@pytest.mark.skipif(not (DB_IDS.exists() and DB_IDS_V3.exists()), reason="ID の対応表がまだない")
def test_fixed_v3_order_is_the_seeded_shuffle_of_all_equations():
    keys = json.loads(DB_IDS.read_text(encoding="utf-8"))["keys"]
    v3 = json.loads(DB_IDS_V3.read_text(encoding="utf-8"))
    assert v3["keys"] == shuffled_keys(keys, v3["seed"])
    assert len(v3["keys"]) == len(set(v3["keys"])) == 11146


@pytest.mark.skipif(not SPLITS.exists(), reason="splits.json がまだない")
def test_v3_cases_are_all_setting_a_cases_including_setting_b():
    d = json.loads(SPLITS.read_text(encoding="utf-8"))["splits"]
    cases = json.loads((ROOT / "training_cases.json").read_text(encoding="utf-8"))
    ids = v3_case_ids({c["case_id"]: c for c in cases}, d)
    assert len(ids) == len(set(ids)) == 1731
    b = set(d["B"]["42"]["train"]) | set(d["B"]["42"]["val"]) | set(d["B"]["42"]["test"])
    assert len(b) == 1000 and b <= set(ids)


def test_test_records_keeps_the_seed_and_the_test_split_only():
    # 第 1 段のセッションの出力は、検証用のケース（split "val"）も含み、全部の種を 1 つのファイルに持つことがある
    pcs = [{"seed": 42, "case_id": "a", "split": "test"}, {"seed": 42, "case_id": "b", "split": "val"},
           {"seed": 123, "case_id": "c", "split": "test"}, {"seed": 42, "case_id": "d"}]
    assert [r["case_id"] for r in split_test_records(pcs, 42)] == ["a", "d"]   # split がない記録はテスト（旧い出力）
    assert [r["case_id"] for r in split_test_records([{"case_id": "e"}], 7)] == ["e"]   # 種ごとのファイル（seed なし）
