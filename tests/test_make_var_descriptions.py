"""make_var_descriptions.py のオフラインの単体テスト（API は呼ばない）.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2
"""
import json
from types import SimpleNamespace

import pytest
from anthropic.types.messages import MessageBatchIndividualResponse

import make_var_descriptions as mvd
from make_var_descriptions import (
    MODEL, OUTPUT_SCHEMA, PROMPT_VERSION, batch_error_text, build_parser, by_case, case_payload,
    consumable, judge, load_batches, request_params, parse_message, pick, record, uncollected,
)
from var_desc_qc import check_output

CASE = {"case_id": "toy_1", "context": "Reactor sharing $T$.",
        "input_variables": ["T", "F"], "output_variables": ["C_A"],
        "correct_model_ids": ["src__eq_2", "src__eq_1"], "variant_type": "original"}
EQK = {"src__eq_1": {"variables": {"T": "temperature in the tank [K]", "F": "feed flow"}},
       "src__eq_2": {"variables": {"C_A": "concentration of A", "T": "reactor temperature"}}}


def test_case_payload_order_roles_descriptions():
    p = case_payload(CASE, EQK)
    assert p["context"] == "Reactor sharing $T$."
    assert [v["symbol"] for v in p["variables"]] == ["T", "F", "C_A"]
    assert [v["role"] for v in p["variables"]] == ["input", "input", "output"]
    assert p["variables"][0]["descriptions"] == ["reactor temperature", "temperature in the tank [K]"]


def test_request_params_uses_opus_effort_and_schema():
    prm = request_params(case_payload(CASE, EQK))
    assert prm["model"] == MODEL == "claude-opus-5-5"
    assert prm["output_config"]["effort"] == "medium"
    assert prm["output_config"]["format"] == {"type": "json_schema", "schema": OUTPUT_SCHEMA}
    assert not {"thinking", "temperature", "tool_choice"} & set(prm)
    assert json.loads(prm["messages"][0]["content"])["variables"][2]["symbol"] == "C_A"


def _msg(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, model=MODEL,
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=20))


def test_parse_message():
    assert parse_message(_msg('{"a": 1}')) == ({"a": 1}, None)
    assert parse_message(_msg("", "refusal")) == (None, "refusal")
    out, err = parse_message(_msg("not json"))
    assert out is None and err.startswith("json")


GOOD = {"context_nosym": "Reactor sharing the temperature.",
        "variables": [
            {"symbol": "T", "paraphrase": "temperature of the liquid in the reactor",
             "alt_symbol": "\\theta", "meaning_conflict": False},
            {"symbol": "F", "paraphrase": "volumetric feed rate", "alt_symbol": "q", "meaning_conflict": False},
            {"symbol": "C_A", "paraphrase": "amount of species A per volume",
             "alt_symbol": "c_A", "meaning_conflict": False}]}


def test_pick_prefers_latest_passing_record():
    payload = case_payload(CASE, EQK)
    bad = {**GOOD, "context_nosym": "has $T$"}
    rec, failed = pick([{"output": GOOD, "time": "1"}, {"output": bad, "time": "2"}], payload)
    assert rec["time"] == "1" and failed is False
    rec, failed = pick([{"output": bad, "time": "2"}], payload)
    assert rec["time"] == "2" and failed is True
    rec, failed = pick([{"output": None, "time": "3"}], payload)
    assert rec is None and failed is True


def test_pick_excludes_malformed_output():
    payload = case_payload(CASE, EQK)
    rec, failed = pick([{"output": ["not", "a", "dict"], "time": "4"}], payload)
    assert rec is None and failed is True


def test_pick_excludes_a_consumable_output_missing_a_payload_variable():
    """形は読める（consumable）が payload の変数が 1 つ欠けた出力は、symbol-set| が付くので「完全」でなく、記録なしと同じ."""
    payload = case_payload(CASE, EQK)
    short = {**GOOD, "variables": GOOD["variables"][:2]}          # C_A が欠けている
    assert consumable(short) is True
    assert "symbol-set|C_A" in check_output(payload, short)[0]    # 欠けた記号が符号に出る
    assert judge({"output": short}, payload)[0] is False
    assert pick([{"output": short, "time": "8"}], payload) == (None, True)


def test_pick_returns_the_latest_of_several_passing_records():
    payload = case_payload(CASE, EQK)
    rec, failed = pick([{"output": GOOD, "time": "1"}, {"output": GOOD, "time": "2"}], payload)
    assert rec["time"] == "2" and failed is False


def test_pick_returns_the_latest_of_several_failing_but_complete_records():
    payload = case_payload(CASE, EQK)
    leak = {**GOOD, "context_nosym": "has $T$"}                   # 読める形だが context に記号が残る不合格
    assert consumable(leak) is True and judge({"output": leak}, payload)[0] is False
    rec, failed = pick([{"output": leak, "time": "1"}, {"output": leak, "time": "2"}], payload)
    assert rec["time"] == "2" and failed is True


# --- finalize が例外なく読める形でない出力は、合格にも「不合格だが完全」にもしない ---------------------
# check_output は hard が空か、para-empty|・context-empty| のような中身の問題だけを返す形でも、
# finalize は rec["output"]["context_nosym"]・v["paraphrase"]・v["meaning_conflict"] を直接読む。
UNCONSUMABLE = {
    "symbols-only": {"context_nosym": GOOD["context_nosym"],
                     "variables": [{"symbol": v["symbol"]} for v in GOOD["variables"]]},
    "no-context-nosym": {"variables": GOOD["variables"]},
    "paraphrase-is-int": {**GOOD, "variables": [{**GOOD["variables"][0], "paraphrase": 5},
                                                *GOOD["variables"][1:]]},
    "no-meaning-conflict": {**GOOD, "variables": [{k: x for k, x in v.items() if k != "meaning_conflict"}
                                                  for v in GOOD["variables"]]},
}


@pytest.mark.parametrize("name", list(UNCONSUMABLE))
def test_unconsumable_output_neither_passes_nor_is_picked(name):
    payload = case_payload(CASE, EQK)
    rec = {"output": UNCONSUMABLE[name], "time": "5"}
    assert judge(rec, payload)[0] is False
    assert pick([rec], payload) == (None, True)


def test_unconsumable_record_does_not_shadow_a_consumable_one():
    payload = case_payload(CASE, EQK)
    bad = {"output": UNCONSUMABLE["no-meaning-conflict"], "time": "6"}
    rec, failed = pick([{"output": GOOD, "time": "1"}, bad], payload)
    assert rec["time"] == "1" and failed is False
    rec, failed = pick([bad, {"output": {**GOOD, "context_nosym": "has $T$"}, "time": "7"}], payload)
    assert rec["time"] == "7" and failed is True       # 完全な不合格は従来どおり選ばれる


# --- 集計に使うのは今のプロンプト版の記録だけ --------------------------------------------------------
def test_by_case_keeps_only_current_prompt_version():
    payloads = {"toy_1": case_payload(CASE, EQK)}
    cur = {"case_id": "toy_1", "prompt_version": PROMPT_VERSION, "time": "2"}
    old = {"case_id": "toy_1", "prompt_version": "v0", "time": "1"}
    unversioned = {"case_id": "toy_1", "time": "0"}               # 版が不明な記録は今の版と確認できない
    other_case = {"case_id": "toy_2", "prompt_version": PROMPT_VERSION, "time": "3"}
    assert by_case([unversioned, old, cur, other_case], payloads) == {"toy_1": [cur]}
    assert by_case([old, unversioned], payloads) == {}


# --- バッチの記録には、送ったときのプロンプト版を付ける（collect で取り込むときに今の版を付けない） --------
def test_record_stamps_the_given_prompt_version():
    ok = record("toy_1", "batch", _msg('{"a": 1}'), prompt_version="v0")
    assert ok["prompt_version"] == "v0"
    failed = record("toy_1", "batch", None, "batch expired", prompt_version="v0")    # 成功でなかった結果も同じ
    assert failed["prompt_version"] == "v0" and failed["error"] == "batch expired" and failed["output"] is None
    assert record("toy_1", "batch", _msg('{"a": 1}'), prompt_version="")["prompt_version"] == ""   # None だけが既定


def test_record_prompt_version_defaults_to_the_current_one():
    assert record("toy_1", "pilot", _msg('{"a": 1}'))["prompt_version"] == PROMPT_VERSION
    assert record("toy_1", "pilot", _msg('{"a": 1}'), prompt_version=None)["prompt_version"] == PROMPT_VERSION
    assert record("toy_1", "retry", None, "429: slow down")["prompt_version"] == PROMPT_VERSION


def test_record_stamped_unknown_is_never_counted_as_current():
    """batches.json に版の記録がない項目は、collect が unknown を付ける。by_case はそれを今の版とは数えない."""
    payloads = {"toy_1": case_payload(CASE, EQK)}
    unknown = record("toy_1", "batch", _msg(json.dumps(GOOD)), prompt_version="unknown")
    cur = record("toy_1", "batch", _msg(json.dumps(GOOD)))
    assert PROMPT_VERSION != "unknown"
    assert by_case([unknown], payloads) == {}
    assert by_case([unknown, cur], payloads) == {"toy_1": [cur]}


# --- consumable: finalize が KeyError・AttributeError を出さずに読める形の述語（形だけを見る） -----------
_DROP = object()


def _var(**changes):
    """正しい変数 1 件。値が _DROP のキーは取り除く."""
    v = {"symbol": "T", "paraphrase": "p", "alt_symbol": "q", "meaning_conflict": False, **changes}
    return {k: x for k, x in v.items() if x is not _DROP}


BAD_OUTPUTS = {
    "none": None, "list": [], "str": "text", "int": 5, "empty-dict": {},
    "no-context": {"variables": []},
    "context-none": {"context_nosym": None, "variables": []},
    "context-int": {"context_nosym": 1, "variables": []},
    "no-variables": {"context_nosym": "c"},
    "variables-none": {"context_nosym": "c", "variables": None},
    "variables-str": {"context_nosym": "c", "variables": "T"},
    "variables-dict": {"context_nosym": "c", "variables": {"symbol": "T"}},
}
BAD_VARS = {
    "no-symbol": _var(symbol=_DROP), "no-paraphrase": _var(paraphrase=_DROP),
    "no-alt_symbol": _var(alt_symbol=_DROP), "no-meaning_conflict": _var(meaning_conflict=_DROP),
    "symbol-int": _var(symbol=5), "symbol-none": _var(symbol=None),
    "paraphrase-int": _var(paraphrase=5), "paraphrase-none": _var(paraphrase=None),
    "paraphrase-list": _var(paraphrase=["p"]),
    "alt_symbol-int": _var(alt_symbol=5), "alt_symbol-none": _var(alt_symbol=None),
    "meaning_conflict-0": _var(meaning_conflict=0),          # bool でなければ不可（int は bool の親）
    "meaning_conflict-1": _var(meaning_conflict=1),
    "meaning_conflict-str": _var(meaning_conflict="false"),
    "meaning_conflict-none": _var(meaning_conflict=None),
    "item-str": "T", "item-none": None, "item-int": 5, "item-list": ["T"],
}


def test_consumable_accepts_complete_outputs():
    assert consumable(GOOD) is True
    assert consumable({"context_nosym": "", "variables": []}) is True            # 形だけを見る。中身の良し悪しは check_output
    assert consumable({"context_nosym": "c", "variables": [_var(meaning_conflict=True)]}) is True


@pytest.mark.parametrize("name", list(BAD_OUTPUTS))
def test_consumable_rejects_wrong_top_level(name):
    assert consumable(BAD_OUTPUTS[name]) is False


@pytest.mark.parametrize("name", list(BAD_VARS))
def test_consumable_rejects_a_bad_variable(name):
    assert consumable({"context_nosym": "c", "variables": [_var(), BAD_VARS[name]]}) is False


def test_judge_passes_clean_consumable_output():
    assert judge({"output": GOOD}, case_payload(CASE, EQK))[0] is True


def test_missing_meaning_conflict_passes_qc_but_is_not_consumable():
    payload = case_payload(CASE, EQK)
    out = UNCONSUMABLE["no-meaning-conflict"]
    assert check_output(payload, out)[0] == []                  # check_output だけを見ると問題なし
    assert consumable(out) is False
    assert judge({"output": out}, payload)[0] is False          # それでも合格にはしない


# --- 回収していないバッチがあるあいだは submit しない -------------------------------------------------
def test_uncollected_lists_ids_of_batches_not_yet_collected():
    batches = [{"id": "msgbatch_1", "n": 3, "collected": True},
               {"id": "msgbatch_2", "n": 5, "collected": False},
               {"id": "msgbatch_3", "n": 1, "collected": False}]
    assert uncollected(batches) == ["msgbatch_2", "msgbatch_3"]
    assert uncollected(batches[:1]) == []
    assert uncollected([]) == []
    assert uncollected([{"id": "msgbatch_9"}]) == ["msgbatch_9"]   # collected の記録がなければ回収済みと言えない


def test_load_batches_reads_the_log_or_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(mvd, "RAW_DIR", tmp_path)                # 本物の記録ディレクトリには触れない
    assert load_batches() == []                                   # まだ 1 度も送っていない
    (tmp_path / "batches.json").write_text(
        json.dumps([{"id": "msgbatch_1", "n": 2, "collected": True},
                    {"id": "msgbatch_2", "n": 4, "collected": False}]), encoding="utf-8")
    assert uncollected(load_batches()) == ["msgbatch_2"]


def test_submit_parser_has_force_flag_defaulting_to_false():
    assert build_parser().parse_args(["submit"]).force is False
    assert build_parser().parse_args(["submit", "--force"]).force is True


def test_retry_parser_has_force_flag_defaulting_to_false():
    assert build_parser().parse_args(["retry"]).force is False
    assert build_parser().parse_args(["retry", "--force"]).force is True
    args = build_parser().parse_args(["retry", "--no-fallbacks", "--force"])        # 従来の --no-fallbacks と併用できる
    assert args.force is True and args.no_fallbacks is True


# --- バッチが成功でなかったとき、エラーの種類と本文を残す ----------------------------------------------
def _errored(etype, message):
    """anthropic 0.103.1 の入れ子: result(MessageBatchErroredResult).error(ErrorResponse).error(type, message)."""
    return SimpleNamespace(
        type="errored",
        error=SimpleNamespace(type="error", request_id="req_01",
                              error=SimpleNamespace(type=etype, message=message)))


def test_batch_error_text_errored_keeps_error_type_and_message():
    text = batch_error_text(_errored("invalid_request_error", "max_tokens: 16000 > 8192"))
    assert text == "batch errored: invalid_request_error: max_tokens: 16000 > 8192"


def test_batch_error_text_other_result_types():
    assert batch_error_text(SimpleNamespace(type="canceled")) == "batch canceled"
    assert batch_error_text(SimpleNamespace(type="expired")) == "batch expired"


def test_batch_error_text_tolerates_missing_levels():
    assert batch_error_text(SimpleNamespace(type="errored")) == "batch errored"
    assert batch_error_text(SimpleNamespace(type="errored", error=SimpleNamespace(type="error"))) == "batch errored"
    only_type = SimpleNamespace(type="errored", error=SimpleNamespace(
        type="error", error=SimpleNamespace(type="api_error")))
    assert batch_error_text(only_type) == "batch errored: api_error"
    only_message = SimpleNamespace(type="errored", error=SimpleNamespace(
        type="error", error=SimpleNamespace(message="boom")))
    assert batch_error_text(only_message) == "batch errored: boom"


def test_batch_error_text_with_real_sdk_objects():
    """SimpleNamespace の入れ子が SDK の実物と同じ形であることの確認（ネットワークは使わない）."""
    errored = MessageBatchIndividualResponse.model_validate({
        "custom_id": "toy_1",
        "result": {"type": "errored", "error": {"type": "error", "error": {
            "type": "overloaded_error", "message": "Overloaded"}}}})
    assert batch_error_text(errored.result) == "batch errored: overloaded_error: Overloaded"
    for kind in ("canceled", "expired"):
        res = MessageBatchIndividualResponse.model_validate({"custom_id": "toy_1", "result": {"type": kind}})
        assert batch_error_text(res.result) == f"batch {kind}"


# --- finalize は、NFC だけ違う記号で返された変数を payload の綴りで書く ---------------------------------------
# payload の記号 OHM は U+2126（OHM SIGN）で始まる。LLM は NFC の綴り OMEGA（U+03A9 GREEK CAPITAL LETTER OMEGA）で返す。
# 見た目が同じで別の文字列なので、ソースではエスケープで書く。最終の JSON は payload の綴りを鍵にする
# （var_conditions.build_view が、ケースの元の記号で引くため）。
OHM, OMEGA = "\u2126_{0}", "\u03a9_{0}"
NFC_CASE = {"case_id": "toy_nfc", "context": "Tank with an initial level.",
            "input_variables": ["T", OHM], "output_variables": ["C_A"],
            "correct_model_ids": ["src__eq_3"], "variant_type": "original"}
NFC_EQK = {"src__eq_3": {"variables": {"T": "reactor temperature", OHM: "initial liquid level",
                                       "C_A": "concentration of A"}}}


def _nfc_output():
    """LLM の出力。payload の OHM の変数を、NFC の綴り OMEGA で返す."""
    return {"context_nosym": "Tank with a starting liquid height.",
            "variables": [
                {"symbol": "T", "paraphrase": "temperature of the liquid in the reactor",
                 "alt_symbol": "\\theta", "meaning_conflict": False},
                {"symbol": OMEGA, "paraphrase": "liquid height at the start of the run",
                 "alt_symbol": "h_s", "meaning_conflict": False},
                {"symbol": "C_A", "paraphrase": "amount of species A per volume",
                 "alt_symbol": "c_A", "meaning_conflict": False}]}


def test_finalize_writes_a_variable_under_the_payload_spelling_not_the_returned_one(tmp_path, monkeypatch):
    payload = case_payload(NFC_CASE, NFC_EQK)
    assert [v["symbol"] for v in payload["variables"]] == ["T", OHM, "C_A"]     # 前提: payload は OHM の綴り
    rec = record("toy_nfc", "batch", _msg(json.dumps(_nfc_output())))            # JSON を読み戻すと OMEGA の綴りになる
    assert rec["output"]["variables"][1]["symbol"] == OMEGA != OHM
    monkeypatch.setattr(mvd, "payloads_all", lambda: {"toy_nfc": payload})      # 本物の payload・記録・出力先には触れない
    monkeypatch.setattr(mvd, "load_raw", lambda: [rec])
    monkeypatch.setattr(mvd, "OUT_JSON", tmp_path / "var_descriptions.json")
    monkeypatch.setattr(mvd, "QC_JSON", tmp_path / "var_descriptions_qc.json")
    monkeypatch.setattr(mvd, "SAMPLE_CSV", tmp_path / "var_descriptions_sample50.csv")
    mvd.cmd_finalize(SimpleNamespace())                                          # 記録がないケースがあれば SystemExit(1)
    final = json.loads((tmp_path / "var_descriptions.json").read_text(encoding="utf-8"))["cases"]["toy_nfc"]
    assert list(final["variables"]) == ["T", OHM, "C_A"]                         # 鍵は payload の綴り（OMEGA ではない）
    assert final["variables"][OHM]["paraphrase"] == "liquid height at the start of the run"
    assert final["qc_failed"] is False and final["qc_hard"] == []
    qc = json.loads((tmp_path / "var_descriptions_qc.json").read_text(encoding="utf-8"))
    assert qc["missing"] == [] and qc["qc_failed"] == [] and qc["n_variables"] == 3   # 3 変数とも payload の記号で見つかる
