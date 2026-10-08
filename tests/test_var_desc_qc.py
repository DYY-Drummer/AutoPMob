"""var_desc_qc.py（生成結果の品質の確認）の単体テスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2, §4 の 6
"""
import copy
import unicodedata

import pytest

from var_desc_qc import bare, symbol_leaks, normalize_desc, word_jaccard, check_output, align_symbol


def test_bare_removes_latex_decorations():
    assert bare("\\alpha") == "alpha" and bare("T_{in}") == "Tin" and bare("$C_A$") == "CA"


def test_symbol_leaks_hard_and_review():
    syms = ["T", "T_{in}", "\\alpha", "UA", "a"]
    assert symbol_leaks("temperature $T$ of the reactor", syms)[0]     # 数式記号の文字
    assert symbol_leaks("feed T_{in}", syms)[0]
    assert symbol_leaks("overall UA value", syms)[0]                   # 2 文字以上の記号
    assert symbol_leaks("the alpha coefficient", syms)[0]              # 装飾を除いた形
    hard, review = symbol_leaks("temperature of the feed entering a reactor", syms)
    assert hard == [] and review == []                                 # 冠詞 a は見ない
    hard, review = symbol_leaks("the T of the jacket", syms)
    assert hard == [] and review == ["single|T"]                       # 1 文字は目視へ
    hard, review = symbol_leaks("A reactor with a jacket", ["A"])
    assert hard == [] and review == []                                 # 文頭の A は冠詞として扱う


def test_normalize_and_overlap():
    assert normalize_desc("Inlet temperature for reactor vessel [K]") == "inlet temperature for reactor vessel"
    assert word_jaccard("temperature of the feed", "inlet temperature [K]") == 1 / 5


PAYLOAD = {"context": "Reactor with jacket; share variables $T$, $V$.",
           "variables": [
               {"symbol": "T", "role": "input", "descriptions": ["temperature inside the reactor vessel [K]"]},
               {"symbol": "T_j", "role": "input", "descriptions": ["temperature inside the jacket [K]"]},
               {"symbol": "V", "role": "output", "descriptions": ["reactor vessel volume [L]"]}]}


def good():
    return {"context_nosym": "Reactor with jacket; the reactor temperature and volume are shared.",
            "variables": [
                {"symbol": "T", "paraphrase": "temperature of the reactor contents",
                 "alt_symbol": "\\theta", "meaning_conflict": False},
                {"symbol": "T_j", "paraphrase": "temperature of the cooling jacket",
                 "alt_symbol": "T_c", "meaning_conflict": False},
                {"symbol": "V", "paraphrase": "volume of liquid held in the reactor",
                 "alt_symbol": "V_R", "meaning_conflict": False}]}


def _has(codes, cat):
    return any(c.startswith(cat + "|") for c in codes)


def test_check_output_good_has_no_hard_issue():
    hard, review = check_output(PAYLOAD, good())
    assert hard == [] and review == []


def test_check_output_detects_each_problem():
    o = good(); o["variables"][0]["paraphrase"] = "Temperature inside the reactor vessel"
    assert _has(check_output(PAYLOAD, o)[0], "verbatim")
    o = good(); o["variables"][1]["paraphrase"] = o["variables"][0]["paraphrase"]
    assert _has(check_output(PAYLOAD, o)[0], "dup-para")
    o = good(); o["variables"][2]["alt_symbol"] = "V"
    assert _has(check_output(PAYLOAD, o)[0], "alt-same")
    o = good(); o["variables"][2]["alt_symbol"] = "T"
    assert _has(check_output(PAYLOAD, o)[0], "alt-collides")
    o = good(); o["variables"][1]["alt_symbol"] = "\\theta"
    assert _has(check_output(PAYLOAD, o)[0], "alt-dup")
    o = good(); o["variables"].pop()
    assert _has(check_output(PAYLOAD, o)[0], "symbol-set")
    o = good(); o["context_nosym"] = "share $T$"
    assert _has(check_output(PAYLOAD, o)[0], "context-leak")
    o = good(); o["variables"][0]["paraphrase"] = "temperature T_{j} nearby"
    assert _has(check_output(PAYLOAD, o)[0], "para-leak")
    assert check_output(PAYLOAD, None) == (["no-output|"], [])


def test_check_output_flags_empty_context():
    o = good(); o["context_nosym"] = ""
    assert "context-empty|" in check_output(PAYLOAD, o)[0]
    o = good(); o["context_nosym"] = "   "                               # 空白だけも空
    assert "context-empty|" in check_output(PAYLOAD, o)[0]


def test_check_output_flags_empty_paraphrase():
    o = good(); o["variables"][0]["paraphrase"] = ""
    assert "para-empty|T" in check_output(PAYLOAD, o)[0]
    o = good(); o["variables"][0]["paraphrase"] = "   "
    assert "para-empty|T" in check_output(PAYLOAD, o)[0]


# --- 壊れた出力でも例外を出さず、符号で返す -------------------------------------------------------


def _without_symbol():
    o = good()
    del o["variables"][0]["symbol"]
    return o


def _variable0(**fields):
    o = good()
    o["variables"][0].update(fields)
    return o


def _top(**fields):
    o = good()
    o.update(fields)
    return o


BROKEN_OUTPUTS = [
    pytest.param(_without_symbol, "symbol-set|T", id="item-without-symbol"),
    pytest.param(lambda: _variable0(symbol=None), "symbol-set|T", id="symbol-none"),
    pytest.param(lambda: _variable0(symbol=5), "symbol-set|T", id="symbol-int"),
    pytest.param(lambda: _top(variables=["T"] + good()["variables"][1:]), "symbol-set|T", id="str-item"),
    pytest.param(lambda: _top(variables=good()["variables"] + ["T"]), "symbol-set|", id="extra-str-item"),   # 件数で気づく
    pytest.param(lambda: _top(variables="T,T_j,V"), "symbol-set|T,T_j,V", id="variables-str"),
    pytest.param(lambda: _top(variables={"T": good()["variables"][0]}), "symbol-set|T,T_j,V", id="variables-dict"),
    pytest.param(lambda: _top(context_nosym=5), "context-empty|", id="context-int"),
    pytest.param(lambda: _variable0(paraphrase=5), "para-empty|T", id="paraphrase-int"),
    pytest.param(lambda: _variable0(alt_symbol=5), "alt-same|T", id="alt-int"),
]


@pytest.mark.parametrize("make, code", BROKEN_OUTPUTS)
def test_check_output_does_not_raise_on_broken_output(make, code):
    hard, review = check_output(PAYLOAD, make())
    assert hard and code in hard


@pytest.mark.parametrize("bad", [[good()], ["T", "T_j", "V"], "{}", 5],
                         ids=["list", "list-of-str", "str", "int"])
def test_check_output_non_dict_output_is_malformed(bad):
    assert check_output(PAYLOAD, bad) == (["no-output|malformed"], [])


@pytest.mark.parametrize("empty", [None, {}, [], ""],
                         ids=["none", "empty-dict", "empty-list", "empty-str"])
def test_check_output_empty_output_is_no_output(empty):
    assert check_output(PAYLOAD, empty) == (["no-output|"], [])


# --- 記号 A・I・a は冠詞・代名詞と紛れるので、そう読めるときだけ目視から外す -----------------------------


def test_symbol_leaks_A_and_I_are_review_unless_sentence_initial():
    assert symbol_leaks("cross-sectional area (A)", ["A"]) == ([], ["single|A"])
    assert symbol_leaks("current (I) in the cell", ["I"]) == ([], ["single|I"])
    assert symbol_leaks("area [A] of the wall", ["A"]) == ([], ["single|A"])
    assert symbol_leaks('the "A" of the pipe', ["A"]) == ([], ["single|A"])
    assert symbol_leaks("cross-sectional area A", ["A"]) == ([], ["single|A"])        # 文中
    assert symbol_leaks("the heat transfer area A of the wall", ["A"]) == ([], ["single|A"])
    assert symbol_leaks("A reactor with a jacket. A pump feeds it.", ["A"]) == ([], [])
    assert symbol_leaks("Is it hot? A pump feeds it. I think so! I stop.", ["A", "I"]) == ([], [])
    assert symbol_leaks("A reactor with area A", ["A"]) == ([], ["single|A"])         # 文頭は冠詞、あとの A は記号
    assert symbol_leaks("a jacket.A pump", ["A"]) == ([], ["single|A"])               # 句点の後に空白がなければ文頭でない


def test_symbol_leaks_lowercase_a_is_review_unless_followed_by_a_word():
    assert symbol_leaks("concentration of species a", ["a"]) == ([], ["single|a"])    # 文末
    assert symbol_leaks("rate constant (a) of step one", ["a"]) == ([], ["single|a"])
    assert symbol_leaks("a, b and c are constants", ["a"]) == ([], ["single|a"])
    assert symbol_leaks("the pair (b, a)", ["a"]) == ([], ["single|a"])
    assert symbol_leaks("a heat exchanger", ["a"]) == ([], [])
    assert symbol_leaks("of a  reactor and a\njacket", ["a"]) == ([], [])             # 空白は複数でも改行でもよい


def test_symbol_leaks_other_single_char_symbols_keep_the_review_rule():
    assert symbol_leaks("T of the jacket", ["T"]) == ([], ["single|T"])               # 文頭でも見る（A・I だけ例外）
    assert symbol_leaks("the rate (k) of step one", ["k"]) == ([], ["single|k"])


def test_check_output_single_letter_symbol_in_paraphrase():
    """言い換え内に単独の 1 文字シンボル（記号でない）があると para-single| で報告される."""
    o = good()
    o["variables"][1]["paraphrase"] = "temperature V of the cooling jacket"
    hard, review = check_output(PAYLOAD, o)
    assert any(r.startswith("para-single|") for r in review)


def test_check_output_catches_decorated_alt_duplicates():
    """別の記号が装飾形式で重複すると alt-dup| で検出される."""
    o = good()
    o["variables"][0]["alt_symbol"] = "T_c"       # bare form: "Tc"
    o["variables"][1]["alt_symbol"] = "T_{c}"     # bare form: "Tc" (装飾形式だが同じ)
    hard, review = check_output(PAYLOAD, o)
    assert any(c.startswith("alt-dup|") for c in hard)


# --- 別の記号は表記の違いを無視して比べる。$ は使わない ------------------------------------------------


def test_check_output_alt_symbol_comparison_ignores_notation():
    o = good(); o["variables"][2]["alt_symbol"] = "$V$"
    hard = check_output(PAYLOAD, o)[0]
    assert _has(hard, "alt-same") and _has(hard, "alt-format")        # $ を外せば元の記号 V と同じ
    o = good(); o["variables"][2]["alt_symbol"] = " V "
    assert _has(check_output(PAYLOAD, o)[0], "alt-same")              # 空白だけの違い
    o = good(); o["variables"][2]["alt_symbol"] = "T_{j}"
    assert _has(check_output(PAYLOAD, o)[0], "alt-collides")          # T_j は同じケースの別の記号
    o = good(); o["variables"][0]["alt_symbol"] = "\\theta"; o["variables"][1]["alt_symbol"] = "\\theta "
    assert _has(check_output(PAYLOAD, o)[0], "alt-dup")               # 末尾の空白だけの違い
    o = good(); o["variables"][0]["alt_symbol"] = "\\vartheta"
    assert not any(c.startswith("alt-") for c in check_output(PAYLOAD, o)[0])   # \theta とは別の記号


def test_check_output_alt_symbol_with_dollar_is_a_format_error_only():
    o = good(); o["variables"][0]["alt_symbol"] = "$\\theta$"
    assert check_output(PAYLOAD, o)[0] == ["alt-format|T"]            # 記号としては新しいので alt-same 等は出ない


# --- 装飾を除いた短い形は普通の語と衝突するので、3 文字以上だけを調べる --------------------------------------


def test_symbol_leaks_short_stripped_forms_are_not_hard():
    assert symbol_leaks("temperature at the inlet", ["a_t"]) == ([], [])              # a_t → at
    assert symbol_leaks("flow to the tank", ["t_o"]) == ([], [])                      # t_o → to
    assert symbol_leaks("the alpha coefficient", ["\\alpha"])[0]                      # 3 文字以上は従来どおり
    assert symbol_leaks("feed Tin value", ["T_{in}"])[0]
    assert symbol_leaks("the Re number", ["Re"])[0]                                   # 装飾のない 2 文字は従来どおり


def test_symbol_leaks_short_decorated_forms_with_english_stop_words():
    """2文字の装飾を除いた形が英語の停止語なら、シンボルとして扱わない."""
    assert symbol_leaks("As the flow rises", ["A_s"]) == ([], [])                     # bare: "As" → "as" (停止語)
    assert symbol_leaks("he said", ["h_e"]) == ([], [])                               # bare: "he" (停止語)


def test_symbol_leaks_short_decorated_forms_without_english_stop_words():
    """2文字の装飾を除いた形が停止語でなければ、シンボルリークとして検出する (hard)."""
    assert symbol_leaks("the mu coefficient", ["\\mu"])[0]                           # bare: "mu" (停止語でない)
    assert symbol_leaks("Cp of the stream", ["C_p"])[0]                               # bare: "Cp" → "cp" (停止語でない)
    assert symbol_leaks("the CA value", ["C_A"])[0]                                   # bare: "CA" → "ca" (停止語でない)
    assert symbol_leaks("T0 at the start", ["T_0"])[0]                                # bare: "T0" → "t0" (停止語でない)


# --- 返された記号が payload の記号と Unicode の正規化（NFC）だけ違うとき --------------------------------------
# LLM は U+2126（OHM SIGN）をそのまま返せず、NFC の U+03A9（GREEK CAPITAL LETTER OMEGA）で返す。
# どちらも見た目が同じで別の文字列なので、ソースではエスケープで書いて区別する。

OHM = "\u2126_{0}"       # payload の綴り（NFC ではない）
OMEGA = "\u03a9_{0}"     # LLM が返す綴り（OHM の NFC）


def _ohm_payload():
    """PAYLOAD の 2 つ目の記号 T_j を OHM に替えたもの（説明は同じ）."""
    p = copy.deepcopy(PAYLOAD)
    p["variables"][1]["symbol"] = OHM
    return p


def _returning(symbol):
    """good() の 2 つ目の変数の記号を symbol にした出力."""
    o = good()
    o["variables"][1]["symbol"] = symbol
    return o


def test_nfc_spellings_used_below_are_different_strings_with_the_same_nfc():
    assert OHM != OMEGA and unicodedata.normalize("NFC", OHM) == OMEGA        # 以下のテストの前提


def test_check_output_accepts_a_symbol_that_differs_from_the_payload_only_by_nfc():
    payload = _ohm_payload()
    assert check_output(payload, _returning(OHM)) == ([], [])                 # payload と同じ綴り
    hard, review = check_output(payload, _returning(OMEGA))                   # NFC だけ違う綴り
    assert not any(c.startswith("symbol-set|") for c in hard)
    assert (hard, review) == ([], [])


def test_check_output_judges_the_nfc_variable_as_usual_under_the_payload_spelling():
    payload = _ohm_payload()
    o = _returning(OMEGA); o["variables"][1]["paraphrase"] = "Temperature inside the jacket"
    assert check_output(payload, o)[0] == [f"verbatim|{OHM}"]                 # 写しは見つかる。符号は payload の綴り
    o = _returning(OMEGA); o["variables"][1]["paraphrase"] = " "
    assert check_output(payload, o)[0] == [f"para-empty|{OHM}"]
    o = _returning(OMEGA); o["variables"][1]["alt_symbol"] = "\u2126_0"
    assert check_output(payload, o)[0] == [f"alt-same|{OHM}"]                 # 別の記号の確認も、この変数に対して行う


def test_check_output_still_counts_the_items_when_nfc_spellings_collapse():
    """同じ変数が 2 つの綴り（OHM と OMEGA）で返されたら、件数が合わないので symbol-set| になる."""
    payload = _ohm_payload()
    o = _returning(OMEGA)
    dup = copy.deepcopy(o["variables"][1])
    dup["symbol"], dup["paraphrase"] = OHM, "liquid level in the jacket"
    o["variables"].insert(2, dup)
    assert _has(check_output(payload, o)[0], "symbol-set")


def test_check_output_requires_exact_symbols_when_payload_symbols_share_an_nfc_form():
    payload = copy.deepcopy(PAYLOAD)
    payload["variables"][1]["symbol"], payload["variables"][2]["symbol"] = OHM, OMEGA   # NFC が等しい 2 つの記号

    def returning(second, third):
        o = good()
        o["variables"][1]["symbol"], o["variables"][2]["symbol"] = second, third
        return o

    assert check_output(payload, returning(OHM, OMEGA)) == ([], [])           # 完全一致なら通る
    assert check_output(payload, returning(OMEGA, OMEGA))[0] == [f"symbol-set|{OHM}"]   # 片方の綴りを 2 回: 写さない
    assert check_output(payload, returning(OHM, OHM))[0] == [f"symbol-set|{OMEGA}"]


def test_check_output_does_not_map_a_third_spelling_when_two_payload_symbols_share_its_nfc():
    acute, oxia, decomposed = "\u03ac_1", "\u1f71_1", "\u03b1\u0301_1"        # どれも NFC は U+03AC
    assert len({acute, oxia, decomposed}) == 3
    assert {unicodedata.normalize("NFC", s) for s in (acute, oxia, decomposed)} == {acute}
    payload = copy.deepcopy(PAYLOAD)
    payload["variables"][1]["symbol"], payload["variables"][2]["symbol"] = acute, oxia
    o = good()
    o["variables"][1]["symbol"] = decomposed                                  # acute とも oxia とも NFC が等しい
    o["variables"][2]["symbol"] = oxia
    assert check_output(payload, o)[0] == [f"symbol-set|{acute},{decomposed}"]    # どちらか決められないので写さない


def test_align_symbol_returns_the_payload_spelling_only_when_it_is_unambiguous():
    assert align_symbol(OMEGA, ["T", OHM]) == OHM                    # NFC が等しい payload の記号が 1 つ → その綴り
    assert align_symbol(OHM, ["T", OHM]) == OHM                      # payload の記号そのもの
    assert align_symbol("T", ["T", OHM]) == "T"
    assert align_symbol("S", ["T", OHM]) == "S"                      # NFC が等しい payload の記号がない → そのまま
    assert align_symbol(OMEGA, [OHM, OMEGA]) == OMEGA                # payload の記号そのものなら、変えない
    assert align_symbol(OHM, [OHM, OMEGA]) == OHM
    acute, oxia, decomposed = "\u03ac_1", "\u1f71_1", "\u03b1\u0301_1"   # どれも NFC は U+03AC
    assert align_symbol(decomposed, [acute]) == acute                # payload の側が 1 つなら写す
    assert align_symbol(decomposed, [acute, oxia]) == decomposed     # NFC が等しい payload の記号が 2 つ → 写さない
