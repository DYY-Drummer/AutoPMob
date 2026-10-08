"""LLM が生成したケース側の説明（記号なしの説明文・言い換え・別の記号）の品質の確認.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.2
問題は "分類|詳細" の文字列で返す。hard は作り直しの対象、review は目視の対象。
生成結果が壊れていても（JSON として読めた形なら）例外は出さず、符号で返す。
"""
from __future__ import annotations

import re
import unicodedata

MATH_CHARS = set("$\\_^{}")
_START_ARTICLES = {"A", "I"}  # 文頭では冠詞・代名詞として現れる 1 文字
_MIN_BARE_LEN = 3             # 装飾を除いた形を調べる最小の長さ（a_t → at、t_o → to は普通の語と衝突する）
_ENGLISH_2 = frozenset({      # 2 文字の英語の停止語（冠詞、前置詞、代名詞など）
    "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in",
    "is", "it", "me", "my", "no", "of", "oh", "on", "or", "ox", "so",
    "to", "up", "us", "we"})
_SENTENCE_START = re.compile(r"[.!?]\s+\Z")  # 直前が「文末の記号 + 空白」


def bare(symbol: str) -> str:
    """LaTeX の装飾を除いた形（\\alpha → alpha、T_{in} → Tin）."""
    return re.sub(r"[\\{}$_^\s]", "", symbol or "")


def _as_str(x) -> str:
    """文字列以外（数・list・dict・None）は空文字にする。壊れた出力に str のメソッドを呼ばないため."""
    return x if isinstance(x, str) else ""


def _word(token: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])")


def _single_leak(tok: str, text: str) -> bool:
    """1 文字の記号 tok が、冠詞・代名詞ではなく記号として現れているか（目視の対象か）.

    a: 冠詞は後ろに空白と文字が続く。(a)・a,・a)・文末の a は記号の可能性がある。
    A・I: 文頭、または文末の記号（. ! ?）と空白の直後だけを冠詞・代名詞とみなす。
      (A)・[A]・area A・文中の I などは記号の可能性がある。
    それ以外の 1 文字: 単独の語として現れたら対象。
    """
    if tok == "a":
        return bool(re.search(r"(?<![A-Za-z0-9])a(?![A-Za-z0-9])(?!\s+[A-Za-z])", text))
    if tok in _START_ARTICLES:
        return any(m.start() > 0 and not _SENTENCE_START.search(text[:m.start()])
                   for m in _word(tok).finditer(text))
    return bool(_word(tok).search(text))


def symbol_leaks(text: str, symbols: list) -> tuple:
    """(hard, review)。
    hard: 数式記号の文字（MATH_CHARS）、2 文字以上の記号の単語としての出現。
      装飾のない記号はそのまま調べる。装飾のある記号（\\alpha、T_{in}）は、装飾を除いた形が
      3 文字以上、または 2 文字で _ENGLISH_2 に含まれない英語停止語のときだけ調べる
      （a_t → at、t_o → to のような英語停止語は普通の語と衝突して誤検出になる）。
    review: 装飾のない 1 文字の記号の単独の語としての出現（_single_leak。冠詞 a・文頭の A, I は見ない）."""
    text = text or ""
    hard, review = [], []
    if any(ch in text for ch in MATH_CHARS):
        hard.append("math-char")
    for s in symbols:
        raw = (s or "").strip()
        toks = [raw] if raw and not set(raw) & MATH_CHARS else []
        stripped = bare(s)
        if stripped != raw and (len(stripped) >= _MIN_BARE_LEN or
                                (len(stripped) == 2 and stripped.lower() not in _ENGLISH_2)):
            toks.append(stripped)
        for tok in toks:
            if len(tok) >= 2:
                if _word(tok).search(text):
                    hard.append(f"symbol:{s}")
                    break
            elif _single_leak(tok, text):
                review.append(f"single|{s}")
                break
    return hard, review


def normalize_desc(s: str) -> str:
    """小文字化し、角括弧の単位と句読点を除く（写しの判定用）."""
    s = re.sub(r"\[[^\]]*\]", " ", (s or "").lower())
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def word_jaccard(a: str, b: str) -> float:
    A, B = set(normalize_desc(a).split()), set(normalize_desc(b).split())
    return len(A & B) / len(A | B) if (A | B) else 0.0


def align_symbol(symbol: str, payload_symbols: list) -> str:
    """返された記号を payload の記号の綴りにそろえる（Unicode の NFC だけ違う記号を同じ記号として扱う）.

    LLM は U+2126（OHM SIGN）のような文字をそのまま返せず、NFC の U+03A9（GREEK CAPITAL LETTER OMEGA）で返す。
    見た目が同じで NFC が等しい 2 つの文字列を別の記号と数えると、作り直しても記号の集合が合わず通らない。
    - 返された記号が payload の記号そのものなら、そのまま返す。
    - そうでなく、NFC が等しい payload の記号がちょうど 1 つなら、その payload の記号を返す。
    - それ以外はそのまま返す。payload の記号どうしが NFC で等しいときは、どちらを指すか決められないので
      写さず、完全一致を求める。
    """
    if symbol in payload_symbols:
        return symbol
    nfc = unicodedata.normalize("NFC", symbol)
    same = [s for s in payload_symbols if unicodedata.normalize("NFC", s) == nfc]
    return same[0] if len(same) == 1 else symbol


def check_output(payload: dict, output) -> tuple:
    """生成結果 1 件を確かめる。(hard, review) の問題の一覧.

    壊れた出力（JSON として読めた形）でも例外は出さず、符号で返す。
    - None・空は no-output|、dict でない値は no-output|malformed。
    - variables が list でなければ空とみなす。dict でない要素、symbol が文字列でない要素は無視する
      （件数が合わなくなるので symbol-set になる）。
    - 返された symbol が payload の記号と NFC だけ違うときは、payload の綴りにそろえて比べる（align_symbol）。
    - context_nosym・paraphrase・alt_symbol が文字列でなければ空文字として扱う。
    alt_symbol は表記の違い（\\theta と「\\theta 」、T_j と T_{j}）を無視して（bare で）比べる。
    DB の記号は $ なしの LaTeX なので、$ を含む alt_symbol は alt-format（どの記号とも一致しない）。
    """
    if not output:
        return ["no-output|"], []
    if not isinstance(output, dict):
        return ["no-output|malformed"], []
    syms = [v["symbol"] for v in payload["variables"]]
    descs = {v["symbol"]: v["descriptions"] for v in payload["variables"]}
    raw_list = output.get("variables")
    got_list = raw_list if isinstance(raw_list, list) else []
    got = {align_symbol(v["symbol"], syms): v for v in got_list
           if isinstance(v, dict) and isinstance(v.get("symbol"), str)}
    hard, review = [], []
    if sorted(got) != sorted(syms) or len(got_list) != len(syms):
        hard.append("symbol-set|" + ",".join(sorted(set(syms) ^ set(got))))
    ctx = _as_str(output.get("context_nosym"))
    if not ctx.strip():
        hard.append("context-empty|")
    h, r = symbol_leaks(ctx, syms)
    hard += [f"context-leak|{x}" for x in h]
    review += [f"context-{x}" for x in r]
    seen, alts = {}, []
    for s in syms:
        v = got.get(s)
        if v is None:
            continue
        p = _as_str(v.get("paraphrase"))
        if not p.strip():
            hard.append(f"para-empty|{s}")
        else:
            h, r = symbol_leaks(p, syms)
            hard += [f"para-leak|{s}:{x}" for x in h]
            review += [f"para-{x}" for x in r]
            key = normalize_desc(p)
            if any(key == normalize_desc(d) for d in descs.get(s, [])):
                hard.append(f"verbatim|{s}")
            if key in seen:
                hard.append(f"dup-para|{s}={seen[key]}")
            else:
                seen[key] = s
        alt = _as_str(v.get("alt_symbol")).strip()
        a = bare(alt)                                    # 表記の違い（\theta と T_{j} と T_j など）を無視して比べる
        if not a or a == bare(s):
            hard.append(f"alt-same|{s}")
        if "$" in alt:
            hard.append(f"alt-format|{s}")
        alts.append((s, alt, a))
    orig = {bare(s) for s in syms}
    values = [a for _, _, a in alts]
    for s, alt, a in alts:
        if a and a in orig and a != bare(s):
            hard.append(f"alt-collides|{s}->{alt}")
        if a and values.count(a) > 1:
            hard.append(f"alt-dup|{s}->{alt}")
    return hard, review
