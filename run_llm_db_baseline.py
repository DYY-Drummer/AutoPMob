#!/usr/bin/env python3
"""LLM に数式データベースと参考ケースを渡し、テストケースごとに使う式の ID を選ばせる比較（LLM 直接選択）.

加藤先生の指示（2026-10-14 のミーティング）による、8/31 報告の「LLM 直接生成」のやり直し:
- 出発点は文献から抽出した数式データベース。LLM にもデータベースを渡し、並べ替える性能を比べる（文献は渡さない）。
- テストケースごとに API を 1 回呼ぶ。毎回、その分割の訓練ケースと検証ケース（説明文・入出力変数・正解式の ID）を参考として渡す。
  テストケースは渡さない。
- 判定は ID だけで行い、等価性の判定はしない。DB にない ID と読めない出力は、その順位を占める不正解とする。
  同じ ID が重なったら 2 回目以降は除く。
- 指標は Recall@20 と MAP（標準の定義。上位 20 件に入らなかった正解式は P(e) = 0）。R@K と自由度ゼロ停止は使わない。
- 分割を 3 通り（乱数の種 42・123・456）にして各分割のテストケースを 1 回ずつ解かせる。分割間のぶれが大きければ増やす。

  python3 run_llm_db_baseline.py splits                 # 分割と ID の対応を JSON に固定する（パイプラインのコードを使うのはここだけ）
  python3 run_llm_db_baseline.py count --models ...     # count_tokens で 1 回分の入力トークン数を数える（無料）
  python3 run_llm_db_baseline.py pilot --n 2 ...        # 形式を確かめる試しの呼び出し（本人の承認前は 1〜3 件まで）
  python3 run_llm_db_baseline.py run ...                # 本実行（通常の API・プロンプトキャッシュ・途中から再開できる）
  python3 run_llm_db_baseline.py evaluate ...           # 判定・指標・既存の手法との比較

生の応答はすべて experiments/llm_db_baseline/raw/<設定の名前>.jsonl に追記する。
分割は splits.json、短い ID（1〜N）と式の ID（eq_key）の対応は db_ids.json に固定する。どちらも splits で 1 回だけ作る。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"
OUT_DIR = EXP / "llm_db_baseline"
RAW_DIR = OUT_DIR / "raw"
SPLITS_JSON = OUT_DIR / "splits.json"
DB_IDS_JSON = OUT_DIR / "db_ids.json"
DB_IDS_V3_JSON = OUT_DIR / "db_ids_v3.json"
EQ_JSON = ROOT / "unified_equations.json"
CASES_JSON = ROOT / "training_cases.json"

PROMPT_VERSION = "v3"         # v1: 10-14 の試しの呼び出し、v2: 10-15 の案 4 の試算（どちらも関数を残す）
V3_ORDER_SEED = 20261015      # 版 3 で DB の式を無作為に並べる乱数の種（db_ids_v3.json に固定）
TOP_N = 20
MISS_RANK = 10_000           # 上位に入らなかった正解式の順位（evaluate_multi_eq と同じ値）
ALL_SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
DEFAULT_SEEDS = [42, 123, 456]
# set_aware_reranker.py --variants と同じ絞り込み（接尾辞 "_" は前方一致）。拡張ケースはここで除かれる。
SETTINGS = {"A": ["original", "multisource_", "dae_"], "B": ["dae_"]}
SEP = " ; "                  # 変数記号の区切り（記号の中に現れない。記号には "," や ";" を含むものがある）

# DB の式ごとに何を渡すか。式の表記・分野ラベル・出典（見出し）は常に渡す。
DB_FORMATS = {
    "eq": {"vars": "none", "context": False},
    "eq+sym": {"vars": "symbols", "context": False},
    "eq+def": {"vars": "definitions", "context": False},
    "eq+ctx": {"vars": "none", "context": True},
    "full": {"vars": "definitions", "context": True},
    # 版 2：式の表記を構造だけ短くしたもの（compress_latex）。+miss は、式の表記に同じ文字列で現れない変数記号を補う
    "eqc": {"vars": "none", "context": False, "compress": True},
    "eqc+miss": {"vars": "missing", "context": False, "compress": True},
    # 版 3：ID・分野ラベル・式の表記（元のまま）をタブで区切って 1 行に。出典は渡さない
    "eqd": {"vars": "none", "context": False},
}

# USD / 100 万トークン（2026-10-14 に料金表 platform.claude.com/docs/en/about-claude/pricing で確認）。
# in: キャッシュなしの入力、w5m・w1h: キャッシュの書き込み、read: キャッシュの読み出し、out: 出力（思考を含む）。
PRICES = {
    "claude-opus-5-5": {"in": 4.0, "w5m": 5.0, "w1h": 8.0, "read": 0.20, "out": 20.0},
    "claude-sonnet-5-5": {"in": 2.0, "w5m": 2.5, "w1h": 4.0, "read": 0.10, "out": 10.0},
    "claude-haiku-5-5": {"in": 0.10, "w5m": 0.125, "w1h": 0.20, "read": 0.01, "out": 0.50,
                         "long_over": 100_000,
                         "long": {"in": 0.50, "w5m": 0.625, "w1h": 1.0, "read": 0.05, "out": 2.50}},
    "claude-sonnet-4-5": {"in": 3.0, "w5m": 3.75, "w1h": 6.0, "read": 0.30, "out": 15.0},
    "claude-fable-5-1": {"in": 10.0, "w5m": 12.5, "w1h": 20.0, "read": 0.25, "out": 50.0},
}
CONTEXT_LIMIT = {"claude-opus-5-5": 1_000_000, "claude-sonnet-5-5": 1_000_000, "claude-haiku-5-5": 1_000_000,
                 "claude-fable-5-1": 1_000_000, "claude-sonnet-4-5": 200_000}  # Models API の max_input_tokens

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"equation_ids": {"type": "array", "items": {"type": "integer"}}},
    "required": ["equation_ids"],
    "additionalProperties": False,
}

VARS_NOTE = {
    "none": "",
    "symbols": ' Below each equation, "variables:" lists the symbols of its variables.',
    "definitions": ' Below each equation, "variables:" lists the symbols of its variables, each with its'
                   ' definition ("symbol = definition").',
}
CONTEXT_NOTE = ' "context:" is a short text that explains the physical meaning of the equation.'

SYSTEM_TEMPLATE = """You select equations for building a physical model.

A physical model here is a set of equations with which an engineer computes the output variables of a task from its input variables. The user message has three parts:

1. <equation_database>: equations extracted from the literature. Each equation has a numeric ID and is written in LaTeX. The equations are grouped by the source they were extracted from ("## Source") and by the domain label given to them at extraction ("### Domain").{notes}
2. <examples>: solved tasks. Each example gives a description of the model, its input variables and output variables, and, after "Equations:", the IDs of the database equations that form the model.
3. <task>: a new task with a description, input variables and output variables.

Choose the database equations that the engineer should use to build the model for the new task. List up to 20 IDs from the most to the least important: first the equations that form the model, then, if the model has fewer than 20 equations, other equations that are likely to be useful for this model, until the list has 20 IDs. Use only IDs from the database, and list each ID once. Variable symbols in the examples and the task are written as in the database; several symbols are separated by " ; "."""

FINAL_INSTRUCTION = "List the IDs of the database equations for this task, as instructed."

# 版 2（2026-10-15）：課題の説明、正解の式の組が満たす条件（8/31 報告 §2.1 の要件 1〜3。評価ケースの正解集合は
# すべて満たす）、データベースと参考ケースの読み方、答え方を書く。テストケースの答えにつながる情報は入れない。
SYSTEM_V2 = """You are an expert in physical modeling. You select equations from an equation database for building a physical model.

# Task
An engineer wants to build a physical model: a set of equations with which the output variables can be computed once the input variables are given. The engineer describes the model and lists its input variables and output variables. Choose, from the equation database, the equations that the engineer should use.

The set of equations that forms the model satisfies these conditions:
1. Every input variable and every output variable appears in at least one equation of the set.
2. The number of equations equals the number of variables of the set that are not input variables.
3. Every variable of the set that is neither an input nor an output variable appears in at least two equations of the set.
A variable is identified by its symbol: two variables are the same only when their symbols are written identically.

# Equation database
<equation_database> lists equations extracted from the literature. Each equation line reads "ID: equation", with the equation in LaTeX. A line starting with "# " names the source of the equations below it, and a line starting with "## " gives their domain label.{notes}

# Solved examples
<examples> lists solved tasks of the same kind. Each example has four lines: "D:" the description of the model, "I:" the input variables, "O:" the output variables, and "E:" the IDs of the database equations that form the model. Symbols in a list are separated by " ; ".

# Answer
For the task in <task>, list up to 20 equation IDs, from the most to the least likely to belong to the model. Put first the equations that you judge to form the model. If they are fewer than 20, continue with the other equations most likely to be needed for this model, until the list has 20 IDs. Use only IDs from the database, and list each ID once."""

# 版 3（2026-10-15、本人の指示）：ゼロショット。参考ケースは渡さない。DB は ID・式の表記・分野ラベルだけで、出典は渡さず、
# 式は固定した乱数の種で無作為に並べる（出典のまとまりや並びが手がかりにならないように）。
SYSTEM_V3 = """You are an expert in physical modeling. You select equations from an equation database for building a physical model.

# Task
An engineer wants to build a physical model: a set of equations with which the output variables can be computed once the input variables are given. The engineer describes the model and lists its input variables and output variables. Choose, from the equation database, the equations that the engineer should use.

The set of equations that forms the model satisfies these conditions:
1. Every input variable and every output variable appears in at least one equation of the set.
2. The number of equations equals the number of variables of the set that are not input variables.
3. Every variable of the set that is neither an input nor an output variable appears in at least two equations of the set.

# Equation database
<equation_database> lists equations extracted from the literature, in no particular order. Each line has three fields separated by tabs: the equation ID, the domain label of the equation, and the equation in LaTeX.

# Answer
For the task in <task>, list up to 20 equation IDs, from the most to the least likely to belong to the model. Put first the equations that you judge to form the model. If they are fewer than 20, continue with the other equations most likely to be needed for this model, until the list has 20 IDs. Use only IDs from the database, and list each ID once. In the task, variable symbols in a list are separated by " ; "."""

MISSING_NOTE = (' A line "   symbols:" below an equation lists symbols of variables of that equation that are'
                ' written differently in the equation itself.')
FINAL_V2 = "List the IDs of the database equations for this task."


# ---- 小さな道具（two_stage_query_conditioned.py の同名関数と同じ規則。パイプラインの書き換えの影響を受けないよう写す） ----

def norm(s) -> str:
    return (s or "").strip()


def get_src(e: dict) -> str:
    return norm(e.get("source_id") or "unknown") or "unknown"


def eq_key(e: dict):
    eid = norm(e.get("eq_id") or "")
    return eid if "__" in eid else f"{get_src(e)}__{eid}" if eid else None


def one_line(s) -> str:
    return " ".join(str(s or "").split())


def var_list(xs) -> list:
    """記号の並び（ケースに書かれた順。空白を除き、重複は 1 回）."""
    return list(dict.fromkeys(norm(x) for x in xs or [] if norm(x)))


def variant_matches(v: str, wanted: list) -> bool:
    return any(v.startswith(w) if w.endswith("_") else v == w for w in wanted)


# ---- 式の表記の短縮（案 4：DB 全体と全参考ケースを 1 回の入力に収めるための試み） ----------------------
# 構造の部分（\left・\right、細い空白、1 文字の \mathrm{}・添字の波括弧、\frac、空白）だけを短くする。
# 式の変数記号（DB の変数の一覧の記号）が現れる箇所はそのまま残し、各規則は、式の変数記号のうち
# 式の表記に文字列として現れるものの集合が変わらない場合にだけ使う（式ごと・規則ごとに確かめる）。

_SIZERS = {"\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg", "\\bigl", "\\bigr", "\\Bigl", "\\Bigr",
           "\\biggl", "\\biggr", "\\Biggl", "\\Biggr"}
_DELIMS = {"(", ")", "[", "]", "|", "/", "<", ">", "\\{", "\\}", "\\|", "\\langle", "\\rangle", "\\lfloor",
           "\\rfloor", "\\lceil", "\\rceil", "\\vert", "\\Vert", "\\lvert", "\\rvert", "\\lVert", "\\rVert"}
_THIN = {"\\,", "\\;", "\\:", "\\!", "\\quad", "\\qquad", "\\displaystyle", "\\textstyle", "\\scriptstyle"}
_FRACS = {"\\frac", "\\dfrac", "\\tfrac"}
_REL = {"\\\\", "\\le", "\\leq", "\\ge", "\\geq", "\\approx", "\\equiv", "\\sim", "\\simeq", "\\to", "\\rightarrow",
        "\\Rightarrow", "\\leftarrow", "\\Leftarrow", "\\leftrightarrow", "\\pm", "\\mp", "\\neq", "\\ne",
        "\\propto", "\\quad", "\\qquad", "\\in"}
_SAFE_PREV = {None, "=", "+", "-", "(", "[", "{", ",", "&", "<", ">", "|", ";", ":", "\\{"} | _REL
_SAFE_NEXT = {None, "=", "+", "-", ")", "]", "}", ",", "&", "<", ">", "|", ";", ":", ".", "\\}"} | _REL
_ATOMIC = re.compile(r"(?:\\[A-Za-z]+|[A-Za-z0-9.]+)(?:[_^](?:\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z0-9]))*'*")


def _tokens(x: str) -> list:
    """LaTeX を、制御綴り（\\alpha）・制御記号（\\,）・1 文字に分ける."""
    out, i, n = [], 0, len(x)
    while i < n:
        if x[i] == "\\" and i + 1 < n:
            j = i + 2
            if x[i + 1].isascii() and x[i + 1].isalpha():
                while j < n and x[j].isascii() and x[j].isalpha():
                    j += 1
            out.append(x[i:j])
            i = j
        else:
            out.append(x[i])
            i += 1
    return out


def _starts(toks: list) -> list:
    out, pos = [], 0
    for t in toks:
        out.append(pos)
        pos += len(t)
    return out


class KeyIndex:
    """変数記号の集まり。real(text) は、text の中に「記号として」現れる記号の集合を返す.

    記号として現れる＝字句の切れ目で始まり、字句の切れ目で終わる（\\left の綴りの中の t のような、
    命令の一部の文字は数えない）。
    """

    def __init__(self, keys):
        self.one, self.by2 = set(), {}
        for k in {norm(k) for k in keys if norm(k)}:
            if len(k) == 1:
                self.one.add(k)
            else:
                self.by2.setdefault(k[:2], []).append(k)

    def occurrences(self, toks: list):
        """(記号, 始まり, 終わり) の列。始まり・終わりは文字の位置."""
        text = "".join(toks)
        starts = _starts(toks)
        cuts = set(starts) | {len(text)}
        for p in starts:
            if text[p] in self.one and p + 1 in cuts:
                yield text[p], p, p + 1
            for k in self.by2.get(text[p:p + 2], ()):
                if text.startswith(k, p) and p + len(k) in cuts:
                    yield k, p, p + len(k)

    def real(self, text: str) -> set:
        return {k for k, _, _ in self.occurrences(_tokens(text))}


def _protected(toks: list, index: "KeyIndex") -> list:
    """index の記号が現れる箇所に含まれる字句は True.

    記号として現れる箇所に加えて、2 文字以上の記号は文字列として含まれる箇所も守る（\\% MOR の % MOR のように、
    命令の中から始まる記号があるため）。1 文字の記号は、命令の綴りの中の文字（\\left の t）を守らない。
    """
    text = "".join(toks)
    mark = [False] * len(text)
    for _, a, b in index.occurrences(toks):
        mark[a:b] = [True] * (b - a)
    for ks in index.by2.values():
        for k in ks:
            p = text.find(k)
            while p != -1:
                mark[p:p + len(k)] = [True] * len(k)
                p = text.find(k, p + 1)
    return [any(mark[st:st + len(t)]) for st, t in zip(_starts(toks), toks)]


def _next_nonspace(toks: list, i: int, step: int):
    j = i + step
    while 0 <= j < len(toks) and toks[j] == " ":
        j += step
    return j if 0 <= j < len(toks) else None


def _r_leftright(toks, prot):
    out, skip = [], set()
    for i, t in enumerate(toks):
        if i in skip:
            continue
        if t in _SIZERS and not prot[i]:
            j = _next_nonspace(toks, i, 1)
            if j is not None and toks[j] == "." and not prot[j]:
                skip.update(range(i + 1, j + 1))     # \left. と \right. は見えない区切り：区切りごと除く
                continue
            if j is not None and toks[j] in _DELIMS:
                continue
        out.append(t)
    return out


def _r_thin(toks, prot):
    return [" " if t in _THIN and not p else t for t, p in zip(toks, prot)]


def _r_mathrm1(toks, prot):
    out, i = [], 0
    while i < len(toks):
        w = toks[i:i + 4]
        if (len(w) == 4 and w[0] == "\\mathrm" and w[1] == "{" and len(w[2]) == 1 and w[2].isalnum()
                and w[3] == "}" and not any(prot[i:i + 4])):
            out.append(w[2])
            i += 4
        else:
            out.append(toks[i])
            i += 1
    return out


def _r_braces1(toks, prot):
    out, i = [], 0
    while i < len(toks):
        w = toks[i:i + 4]
        if (len(w) == 4 and w[0] in ("_", "^") and w[1] == "{" and len(w[2]) == 1 and w[2].isalnum()
                and w[3] == "}" and not any(prot[i:i + 4])):
            out += [w[0], w[2]]
            i += 4
        else:
            out.append(toks[i])
            i += 1
    return out


def _group(toks: list, i: int):
    """toks[i] から始まる {…} の終わり（閉じ括弧の次の位置）。波括弧で始まらなければ None."""
    if i is None or i >= len(toks) or toks[i] != "{":
        return None
    depth = 0
    for j in range(i, len(toks)):
        depth += toks[j] == "{"
        depth -= toks[j] == "}"
        if depth == 0:
            return j + 1
    return None


def _r_frac(toks, prot):
    """\\frac{A}{B} を A/B にする（A・B が 1 つの記号でなければ括弧で囲む）。内側の分数から順に."""
    toks, prot = list(toks), list(prot)
    blocked = set()
    while True:
        for i, t in enumerate(toks):
            if t not in _FRACS or prot[i] or i in blocked:
                continue
            a0 = _next_nonspace(toks, i, 1)
            a1 = _group(toks, a0)
            b0 = _next_nonspace(toks, a1 - 1, 1) if a1 else None
            b1 = _group(toks, b0)
            if b1 is None or any(prot[k] for k in (i, a0, a1 - 1, b0, b1 - 1)):
                blocked.add(i)
                continue
            if any(x in _FRACS for x in toks[a0 + 1:a1 - 1] + toks[b0 + 1:b1 - 1]):
                continue                              # 内側の分数を先に
            # 字句と保護の印を組で動かす（分子・分母の字句は印ごと残し、足す括弧と斜線は保護しない）
            num = list(zip(toks[a0 + 1:a1 - 1], prot[a0 + 1:a1 - 1]))
            den = list(zip(toks[b0 + 1:b1 - 1], prot[b0 + 1:b1 - 1]))
            par = lambda part: [("(", False)] + part + [(")", False)]
            wrap = lambda part: part if _ATOMIC.fullmatch("".join(t for t, _ in part).strip()) else par(part)
            new = wrap(num) + [("/", False)] + wrap(den)
            p, q = _next_nonspace(toks, i, -1), _next_nonspace(toks, b1 - 1, 1)
            if (toks[p] if p is not None else None) not in _SAFE_PREV or \
                    (toks[q] if q is not None else None) not in _SAFE_NEXT:
                new = par(new)                        # 前後が掛け算・割り算・指数になる分数は全体を囲む
            toks = toks[:i] + [t for t, _ in new] + toks[b1:]
            prot = prot[:i] + [f for _, f in new] + prot[b1:]
            blocked = {k if k < i else k - (b1 - i) + len(new) for k in blocked}
            break
        else:
            return toks


def _r_spaces(toks, prot):
    out = []
    for i, t in enumerate(toks):
        if t != " " or prot[i]:
            out.append(t)
            continue
        prev = out[-1] if out else ""
        j = _next_nonspace(toks, i, 1)
        nxt = toks[j] if j is not None else ""
        if prev and nxt and prev != " " and prev[-1].isalnum() and nxt[0].isalnum():
            out.append(" ")
    return out


_RULES = (_r_leftright, _r_thin, _r_mathrm1, _r_frac, _r_braces1, _r_spaces)


def compress_latex(latex: str, symbols, index: "KeyIndex | None" = None) -> str:
    """式の表記の構造の部分だけを短くする。symbols はその式の変数記号、index は DB の全変数記号.

    その式の変数記号が現れる箇所は変えない。各規則は、index の記号（index がなければ symbols）のうち
    式の表記に記号として現れるものの集合が、元の式と同じになる場合にだけ使う。
    """
    own = KeyIndex(symbols or [])
    check = index or own
    base = one_line(latex)
    visible = check.real(base)
    toks = _tokens(base)
    for rule in _RULES:
        cand = rule(toks, _protected(toks, own))
        if check.real("".join(cand)) == visible:
            toks = cand
    return "".join(toks).strip()


# ---- プロンプト ------------------------------------------------------------------

def system_prompt(fmt: str, version: str = PROMPT_VERSION) -> str:
    opt = DB_FORMATS[fmt]
    if version == "v1":
        notes = VARS_NOTE[opt["vars"]] + (CONTEXT_NOTE if opt["context"] else "")
        return SYSTEM_TEMPLATE.format(notes=notes)
    if version == "v3":
        return SYSTEM_V3
    return SYSTEM_V2.format(notes=MISSING_NOTE if opt["vars"] == "missing" else "")


def db_text(eqs: list, fmt: str, version: str = PROMPT_VERSION) -> str:
    """データベースの全式を、短い ID（データベースの順に 1, 2, ...）を付けて並べた文字列.

    出典が変わるところと分野ラベルが変わるところに見出しを置く（v1: "## Source:"・"### Domain:"、v2: "# "・"## "）。
    同じ入力からは同じ文字列を返す（プロンプトキャッシュの前提）。
    """
    opt = DB_FORMATS[fmt]
    if version == "v3":   # 1 行に「ID<タブ>分野ラベル<タブ>式の表記」。見出しも出典もなし。並びは呼び出し側（load_db）で決める
        body = [f"{i}\t{one_line(e.get('domain'))}\t{one_line(e.get('equation'))}" for i, e in enumerate(eqs, 1)]
        return "\n".join([f'<equation_database count="{len(eqs)}">', *body, "</equation_database>"])
    h_src, h_dom = ("## Source: ", "### Domain: ") if version == "v1" else ("# ", "## ")
    index = KeyIndex(k for e in eqs for k in (e.get("variables") or {})) if opt.get("compress") else None
    lines = [f'<equation_database count="{len(eqs)}">']
    prev_src = prev_dom = None
    for i, e in enumerate(eqs, 1):
        src, dom = get_src(e), one_line(e.get("domain"))
        if src != prev_src:
            lines.append(f"{h_src}{src}")
            prev_dom = None
        if dom != prev_dom:
            lines.append(f"{h_dom}{dom}")
        prev_src, prev_dom = src, dom
        variables = e.get("variables") or {}
        if not isinstance(variables, dict):
            variables = {}
        syms = [(norm(k), one_line(v)) for k, v in variables.items() if norm(k)]
        latex = (compress_latex(e.get("equation"), [s for s, _ in syms], index) if opt.get("compress")
                 else one_line(e.get("equation")))
        lines.append(f"{i}: {latex}")
        if opt["vars"] == "missing":
            miss = [s for s, _ in syms if s not in latex]
            if miss:
                lines.append("   symbols: " + SEP.join(miss))
        if opt["vars"] == "symbols" and syms:
            lines.append("   variables: " + SEP.join(s for s, _ in syms))
        elif opt["vars"] == "definitions" and syms:
            lines.append("   variables: " + SEP.join(f"{s} = {d}" if d else s for s, d in syms))
        ctx = one_line(e.get("context_text"))
        if opt["context"] and ctx:
            lines.append(f"   context: {ctx}")
    lines.append("</equation_database>")
    return "\n".join(lines)


_LABELS = {"v1": ("Description: ", "Input variables: ", "Output variables: ", "Equations: "),
           "v2": ("D: ", "I: ", "O: ", "E: "),
           "v3": ("Description: ", "Input variables: ", "Output variables: ", None)}


def _case_lines(case: dict, version: str = PROMPT_VERSION) -> list:
    d, i, o, _ = _LABELS[version]
    return [f"{d}{one_line(case.get('context'))}",
            f"{i}{SEP.join(var_list(case.get('input_variables')))}",
            f"{o}{SEP.join(var_list(case.get('output_variables')))}"]


def answer_ids(case: dict, key_to_id: dict) -> list:
    """正解式の短い ID（データベースにある式だけ、重複は 1 回）。set_aware_reranker と同じく DB にない ID は落とす."""
    return list(dict.fromkeys(key_to_id[norm(m)] for m in case.get("correct_model_ids") or []
                              if norm(m) in key_to_id))


def examples_text(cases: list, key_to_id: dict, version: str = PROMPT_VERSION) -> str:
    e = _LABELS[version][3]
    blocks = ["\n".join(_case_lines(c, version) + [f"{e}{', '.join(map(str, answer_ids(c, key_to_id)))}"])
              for c in cases]
    return f'<examples count="{len(cases)}">\n' + "\n\n".join(blocks) + "\n</examples>"


def task_text(case: dict, version: str = PROMPT_VERSION) -> str:
    final = FINAL_INSTRUCTION if version == "v1" else FINAL_V2   # v2・v3 は同じ一文
    return "<task>\n" + "\n".join(_case_lines(case, version)) + "\n</task>\n\n" + final


def build_params(model: str, effort: str, db_block: str, ex_block: str, task_block: str, *,
                 fmt: str = "eqc+miss", ttl: str = "5m", max_tokens: int = 16000,
                 version: str = PROMPT_VERSION) -> dict:
    """Messages API の引数。(a) 指示文（system）→ (b) DB → (c) 参考ケース → (d) テストケースの順.

    (b) と (c) の末尾にキャッシュの区切りを置く。(b) は全分割で、(c) は同じ分割の全テストケースで同じなので、
    2 件目以降は (a)〜(c) をキャッシュから読む。
    """
    cc = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": ttl}
    blocks = [{"type": "text", "text": db_block, "cache_control": dict(cc)}]
    if version != "v3":                       # 版 3 はゼロショット（参考ケースなし）
        blocks.append({"type": "text", "text": ex_block, "cache_control": dict(cc)})
    blocks.append({"type": "text", "text": task_block})
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt(fmt, version),
        "messages": [{"role": "user", "content": blocks}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }
    if model != "claude-sonnet-4-5":
        params["thinking"] = {"type": "adaptive", "display": "summarized"}
    return params


# ---- 応答の判定と指標 ------------------------------------------------------------------

def parse_output(text):
    """(ID の列 or None, 読めなかった理由 or None)。{"equation_ids": [...]} の形だけを読む."""
    try:
        obj = json.loads(text or "")
    except json.JSONDecodeError as e:
        return None, f"unreadable: {e}"
    if not isinstance(obj, dict) or not isinstance(obj.get("equation_ids"), list):
        return None, "unreadable: no equation_ids list"
    return obj["equation_ids"], None


def _as_id(x):
    """整数（bool は除く）または数字だけの文字列なら int。それ以外は None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


def judge_ids(raw, n_db: int, top_n: int = TOP_N) -> dict:
    """LLM が返した ID の列を、順位の列（DB にない ID は None）にする.

    同じ値の 2 回目以降は除く。DB にない ID（範囲外の整数・整数でないもの）は None としてその順位を占める。
    上位 top_n 件で打ち切る。読めない出力（raw が None）は空の列。
    """
    seen, ranked, dups = set(), [], 0
    for x in raw or []:
        v = _as_id(x)
        token = ("id", v) if v is not None else ("bad", json.dumps(x, sort_keys=True, ensure_ascii=False))
        if token in seen:
            dups += 1
            continue
        seen.add(token)
        ranked.append(v if v is not None and 1 <= v <= n_db else None)
    return {"ranked_ids": ranked[:top_n], "n_raw": len(raw or []), "n_duplicates": dups,
            "n_invalid": sum(1 for v in ranked[:top_n] if v is None),
            "n_truncated": max(0, len(ranked) - top_n)}


def ranks_in_list(ranked_ids: list, id_to_key: dict, correct_keys) -> list:
    """正解式（重複は 1 回）それぞれの順位。列にない正解式は MISS_RANK。None も 1 つの順位を占める."""
    pos = {}
    for r, i in enumerate(ranked_ids, 1):
        if i is not None and id_to_key.get(i) is not None:
            pos.setdefault(id_to_key[i], r)
    return [pos.get(k, MISS_RANK) for k in dict.fromkeys(correct_keys)]


def metrics_from_ranks(ranks: list, top_n: int = TOP_N) -> dict:
    """Recall@top_n と AP@top_n（標準の定義: AP = (1/K) Σ P(e)。上位 top_n 件の外の正解式は P(e) = 0）."""
    k = len(ranks)
    hits = sorted(r for r in ranks if r <= top_n)
    return {"K": k, "recall@20": len(hits) / k,
            "AP@20": sum(j / r for j, r in enumerate(hits, 1)) / k}


def ap_full(ranks: list) -> float:
    """並べた全体での AP（標準の定義）。8/31 報告の MAP。列にない正解式（MISS_RANK）は P(e) = 0."""
    found = sorted(r for r in ranks if r < MISS_RANK)
    return sum(j / r for j, r in enumerate(found, 1)) / len(ranks)


def _mean_sd(vals: list) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def summarize(recs: list) -> dict:
    """設定ごとに、分割（種）ごとのケース平均と、分割間の平均 ± 標準偏差（ddof=1）."""
    out = {}
    for setting in sorted({r["setting"] for r in recs}):
        per_seed = {}
        for seed in sorted({r["seed"] for r in recs if r["setting"] == setting}):
            rs = [r for r in recs if r["setting"] == setting and r["seed"] == seed]
            per_seed[str(seed)] = {"n": len(rs),
                                   "recall@20": float(np.mean([r["recall@20"] for r in rs])),
                                   "MAP@20": float(np.mean([r["AP@20"] for r in rs]))}
        out[setting] = {"n_seeds": len(per_seed), "per_seed": per_seed,
                        "recall@20": _mean_sd([v["recall@20"] for v in per_seed.values()]),
                        "MAP@20": _mean_sd([v["MAP@20"] for v in per_seed.values()])}
    return out


def shuffled_keys(keys, seed: int) -> list:
    """式の ID（eq_key）を、入力の順によらない固定の無作為な順に並べる（版 3 の DB の並び）."""
    out = sorted(keys)
    random.Random(seed).shuffle(out)
    return out


def split_test_records(per_case: list, seed: int) -> list:
    """per-case 出力のうち、その種のテストのケースの記録（並びは保つ）.

    種ごとのファイル（seed がない記録）も、全部の種を持つファイルも読む。検証用のケース（split "val"）は除く。
    """
    return [r for r in per_case if r.get("seed", seed) == seed and r.get("split", "test") == "test"]


def v3_plan(splits: dict, seeds, cases: dict) -> list:
    """版 3 で解かせるケース：seeds の分割の、設定 A と設定 B のテストケースの和集合（training_cases.json の順）.

    例題を渡さないので、同じケースへの入力はどの分割でも同じ。答えは、そのケースがテストに入るすべての分割で使う。
    """
    ids = set()
    for st in ("A", "B"):
        for sd in seeds:
            ids |= set(splits[st][str(sd)]["test"])
    return [cid for cid in cases if cid in ids]


def split_summary(per_case: dict, splits: dict, setting: str, seeds: list) -> dict:
    """版 3 の集計。per_case[case_id] = {"recall@20", "AP@20"}。各分割のテストケースの平均と、
    分割間の平均 ± 標準偏差（ddof=1）."""
    per_seed = {}
    for sd in seeds:
        ids = splits[setting][str(sd)]["test"]
        per_seed[str(sd)] = {"n": len(ids),
                             "recall@20": float(np.mean([per_case[c]["recall@20"] for c in ids])),
                             "MAP@20": float(np.mean([per_case[c]["AP@20"] for c in ids]))}
    return {"per_seed": per_seed,
            "recall@20": _mean_sd([v["recall@20"] for v in per_seed.values()]),
            "MAP@20": _mean_sd([v["MAP@20"] for v in per_seed.values()])}


def compare_splits(a: dict, b: dict) -> dict:
    """分割ごとの値（種 → 値）を対にした比較：差の平均と、a が上回った・同じ・下回った分割の数.

    検定はしない（本人の決定、2026-10-20）。結果は「10 通りの分割のうち何通りで上回ったか」で書く。
    """
    keys = sorted(set(a) & set(b), key=int)
    d = [a[k] - b[k] for k in keys]
    return {"n": len(d), "mean_diff": float(np.mean(d)), "wins": sum(x > 0 for x in d),
            "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d)}


def price_for(model: str, prompt_tokens: int) -> dict:
    """そのプロンプトの長さで使う料金（Haiku 5.5 は 100,000 トークンを超えると高い料金）."""
    p = PRICES[model]
    return p["long"] if "long_over" in p and prompt_tokens > p["long_over"] else p


def cost_usd(usage: dict, model: str, batch: bool = False) -> float:
    """1 回の呼び出しの料金（USD）。Haiku 5.5 はプロンプトが 100,000 トークンを超えると高い料金になる."""
    u = {k: usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                         "cache_read_input_tokens", "output_tokens")}
    p = price_for(model, u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"])
    cc = usage.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens") or 0
    w5m = u["cache_creation_input_tokens"] - w1h
    usd = (u["input_tokens"] * p["in"] + w5m * p["w5m"] + w1h * p["w1h"]
           + u["cache_read_input_tokens"] * p["read"] + u["output_tokens"] * p["out"]) / 1e6
    return usd * (0.5 if batch else 1.0)


def split_cost(p: dict, *, db: float, ex: float, task: float, n: int, out: float, first: bool,
               option: str, cand: float = 3000) -> float:
    """1 分割の費用（USD）の見積もり。プロンプトキャッシュ（5 分）を使い、キャッシュは毎回当たるとする.

    db: 指示文＋DB、ex: 参考ケース、task: テストケース 1 件（キャッシュしない）、out: 1 回の出力、n: テストケースの数。
    first: この分割で DB を書き込むか（最初の分割だけ。以後は DB のキャッシュを読む）。
    option "single": 1 ケース 1 回（案 4、案 2 は ex を収まる分にして渡す）。
    option "halves": DB を半分ずつ 2 回（毎回、全参考ケース）＋候補 cand トークンを付けて選び直す 1 回（案 1）。
    """
    if option == "single":
        per = (db + ex) * p["read"] + task * p["in"] + out * p["out"]
        write = (db if first else 0) + ex
    elif option == "halves":
        per = (2 * (db / 2 + ex) + ex) * p["read"] + (3 * task + cand) * p["in"] + 3 * out * p["out"]
        write = (db if first else 0) + 3 * ex
    else:
        raise ValueError(option)
    return (n * per + write * p["w5m"]) / 1e6


# ---- 記録と再開 ------------------------------------------------------------------

CFG_KEYS = ("model", "effort", "db_format", "examples", "prompt_version")


def cfg_name(cfg: dict) -> str:
    return (f"{cfg['model']}__{cfg['db_format']}__ex-{cfg['examples']}__{cfg['effort']}"
            f"__{cfg['prompt_version']}")


def pending_cases(recs: list, plan: list, cfg: dict, fields=("setting", "seed", "case_id")) -> list:
    """plan（fields の値の組の列。版 1・2 は (設定, 種, case_id)、版 3 は (回, case_id)）のうち、同じ条件
    （モデル・effort・DB の形式・参考ケース・プロンプト版）で API が応答を返した記録がまだないもの。
    API の失敗（error あり）は解き直す."""
    done = {tuple(r.get(f) for f in fields) for r in recs
            if all(r.get(k) == cfg[k] for k in CFG_KEYS) and r.get("error") is None}
    return [p for p in plan if p not in done]


def load_raw(cfg: dict) -> list:
    f = RAW_DIR / f"{cfg_name(cfg)}.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


_RAW_LOCK = threading.Lock()


def append_raw(cfg: dict, rec: dict) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with _RAW_LOCK, open(RAW_DIR / f"{cfg_name(cfg)}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---- データの読み込み（固定した分割と ID の対応を使う） ------------------------------------------

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_db() -> tuple:
    """(データベースの式の列, 短い ID → eq_key, eq_key → 短い ID)。db_ids.json と食い違えば止める."""
    eqs = json.loads(EQ_JSON.read_text(encoding="utf-8"))
    eqs = eqs if isinstance(eqs, list) else eqs.get("equations", [])
    eqs = [e for e in eqs if eq_key(e)]
    fixed = json.loads(DB_IDS_JSON.read_text(encoding="utf-8"))
    keys = [eq_key(e) for e in eqs]
    if keys != fixed["keys"]:
        sys.exit("unified_equations.json の式の並びが db_ids.json と食い違う（データベースが変わった）")
    id_to_key = {i: k for i, k in enumerate(keys, 1)}
    return eqs, id_to_key, {k: i for i, k in id_to_key.items()}


def load_case_index() -> dict:
    return {c["case_id"]: c for c in json.loads(CASES_JSON.read_text(encoding="utf-8"))}


def load_splits() -> dict:
    return json.loads(SPLITS_JSON.read_text(encoding="utf-8"))["splits"]


def ref_cases(split: dict, cases: dict, max_examples: int | None = None, seed: int = 0) -> list:
    """参考ケース＝訓練ケースと検証ケース（training_cases.json の順）。テストケースは含めない.

    max_examples を与えると、その件数だけ無作為に選ぶ（seed で固定。入力の上限に収めるための案）。
    """
    ids = set(split["train"]) | set(split["val"])
    if max_examples is not None and max_examples < len(ids):
        ids = set(random.Random(seed).sample(sorted(ids), max_examples))
    return [c for cid, c in cases.items() if cid in ids]


# ---- splits：分割を固定する ------------------------------------------------------------

def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cmd_splits(args) -> None:
    if SPLITS_JSON.exists() and not args.force:
        sys.exit(f"{SPLITS_JSON} はもうある（固定済み）。作り直すなら --force")
    # パイプラインのコードを使うのはここだけ（この時点の版で分割を作り、JSON に固定する）
    sys.path.insert(0, str(ROOT))
    from set_aware_reranker import stratified_src_split
    from two_stage_query_conditioned import (load_equations, load_cases, case_src, in_vars, out_vars,
                                             eq_key as pipe_eq_key, norm as pipe_norm)

    eqs = load_equations()
    ek = [pipe_eq_key(e) for e in eqs if pipe_eq_key(e)]
    if len(set(ek)) != len(ek):
        sys.exit("eq_key が重複している")
    ki = {k: i for i, k in enumerate(ek)}
    all_cases = load_cases()
    excluded = set()
    if args.exclude:   # 評価と訓練から外すケース（外してから層化分割する。set_aware_reranker の --exclude-cases と同じ）
        ex = json.loads(Path(args.exclude).read_text(encoding="utf-8"))
        excluded = set(ex["excluded_case_ids"] if isinstance(ex, dict) else ex)
    checks = {}
    for spec in args.check or []:
        setting, pattern = spec.split("=", 1)
        checks.setdefault(setting, []).append(pattern)
    out = {"splits": {}, "n_cases": {}, "check": {}}
    for setting, wanted in SETTINGS.items():
        cases = [c for c in all_cases if variant_matches(c.get("variant_type", ""), wanted)
                 and c["case_id"] not in excluded]
        if len({c["case_id"] for c in cases}) != len(cases):
            sys.exit(f"設定 {setting} に同じ case_id がある")
        # set_aware_reranker.run_mode（split_mode="stratified"）と同じ入力
        cl = [[ki[pipe_norm(m)] for m in (c.get("correct_model_ids") or []) if pipe_norm(m) in ki] for c in cases]
        cs = [case_src(c) for c in cases]
        feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
                  len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
        out["n_cases"][setting] = len(cases)
        out["splits"][setting], out["check"][setting] = {}, {}
        for seed in ALL_SEEDS:
            sp = stratified_src_split(cs, feats, seed)
            parts = {part: [cases[i]["case_id"] for i, s in enumerate(cs) if s in sp[part] and cl[i]]
                     for part in ("train", "val", "test")}
            out["splits"][setting][str(seed)] = parts
            out["check"][setting][str(seed)] = (_check_patterns(checks.get(setting, []), seed, parts["test"])
                                                if args.check else _check_against_saved(setting, seed, parts["test"]))
            print(f"{setting} seed={seed}: train {len(parts['train'])} val {len(parts['val'])} "
                  f"test {len(parts['test'])}  check={out['check'][setting][str(seed)]}")
    files = ["set_aware_reranker.py", "two_stage_query_conditioned.py"]
    out["provenance"] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD"),
        "code_blob": {f: _git("hash-object", f) for f in files},
        "code_differs_from_head": {f: bool(_git("diff", "--name-only", "HEAD", "--", f)) for f in files},
        "data_sha256": {p.name: sha256(p) for p in (CASES_JSON, EQ_JSON)},
        "rule": "set_aware_reranker.run_mode(split_mode='stratified') と同じ特徴量タプル（正解式数・入力変数数・"
                "出力変数数・正解式の出典数）で stratified_src_split を呼んだ。train/val/test は正解式が DB にある"
                "ケースだけ（run_mode の tr/te と同じ）。",
        "variants": SETTINGS,
        "exclude_file": args.exclude, "exclude_sha256": sha256(Path(args.exclude)) if args.exclude else None,
        "excluded_case_ids": sorted(excluded),
        "checked_against": checks,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    DB_IDS_JSON.write_text(json.dumps({
        "note": "短い ID i（1 始まり）は keys[i-1] の式。unified_equations.json の並び順。",
        "unified_equations_sha256": sha256(EQ_JSON), "n": len(ek), "keys": ek,
    }, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {SPLITS_JSON} and {DB_IDS_JSON}")


def _check_patterns(patterns: list, seed: int, test: list) -> dict:
    """--check で与えた per-case 出力（パターンの {seed} を種に置き換えたファイル）のテストの並びと比べる.

    ファイルが種ごとでも、全部の種を含む 1 つのファイルでも読む（per_case の seed で絞る）。ファイルがなければ記録しない。
    """
    res = {}
    for pat in patterns:
        f = ROOT / pat.format(seed=seed)
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))["results"]
        res[pat] = [x["case_id"] for x in split_test_records(next(iter(r.values()))["per_case"], seed)] == test
    return res


def _check_against_saved(setting: str, seed: int, test: list) -> dict:
    """既存の per-case 出力のテストケースの並びと比べる（ファイルがあるものだけ）."""
    files = {"proposed_w30-70": EXP / ("xs1" if setting == "A" else "xs1_B") / f"reranker-10S_w30-70__{seed}.json",
             "baseline_w40-60": EXP / "xk50" / f"baseline40_{setting}__{seed}.json"}
    res = {}
    for name, f in files.items():
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))["results"]
            res[name] = [x["case_id"] for x in next(iter(r.values()))["per_case"]] == test
    return res


# ---- API ----------------------------------------------------------------------

def client_(max_retries: int = 3):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    import anthropic
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=max_retries,
                               timeout=900.0)


def message_record(msg, headers: dict | None) -> dict:
    text = "".join(b.text for b in msg.content if b.type == "text")
    thinking = "".join(getattr(b, "thinking", "") or "" for b in msg.content if b.type == "thinking")
    return {"model_served": msg.model, "stop_reason": msg.stop_reason, "usage": msg.usage.model_dump(),
            "text": text, "thinking": thinking, "request_id": getattr(msg, "_request_id", None),
            "ratelimit": headers}


def _ratelimit_headers(headers) -> dict:
    return {k: v for k, v in (headers or {}).items() if k.startswith("anthropic-ratelimit") or k == "request-id"}


def call_api(client, params: dict, attempts: int = 4, stream: bool = False) -> tuple:
    """(記録用の dict or None, エラー文字列 or None)。429・5xx・接続エラーは待って繰り返す.

    stream=True はストリーミングで受ける（max_tokens が大きいと、ストリーミングでない要求は SDK が止めるため）。
    """
    import anthropic
    last = None
    for i in range(attempts):
        try:
            if stream:
                with client.messages.stream(**params) as st:
                    msg = st.get_final_message()
                    headers = _ratelimit_headers(getattr(st.response, "headers", {}))
                return message_record(msg, headers), None
            raw = client.messages.with_raw_response.create(**params)
            return message_record(raw.parse(), _ratelimit_headers(raw.headers)), None
        except anthropic.RateLimitError as e:
            last = e
            time.sleep(30 * (i + 1))
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last = e
                time.sleep(10 * (i + 1))
                continue
            return None, f"api {e.status_code}: {e.message}"
        except anthropic.APIConnectionError as e:
            last = e
            time.sleep(10 * (i + 1))
    return None, f"api gave up: {last}"


def score_record(rec: dict, id_to_key: dict, correct_keys: list) -> dict:
    ids, err = parse_output(rec.get("text")) if rec.get("error") is None else (None, rec["error"])
    j = judge_ids(ids, n_db=len(id_to_key))
    ranks = ranks_in_list(j["ranked_ids"], id_to_key, correct_keys)
    return {**metrics_from_ranks(ranks), "ranks": sorted(ranks), "parse_error": err,
            "ranked_ids": j["ranked_ids"], "ranked_keys": [id_to_key.get(i) for i in j["ranked_ids"]],
            **{k: j[k] for k in ("n_raw", "n_duplicates", "n_invalid", "n_truncated")}}


def _cfg(args) -> dict:
    return {"model": args.model, "effort": args.effort, "db_format": args.db_format,
            "examples": "all" if args.max_examples is None else str(args.max_examples),
            "prompt_version": args.prompt_version}


def _plan(splits: dict, settings: list, seeds: list) -> list:
    return [(s, seed, cid) for s in settings for seed in seeds for cid in splits[s][str(seed)]["test"]]


def cmd_count(args) -> None:
    """count_tokens で 1 回分の入力トークン数を数え、DB 全体と全参考ケースが 1 回に収まるかを調べる（無料）.

    モデルごとに、(指示文＋DB)・参考ケース（全分割）・最も長いテストケースを数え、
    最も大きい分割で 1 回分の要求をそのまま数えて、部分の和との差（出力の形の指定などの分）を足す。
    """
    eqs, id_to_key, key_to_id = load_db()
    cases, splits = load_case_index(), load_splits()
    client = client_()
    v, seeds, settings = args.prompt_version, [int(x) for x in args.seeds.split(",")], args.settings.split(",")
    out = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prompt_version": v,
           "limit": 1_000_000, "reserve_for_output": args.reserve, "models": {}}

    def count(**kw):
        return client.messages.count_tokens(**kw).input_tokens

    def user(blocks):
        return [{"role": "user", "content": [{"type": "text", "text": b} for b in blocks]}]

    ex_text = {(st, sd): examples_text(ref_cases(splits[st][str(sd)], cases), key_to_id, v)
               for st in settings for sd in seeds}
    longest = {st: max((c for sd in seeds for c in splits[st][str(sd)]["test"]),
                       key=lambda c: len(task_text(cases[c], v))) for st in settings}
    db = {fmt: db_text(eqs, fmt, v) for fmt in args.formats.split(",")}
    for model in args.models.split(","):
        sys_p = system_prompt(args.formats.split(",")[0], v)
        base = count(model=model, system=sys_p, messages=user(["x"]))
        row = {"system_db": {}, "examples": {}, "n_examples": {}, "task_max": {}, "total": {}}
        for fmt, text in db.items():
            row["system_db"][fmt] = count(model=model, system=system_prompt(fmt, v), messages=user([text]))
        for (st, sd), ex in ex_text.items():
            row["examples"][f"{st}{sd}"] = count(model=model, system=sys_p, messages=user(["x", ex])) - base
            row["n_examples"][f"{st}{sd}"] = ex.count("\n\n") + 1
        for st in settings:
            row["task_max"][st] = count(model=model, system=sys_p,
                                        messages=user(["x", task_text(cases[longest[st]], v)])) - base
        # 部分の和と、1 回分の要求をそのまま数えた値の差（出力の形の指定・思考の設定・ブロックの区切りの分）
        fmt0 = args.formats.split(",")[0]
        st0, sd0 = max(ex_text, key=lambda k: row["examples"][f"{k[0]}{k[1]}"])
        p = build_params(model, "high", db[fmt0], ex_text[(st0, sd0)], task_text(cases[longest[st0]], v),
                         fmt=fmt0, version=v)
        full = count(**{k: p[k] for k in ("model", "system", "messages", "output_config", "thinking") if k in p})
        row["overhead"] = full - (row["system_db"][fmt0] + row["examples"][f"{st0}{sd0}"] + row["task_max"][st0])
        row["calibration"] = {"setting": st0, "seed": sd0, "format": fmt0, "full_request": full}
        for fmt in db:
            row["total"][fmt] = {f"{st}{sd}": row["system_db"][fmt] + row["examples"][f"{st}{sd}"]
                                 + row["task_max"][st] + row["overhead"] for st in settings for sd in seeds}
        out["models"][model] = row
        worst = {fmt: max(t.values()) for fmt, t in row["total"].items()}
        print(model, json.dumps({"system_db": row["system_db"], "worst_total": worst,
                                 "room_after_reserve": {f: 1_000_000 - args.reserve - w for f, w in worst.items()}}),
              flush=True)
        (OUT_DIR / f"token_counts_{v}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_costs(args) -> None:
    """token_counts_<版>.json から、案ごと・モデルごとの費用と、1 回に収まるかを計算する（API は呼ばない）."""
    tc = json.loads((OUT_DIR / f"token_counts_{args.prompt_version}.json").read_text(encoding="utf-8"))
    splits = load_splits()
    limit, reserve, fmt = tc["limit"], args.reserve, args.db_format
    seeds_by = {"3": [42, 123, 456], "10": ALL_SEEDS}
    out = {"prompt_version": args.prompt_version, "db_format": fmt, "reserve_for_output": reserve,
           "out_tokens_per_call": args.out_tokens, "models": {}}
    for model, row in tc["models"].items():
        db, oh = row["system_db"][fmt], row["overhead"]
        p = price_for(model, db)                      # どの案でもプロンプトは 10 万トークンを超える
        fit, cost = {}, {}
        for st in ("A", "B"):
            for sd in ALL_SEEDS:
                key = f"{st}{sd}"
                ex, task = row["examples"][key], row["task_max"][st]
                fit[key] = {"single_total": db + ex + task + oh,
                            "single_room": limit - (db + ex + task + oh),
                            "halves_total": round(db / 2) + ex + task + oh,
                            "subset_examples_tokens": min(ex, limit - reserve - db - task - oh),
                            "subset_share": min(1.0, (limit - reserve - db - task - oh) / ex)}
        for o in args.out_tokens:
            for k, seeds in seeds_by.items():
                for option in ("single", "subset", "halves"):
                    total, first = 0.0, True
                    for st in ("A", "B"):
                        for sd in seeds:
                            key = f"{st}{sd}"
                            ex = row["examples"][key] if option != "subset" else fit[key]["subset_examples_tokens"]
                            total += split_cost(p, db=db, ex=ex, task=row["task_max"][st],
                                                n=len(splits[st][str(sd)]["test"]), out=o, first=first,
                                                option="halves" if option == "halves" else "single")
                            first = False
                    cost[f"{option}_{k}splits_out{o}"] = round(total)
        worst = max(v["single_total"] for v in fit.values())
        out["models"][model] = {
            "system_db": db, "overhead": oh, "fit": fit, "cost_usd": cost,
            "single_worst_total": worst, "single_worst_room": limit - worst,
            "single_fits_with_reserve": worst + reserve <= limit,
            "halves_worst_total": max(v["halves_total"] for v in fit.values()),
            "subset_share_min": min(v["subset_share"] for v in fit.values()),
        }
        print(model, json.dumps({k: v for k, v in out["models"][model].items() if k != "fit"}), flush=True)
    (OUT_DIR / f"costs_{args.prompt_version}_{fmt}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_check_compress(args) -> None:
    """式の表記の短縮で、変数記号の現れ方が変わらないことを全式で確かめ、結果を JSON に残す.

    確かめること: (1) DB の全変数記号（13,220 種。評価ケースの入出力変数はすべてこの中にある）のうち、
    各式に記号として現れるものの集合が、短縮の前後で同じ。(2) 各式の変数記号が文字列として現れるか。
    (3) 評価ケースの入出力変数の記号が、正解式に記号として現れるか。
    """
    eqs, id_to_key, key_to_id = load_db()
    cases = load_case_index()
    index = KeyIndex(k for e in eqs for k in (e.get("variables") or {}))
    before, after = {}, {}
    n_changed = n_real = n_sub = 0
    chars0 = chars1 = 0
    bad_real, bad_sub = [], []
    for i, e in enumerate(eqs, 1):
        own = [norm(k) for k in (e.get("variables") or {}) if norm(k)]
        a, b = one_line(e.get("equation")), compress_latex(e.get("equation"), own, index)
        k = id_to_key[i]
        before[k], after[k] = a, b
        chars0, chars1, n_changed = chars0 + len(a), chars1 + len(b), n_changed + (a != b)
        if index.real(a) != index.real(b):
            n_real += 1
            bad_real.append(k)
        if {x for x in own if x in a} != {x for x in own if x in b}:
            n_sub += 1
            bad_sub.append(k)
    n_pairs = n_case_real = n_case_sub = 0
    case_syms = set()
    for c in cases.values():
        if not any(variant_matches(c.get("variant_type", ""), w) for w in SETTINGS.values()):
            continue
        syms = var_list(c.get("input_variables")) + var_list(c.get("output_variables"))
        case_syms.update(syms)
        for m in c.get("correct_model_ids") or []:
            k = norm(m)
            if k not in key_to_id:
                continue
            ra, rb = index.real(before[k]), index.real(after[k])
            for sym in syms:
                n_pairs += 1
                n_case_real += (sym in ra) != (sym in rb)
                n_case_sub += (sym in before[k]) != (sym in after[k])
    allkeys = set(index.one) | {k for ks in index.by2.values() for k in ks}
    res = {
        "n_equations": len(eqs), "n_changed": n_changed, "chars_before": chars0, "chars_after": chars1,
        "db_symbols": len(allkeys), "case_symbols": len(case_syms),
        "case_symbols_not_in_db_symbols": len(case_syms - allkeys),
        "equations_real_symbol_set_changed": n_real, "examples_real": bad_real[:20],
        "case_symbol_pairs": n_pairs, "case_symbol_pairs_real_changed": n_case_real,
        "equations_own_symbol_substring_set_changed": n_sub, "examples_substring": bad_sub[:20],
        "case_symbol_pairs_substring_changed": n_case_sub,
        "note": "real: 記号として現れる（字句の切れ目で始まり切れ目で終わる。\\left の綴りの中の t は数えない）。"
                "substring: 文字列として含まれる（\\frac の f のような命令の中の文字も数える）。"
                "substring の変化は、命令を除いたために見かけの一致が消えた分。",
    }
    (OUT_DIR / "compress_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


def _run_cases(client, cfg: dict, todo: list, params_of, source: str, workers: int,
               budget: float | None, spent: list, attempts: int = 4, meta_of=None,
               fields=("setting", "seed", "case_id"), group_of=lambda t: (t[0], t[1]), stream: bool = False) -> None:
    """todo を、キャッシュの前半が同じまとまり（group_of）ごとに、最初の 1 件は単独で（キャッシュを書く）、
    残りは workers 本の並列で解く。spent[0] がこの条件の費用の累計で、budget に達したら止める."""
    by_split = {}
    for t in todo:
        by_split.setdefault(group_of(t), []).append(t)

    def one(t):
        if budget is not None and spent[0] >= budget:
            return
        t0 = time.time()
        rec, err = call_api(client, params_of(t), attempts=attempts, stream=stream)
        r = {**cfg, **dict(zip(fields, t)), "source": source,
             "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
             "error": err, **(meta_of(t) if meta_of else {}), **(rec or {})}
        append_raw(cfg, r)
        if rec:
            with _RAW_LOCK:
                spent[0] += cost_usd(rec["usage"], cfg["model"])
        u = (rec or {}).get("usage") or {}
        print(f"  {'/'.join(map(str, t))}: err={err} stop={(rec or {}).get('stop_reason')} "
              f"in={u.get('input_tokens')} write={u.get('cache_creation_input_tokens')} "
              f"read={u.get('cache_read_input_tokens')} out={u.get('output_tokens')} "
              f"total=${spent[0]:.2f}", flush=True)

    for key, items in by_split.items():
        one(items[0])
        if len(items) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(one, items[1:]))
        if budget is not None and spent[0] >= budget:
            print(f"予算 ${budget:.2f} に達したので止める（${spent[0]:.2f}）。再開は同じコマンドで")
            return


def _params_factory(args, eqs, key_to_id, cases, splits):
    """(params_of, meta_of)。params_of((設定, 種, case_id)) が API の引数、meta_of が記録に足す情報."""
    v = args.prompt_version
    db_block = db_text(eqs, args.db_format, v)
    ex_cache, n_refs = {}, {}

    def _ex(setting, seed):
        if (setting, seed) not in ex_cache:
            refs = ref_cases(splits[setting][str(seed)], cases, args.max_examples, seed)
            ex_cache[(setting, seed)] = examples_text(refs, key_to_id, v)
            n_refs[(setting, seed)] = len(refs)
        return ex_cache[(setting, seed)]

    def params_of(t):
        setting, seed, cid = t
        return build_params(args.model, args.effort, db_block, _ex(setting, seed), task_text(cases[cid], v),
                            fmt=args.db_format, ttl=args.ttl, max_tokens=args.max_tokens, version=v)

    def meta_of(t):
        _ex(t[0], t[1])
        return {"n_examples": n_refs[(t[0], t[1])], "ttl": args.ttl, "max_tokens": args.max_tokens}
    return params_of, meta_of


def cmd_pilot(args) -> None:
    eqs, id_to_key, key_to_id = load_db()
    cases, splits = load_case_index(), load_splits()
    cfg = _cfg(args)
    plan = _plan(splits, [args.setting], [args.seed])[: args.n]
    todo = pending_cases(load_raw(cfg), plan, cfg)
    params_of, meta_of = _params_factory(args, eqs, key_to_id, cases, splits)
    client, spent = client_(max_retries=0), [0.0]   # 試しは繰り返さない（429 ならそのまま記録して止める）
    if todo:
        p = params_of(todo[0])
        n_in = client.messages.count_tokens(**{k: p[k] for k in ("model", "system", "messages", "output_config")
                                               if k in p}, **({"thinking": p["thinking"]} if "thinking" in p else {})
                                            ).input_tokens
        print(f"1 件目の入力 {n_in:,} トークン（max_tokens {args.max_tokens:,}、上限 {CONTEXT_LIMIT[args.model]:,}）")
        if n_in + args.max_tokens > CONTEXT_LIMIT[args.model] and not args.force:
            sys.exit("入力の上限を超えるので送らない（--force で送る）")
    _run_cases(client, cfg, todo, params_of, "pilot", 1, None, spent, attempts=1, meta_of=meta_of)
    for r in load_raw(cfg):
        if (r["setting"], r["seed"], r["case_id"]) in plan:
            s = score_record(r, id_to_key, [norm(m) for m in cases[r["case_id"]]["correct_model_ids"]
                                            if norm(m) in key_to_id])
            print(f"{r['case_id']}: served={r.get('model_served')} stop={r.get('stop_reason')} "
                  f"cost=${cost_usd(r['usage'], cfg['model']) if r.get('usage') else 0:.3f} "
                  f"ids={s['ranked_ids']} K={s['K']} R@20={s['recall@20']:.3f} AP@20={s['AP@20']:.3f} "
                  f"invalid={s['n_invalid']} dup={s['n_duplicates']} parse_error={s['parse_error']}")
            print(f"   ratelimit={r.get('ratelimit')}")


def cmd_run(args) -> None:
    eqs, id_to_key, key_to_id = load_db()
    cases, splits = load_case_index(), load_splits()
    cfg = _cfg(args)
    plan = _plan(splits, args.settings.split(","), [int(s) for s in args.seeds.split(",")])
    todo = pending_cases(load_raw(cfg), plan, cfg)
    print(f"{cfg_name(cfg)}: plan {len(plan)}, todo {len(todo)}")
    params_of, meta_of = _params_factory(args, eqs, key_to_id, cases, splits)
    client, spent = client_(), [0.0]
    _run_cases(client, cfg, todo, params_of, "run", args.workers, args.budget_usd, spent, meta_of=meta_of)


# ---- evaluate：判定・指標・既存の手法との比較 ----------------------------------------------

REF_RUN = EXP / "kato1021"   # 比べる相手の出力：第 1 段のやり直しのセッションの最終の実行（commit 8b77e73。92 件を外し、説明文 113 件を書き直したデータ）


def _selected_ref(setting: str, seed: int) -> dict:
    """第 1 段のやり直しのセッションが検証用のケースで選んだ、提案手法の設定（重み・不正解の選び方）."""
    return json.loads((REF_RUN / "selected.json").read_text(encoding="utf-8"))["selected"][setting][str(seed)]


# 比べる相手：同じ分割（外す一覧 v3 の 92 件を外したもの）での最終の設定。値は experiments/kato1021/main.json と一致する。
# 出力は REF_RUN/raw/（非追跡）。ranks は各正解式の順位。並びが splits.json と違えば reference_rows が外す。
REFERENCE = {
    "proposed (N1 = 2,000, (0.3, 0.7), validation-selected negatives)": lambda s, seed: (
        REF_RUN / "raw" / "select"
        / f"{s}_{_selected_ref(s, seed)['w']}_{_selected_ref(s, seed)['neg']}__{seed}.json", "reranker-10S"),
    "baseline TF-IDF (validation-selected weights)": lambda s, seed: (
        REF_RUN / "raw" / f"baselines_percase_{s}.json", "baseline"),
}


def reference_rows(settings: list, seeds: list, splits: dict) -> dict:
    """既存の手法の、同じ分割のケースごとの値。ある手法・設定で、どれかの分割の出力がないか、
    テストケースの並びが splits.json と違うときは、その手法・設定を外して知らせる（古い分割の出力を混ぜないため）."""
    out, cache = {}, {}
    for name, where in REFERENCE.items():
        recs = []
        for s in settings:
            rows, reason = [], None
            for seed in seeds:
                try:
                    f, mode = where(s, seed)
                except (FileNotFoundError, KeyError) as e:   # 選んだ設定の一覧（selected.json）がまだない など
                    reason = f"出力の場所が決まらない（{e}）"
                    break
                if not f.exists():
                    reason = f"{f} がない"
                    break
                if f not in cache:
                    cache[f] = json.loads(f.read_text(encoding="utf-8"))["results"]
                pcs = split_test_records(cache[f][mode]["per_case"], seed)
                if [r["case_id"] for r in pcs] != splits[s][str(seed)]["test"]:
                    reason = f"{f} の種 {seed} のテストケースが splits.json と違う"
                    break
                rows += [{"setting": s, "seed": seed, "case_id": r["case_id"],
                          **metrics_from_ranks(r["ranks"]), "AP_full": ap_full(r["ranks"])} for r in pcs]
            if reason:
                print(f"  比べる相手から外す: {name} 設定 {s}（{reason}）", file=sys.stderr)
            else:
                recs += rows
        if recs:
            summ = summarize(recs)
            for s in summ:
                summ[s]["MAP_full"] = _mean_sd([float(np.mean([r["AP_full"] for r in recs
                                                               if r["setting"] == s and r["seed"] == seed]))
                                                for seed in seeds if any(r["seed"] == seed and r["setting"] == s
                                                                         for r in recs)])
            out[name] = {"summary": summ, "per_case": recs}
    return out


def cmd_evaluate(args) -> None:
    eqs, id_to_key, key_to_id = load_db()
    cases, splits = load_case_index(), load_splits()
    cfg = _cfg(args)
    settings, seeds = args.settings.split(","), [int(s) for s in args.seeds.split(",")]
    plan = _plan(splits, settings, seeds)
    latest = {}
    for r in load_raw(cfg):
        if r.get("error") is None:
            latest[(r["setting"], r["seed"], r["case_id"])] = r
    per_case, missing, served, stops, cost = [], [], Counter(), Counter(), 0.0
    for t in plan:
        r = latest.get(t)
        if r is None:
            missing.append(t)
            continue
        correct = [norm(m) for m in cases[t[2]]["correct_model_ids"] if norm(m) in key_to_id]
        s = score_record(r, id_to_key, correct)
        per_case.append({"setting": t[0], "seed": t[1], "case_id": t[2],
                         "variant": cases[t[2]].get("variant_type"), "stop_reason": r.get("stop_reason"),
                         "model_served": r.get("model_served"), "raw_text": r.get("text"), **s})
        served[r.get("model_served")] += 1
        stops[r.get("stop_reason")] += 1
    for r in load_raw(cfg):
        if r.get("usage"):
            cost += cost_usd(r["usage"], cfg["model"])
    n = len(per_case)
    quality = {
        "n_scored": n, "n_missing": len(missing), "missing": missing[:50],
        "model_served": dict(served), "stop_reason": dict(stops),
        "unreadable": sum(1 for p in per_case if p["parse_error"]),
        "cases_with_invalid_id": sum(1 for p in per_case if p["n_invalid"] > 0),
        "invalid_id_share": (sum(p["n_invalid"] for p in per_case)
                             / max(1, sum(len(p["ranked_ids"]) for p in per_case))),
        "cases_with_duplicates": sum(1 for p in per_case if p["n_duplicates"] > 0),
        "cases_over_20": sum(1 for p in per_case if p["n_truncated"] > 0),
        "n_listed": dict(Counter(len(p["ranked_ids"]) for p in per_case)),
        "cases_with_20": sum(1 for p in per_case if len(p["ranked_ids"]) == TOP_N),
        "cost_usd_all_calls": round(cost, 2),
    }
    out = {"config": cfg, "settings": settings, "seeds": seeds,
           "rules": "ID だけで判定（等価性の判定なし）。DB にない ID と読めない出力は、その順位を占める不正解。"
                    "同じ ID の 2 回目以降は除く。上位 20 件。MAP@20 は標準の定義（上位 20 件の外の正解式は P(e)=0）。",
           "quality": quality, "summary": summarize(per_case) if per_case else {},
           "reference": {k: v["summary"] for k, v in reference_rows(settings, seeds, splits).items()},
           "per_case": per_case}
    f = OUT_DIR / f"results_{cfg_name(cfg)}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"quality": quality, "summary": out["summary"]}, ensure_ascii=False, indent=1))
    for name, summ in out["reference"].items():
        for s, v in summ.items():
            print(f"  ref {name} {s}: R@20 {v['recall@20']['mean']:.3f}±{v['recall@20']['sd']:.3f} "
                  f"MAP@20 {v['MAP@20']['mean']:.3f}±{v['MAP@20']['sd']:.3f} "
                  f"MAP(full) {v['MAP_full']['mean']:.3f}")
    print(f"wrote {f}")


# ---- 版 3（ゼロショット、全ケース × 3 回）のコマンド ------------------------------------------------

V3_MAX_TOKENS = 32000   # 思考を含む出力の上限。入力（約 95 万）と合わせて 100 万に収まる。大きいのでストリーミングで受ける


def load_db_v3() -> tuple:
    """版 3 の並び（db_ids_v3.json）の (式の列, 短い ID → eq_key, eq_key → 短い ID)."""
    eqs, id_to_key0, _ = load_db()
    by_key = {id_to_key0[i]: eqs[i - 1] for i in id_to_key0}
    fixed = json.loads(DB_IDS_V3_JSON.read_text(encoding="utf-8"))
    order = fixed["keys"]
    if order != shuffled_keys(by_key, fixed["seed"]):
        sys.exit("db_ids_v3.json の並びが、式の ID と乱数の種から作り直した並びと食い違う")
    id_to_key = {i: k for i, k in enumerate(order, 1)}
    return [by_key[k] for k in order], id_to_key, {k: i for i, k in id_to_key.items()}


def v3_case_ids(cases: dict, splits: dict) -> list:
    """設定 A の全ケース（1,823 件。設定 B の 1,000 件を含む）を training_cases.json の順に."""
    sp = splits["A"]["42"]
    ids = set(sp["train"]) | set(sp["val"]) | set(sp["test"])
    return [cid for cid in cases if cid in ids]


def _cfg3(args) -> dict:
    return {"model": args.model, "effort": args.effort, "db_format": "eqd", "examples": "none",
            "prompt_version": "v3"}


def _params3(args, eqs, cases):
    db_block = db_text(eqs, "eqd", "v3")

    def params_of(t):
        return build_params(args.model, args.effort, db_block, None, task_text(cases[t[0]], "v3"),
                            fmt="eqd", ttl=args.ttl, max_tokens=args.max_tokens, version="v3")

    cases_sha = sha256(CASES_JSON)   # 説明文は training_cases.json の context をそのまま使う。どの版で解いたかを記録する

    def meta_of(t):
        return {"ttl": args.ttl, "max_tokens": args.max_tokens, "order_seed": V3_ORDER_SEED,
                "training_cases_sha256": cases_sha,
                "task_sha256": hashlib.sha256(task_text(cases[t[0]], "v3").encode("utf-8")).hexdigest()}
    return params_of, meta_of


def _spent_so_far(cfg: dict) -> float:
    return sum(cost_usd(r["usage"], cfg["model"]) for r in load_raw(cfg) if r.get("usage"))


def _seeds(text: str) -> list:
    return ALL_SEEDS if text == "all" else [int(x) for x in text.split(",")]


def cmd_fix_order(args) -> None:
    if DB_IDS_V3_JSON.exists() and not args.force:
        sys.exit(f"{DB_IDS_V3_JSON} はもうある（固定済み）。作り直すなら --force")
    _, id_to_key, _ = load_db()
    keys = shuffled_keys(id_to_key.values(), V3_ORDER_SEED)
    DB_IDS_V3_JSON.write_text(json.dumps({
        "note": "版 3 の短い ID i（1 始まり）は keys[i-1] の式。式の ID を並べ替えてから、乱数の種 seed で無作為に並べた。",
        "seed": V3_ORDER_SEED, "unified_equations_sha256": sha256(EQ_JSON), "n": len(keys), "keys": keys,
    }, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"wrote {DB_IDS_V3_JSON}: {len(keys)} equations, seed {V3_ORDER_SEED}")


def cmd_count3(args) -> None:
    """版 3 の 1 回分のトークン数（最も長いテストケースで要求そのものを数える）と、費用の見積もり（無料）.

    呼び出しは、異なるテストケースごとに 1 回（10 通りの分割と 3 通りの分割の 2 つの場合）。
    """
    eqs, _, _ = load_db_v3()
    cases, splits = load_case_index(), load_splits()
    plans = {"10splits": v3_plan(splits, ALL_SEEDS, cases), "3splits": v3_plan(splits, [42, 123, 456], cases)}
    client = client_()
    db_block = db_text(eqs, "eqd", "v3")
    longest = max(plans["10splits"], key=lambda c: len(task_text(cases[c], "v3")))
    chars = [len(task_text(cases[c], "v3")) for c in plans["10splits"]]
    out = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "prompt_version": "v3",
           "n_calls": {k: len(v) for k, v in plans.items()}, "max_tokens": args.max_tokens, "limit": 1_000_000,
           "out_tokens_assumed": args.out_tokens, "models": {}}
    for model in args.models.split(","):
        def count(task):
            p = build_params(model, args.effort, db_block, None, task, fmt="eqd", max_tokens=args.max_tokens,
                             version="v3")
            return client.messages.count_tokens(**{k: p[k] for k in ("model", "system", "messages",
                                                                     "output_config", "thinking") if k in p}).input_tokens
        full_max = count(task_text(cases[longest], "v3"))
        base = count("x")
        task_max = full_max - base + 1
        task_mean = task_max * float(np.mean(chars)) / max(chars)
        p = price_for(model, base)
        cost = {}
        for name, plan in plans.items():
            for o in args.out_tokens:
                per = (base * p["read"] + task_mean * p["in"] + o * p["out"]) / 1e6
                cost[f"{name}_out{o}"] = round(len(plan) * per + base * p["w5m"] / 1e6, 1)
        out["models"][model] = {"prefix_tokens": base, "task_max": task_max, "task_mean_est": round(task_mean),
                                "full_request_max": full_max, "room": 1_000_000 - full_max,
                                "fits_with_max_tokens": full_max + args.max_tokens <= 1_000_000,
                                "cost_usd": cost}
        print(model, json.dumps(out["models"][model]), flush=True)
    (OUT_DIR / "token_counts_v3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def cmd_pilot3(args) -> None:
    """承認後、本実行の前に各モデルで数件だけ流して出力の形を確かめる（3 分割で解かせるケースの先頭 n 件。本実行で再利用される）."""
    eqs, id_to_key, key_to_id = load_db_v3()
    cases, splits = load_case_index(), load_splits()
    cfg = _cfg3(args)
    plan = [(cid,) for cid in v3_plan(splits, [42, 123, 456], cases)[: args.n]]
    todo = pending_cases(load_raw(cfg), plan, cfg, fields=("case_id",))
    params_of, meta_of = _params3(args, eqs, cases)
    client, spent = client_(max_retries=0), [_spent_so_far(cfg)]
    if todo:
        p = params_of(todo[0])
        n_in = client.messages.count_tokens(**{k: p[k] for k in ("model", "system", "messages", "output_config",
                                                                 "thinking") if k in p}).input_tokens
        print(f"1 件目の入力 {n_in:,} トークン（max_tokens {args.max_tokens:,}、上限 {CONTEXT_LIMIT[args.model]:,}）")
        if n_in + args.max_tokens > CONTEXT_LIMIT[args.model]:
            sys.exit("入力と出力の上限の和が、入力の上限を超えるので送らない")
    _run_cases(client, cfg, todo, params_of, "pilot", 1, args.budget_usd, spent, attempts=1, meta_of=meta_of,
               fields=("case_id",), group_of=lambda t: "all", stream=True)
    for r in load_raw(cfg):
        if (r.get("case_id"),) in plan:
            correct = [norm(m) for m in cases[r["case_id"]]["correct_model_ids"] if norm(m) in key_to_id]
            sc = score_record(r, id_to_key, correct)
            u = r.get("usage") or {}
            print(f"{r['case_id']}: served={r.get('model_served')} stop={r.get('stop_reason')} "
                  f"out={u.get('output_tokens')} thinking={(u.get('output_tokens_details') or {}).get('thinking_tokens')} "
                  f"sec={r.get('seconds')} cost=${cost_usd(u, cfg['model']) if u else 0:.3f} "
                  f"n={len(sc['ranked_ids'])} invalid={sc['n_invalid']} dup={sc['n_duplicates']} "
                  f"K={sc['K']} R@20={sc['recall@20']:.3f} AP@20={sc['AP@20']:.3f} parse_error={sc['parse_error']}")


def cmd_run3(args) -> None:
    """本実行：--seeds の分割のテストケース（設定 A と B の和集合）を 1 回ずつ解かせる。途中から再開できる.

    3 通りの分割で流したあとに 10 通りに増やすと、まだ解いていないケースだけを解かせる。
    """
    eqs, _, _ = load_db_v3()
    cases, splits = load_case_index(), load_splits()
    cfg = _cfg3(args)
    plan = [(cid,) for cid in v3_plan(splits, _seeds(args.seeds), cases)]
    todo = pending_cases(load_raw(cfg), plan, cfg, fields=("case_id",))
    spent = [_spent_so_far(cfg)]
    print(f"{cfg_name(cfg)}: plan {len(plan)}, todo {len(todo)}, spent so far ${spent[0]:.2f}, "
          f"budget {args.budget_usd}")
    params_of, meta_of = _params3(args, eqs, cases)
    _run_cases(client_(), cfg, todo, params_of, "run", args.workers, args.budget_usd, spent, meta_of=meta_of,
               fields=("case_id",), group_of=lambda t: "all", stream=True)


def cmd_evaluate3(args) -> None:
    """版 3 の判定と集計：設定 A・B の、10 通りと 3 通りの分割ごとの値（答えのそろった分だけ）と、既存の手法との比較."""
    eqs, id_to_key, key_to_id = load_db_v3()
    cases, splits = load_case_index(), load_splits()
    cfg = _cfg3(args)
    latest = {}
    for r in load_raw(cfg):
        if r.get("error") is None and r.get("case_id"):
            latest[r["case_id"]] = r
    per_case, rows = {}, []
    for cid in v3_plan(splits, ALL_SEEDS, cases):
        r = latest.get(cid)
        if r is None:
            continue
        correct = [norm(m) for m in cases[cid]["correct_model_ids"] if norm(m) in key_to_id]
        sc = score_record(r, id_to_key, correct)
        per_case[cid] = sc
        rows.append({"case_id": cid, "variant": cases[cid].get("variant_type"), "time": r.get("time"),
                     "stop_reason": r.get("stop_reason"), "model_served": r.get("model_served"),
                     "output_tokens": (r.get("usage") or {}).get("output_tokens"),
                     "raw_text": r.get("text"), **sc})
    out = {"config": cfg, "order_seed": V3_ORDER_SEED, "n_scored": len(rows),
           "rules": "ID だけで判定（等価性の判定なし）。DB にない ID と読めない出力は、その順位を占める不正解。"
                    "同じ ID の 2 回目以降は除く。上位 20 件。MAP@20 は標準の定義（上位 20 件の外の正解式は P(e)=0）。"
                    "例題を渡さないので、同じケースの答えを、そのケースがテストに入るすべての分割で使う。",
           "summary": {}, "comparison": {}}
    refs = reference_rows(["A", "B"], ALL_SEEDS, splits)
    out["reference"] = {k: v["summary"] for k, v in refs.items()}
    for name, seeds in (("10splits", ALL_SEEDS), ("3splits", [42, 123, 456])):
        need = v3_plan(splits, seeds, cases)
        lack = [c for c in need if c not in per_case]
        if lack:
            out["summary"][name] = {"missing": len(lack)}
            continue
        for st in ("A", "B"):
            sm = split_summary(per_case, splits, st, seeds)
            out["summary"][f"{st}_{name}"] = sm
            for ref, v in refs.items():
                if st not in v["summary"]:
                    continue
                rs = {sd: x for sd, x in v["summary"][st]["per_seed"].items() if int(sd) in seeds}
                out["comparison"][f"{st}_{name}__vs__{ref}"] = {
                    metric: compare_splits({sd: x[metric] for sd, x in sm["per_seed"].items()},
                                        {sd: x[metric] for sd, x in rs.items()})
                    for metric in ("recall@20", "MAP@20")}
    out["quality"] = {
        "stop_reason": dict(Counter(x["stop_reason"] for x in rows)),
        "model_served": dict(Counter(x["model_served"] for x in rows)),
        "unreadable": sum(1 for x in rows if x["parse_error"]),
        "answers_with_invalid_id": sum(1 for x in rows if x["n_invalid"] > 0),
        "invalid_id_share": sum(x["n_invalid"] for x in rows) / max(1, sum(len(x["ranked_ids"]) for x in rows)),
        "answers_with_duplicates": sum(1 for x in rows if x["n_duplicates"] > 0),
        "answers_over_20": sum(1 for x in rows if x["n_truncated"] > 0),
        "answers_with_20": sum(1 for x in rows if len(x["ranked_ids"]) == TOP_N),
        "n_listed": dict(Counter(len(x["ranked_ids"]) for x in rows)),
        "output_tokens_mean": float(np.mean([x["output_tokens"] or 0 for x in rows])) if rows else None,
        "cost_usd_all_calls": round(_spent_so_far(cfg), 2),
    }
    out["per_case"] = rows
    f = OUT_DIR / f"results_{cfg_name(cfg)}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"n_scored": out["n_scored"], "quality": out["quality"]}, ensure_ascii=False, indent=1))
    for key, v in out["summary"].items():
        if "missing" in v:
            print(f"  {key}: 答えのないケースが {v['missing']} 件")
            continue
        print(f"  {key}: R@20 {v['recall@20']['mean']:.3f}±{v['recall@20']['sd']:.3f}"
              f"  MAP@20 {v['MAP@20']['mean']:.3f}±{v['MAP@20']['sd']:.3f}")
    print(f"wrote {f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("splits")
    p.add_argument("--force", action="store_true")
    p.add_argument("--exclude", default=None, help="外すケースの JSON（キー excluded_case_ids）。外してから分割する")
    p.add_argument("--check", action="append",
                   help="設定=per-case 出力のパターン（{seed} を種に置き換える）。テストの並びが一致するか確かめる。複数可")
    p.set_defaults(fn=cmd_splits)

    def common(p, settings="A,B", seeds="42,123,456"):
        p.add_argument("--model", default="claude-opus-5-5", choices=list(PRICES))
        p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--db-format", default="eqc+miss", choices=list(DB_FORMATS))
        p.add_argument("--prompt-version", default=PROMPT_VERSION, choices=["v1", "v2"])
        p.add_argument("--settings", default=settings)
        p.add_argument("--seeds", default=seeds)
        p.add_argument("--ttl", default="5m", choices=["5m", "1h"])
        p.add_argument("--max-tokens", type=int, default=16000)
        p.add_argument("--max-examples", type=int, default=None,
                       help="参考ケースを無作為にこの件数だけ渡す（既定: 訓練と検証の全ケース）")

    p = sub.add_parser("count")
    p.add_argument("--models", default="claude-fable-5-1,claude-opus-5-5,claude-sonnet-5-5,claude-haiku-5-5")
    p.add_argument("--formats", default="eqc+miss,eqc")
    p.add_argument("--settings", default="A,B")
    p.add_argument("--seeds", default=",".join(map(str, ALL_SEEDS)))
    p.add_argument("--prompt-version", default=PROMPT_VERSION, choices=["v1", "v2"])
    p.add_argument("--reserve", type=int, default=16000, help="出力（思考を含む）のために残すトークン数")
    p.set_defaults(fn=cmd_count)
    p = sub.add_parser("check-compress")
    p.set_defaults(fn=cmd_check_compress)
    p = sub.add_parser("costs")
    p.add_argument("--prompt-version", default=PROMPT_VERSION, choices=["v1", "v2"])
    p.add_argument("--db-format", default="eqc+miss", choices=list(DB_FORMATS))
    p.add_argument("--reserve", type=int, default=16000)
    p.add_argument("--out-tokens", type=lambda x: [int(v) for v in x.split(",")], default=[70, 2000],
                   help="1 回の出力トークン数の想定（カンマ区切り）")
    p.set_defaults(fn=cmd_costs)
    p = sub.add_parser("pilot")
    common(p)
    p.add_argument("--setting", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--force", action="store_true", help="入力の上限の確認を飛ばして送る")
    p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--budget-usd", type=float, default=None, help="この額に達したら止める（再開は同じコマンド）")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("evaluate")
    common(p)
    p.set_defaults(fn=cmd_evaluate)

    # 版 3（ゼロショット、全ケース × 3 回）
    p = sub.add_parser("fix-order", help="版 3 の DB の並び（無作為）を db_ids_v3.json に固定する")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_fix_order)

    def common3(p):
        p.add_argument("--model", required=True, choices=list(PRICES))
        p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
        p.add_argument("--ttl", default="5m", choices=["5m", "1h"])
        p.add_argument("--max-tokens", type=int, default=V3_MAX_TOKENS)
        p.add_argument("--budget-usd", type=float, default=None,
                       help="このモデルのこの条件の費用の累計（試しを含む）がこの額に達したら止める")

    p = sub.add_parser("count3")
    p.add_argument("--models", default="claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5,claude-fable-5-1")
    p.add_argument("--effort", default="high")
    p.add_argument("--max-tokens", type=int, default=V3_MAX_TOKENS)
    p.add_argument("--out-tokens", type=lambda x: [int(v) for v in x.split(",")], default=[70, 2000, 8000])
    p.set_defaults(fn=cmd_count3)
    p = sub.add_parser("pilot3")
    common3(p)
    p.add_argument("--n", type=int, default=3)
    p.set_defaults(fn=cmd_pilot3)
    p = sub.add_parser("run3")
    common3(p)
    p.add_argument("--seeds", required=True, help="解かせる分割の乱数の種（カンマ区切り、または all）。例: 42,123,456")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=cmd_run3)
    p = sub.add_parser("evaluate3")
    common3(p)
    p.set_defaults(fn=cmd_evaluate3)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
