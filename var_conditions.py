"""条件ごとのケースの見え方（説明文・照会の変数）と、変数の照合器.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.1, §3.4
条件:
  S0 元の説明文 ＋ 文献の記号
  S1 記号なしの説明文 ＋ 文献の記号（p > 0 でテスト用に一部を別の記号へ = S1-r）
  D1 記号なしの説明文 ＋ LLM の言い換え
  D0 記号なしの説明文 ＋ 正解式の説明をそのまま（漏れの大きさを測る参考）
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import (  # noqa: E402
    load_equations, norm, eq_key, case_text, io_vars, in_vars, out_vars,
)
from external_baselines import load_setting, embed_texts, E5_MODEL  # noqa: E402

EXP = ROOT / "experiments"
VAR_DESC_JSON = EXP / "var_descriptions.json"
CONDITIONS = ("S0", "S1", "D1", "D0")


@dataclass
class CaseView:
    ctx: str               # 説明文（f0・第 1 段の文章類似度・f6 に使う）
    orig_syms: list        # 照会の変数の元の記号（sorted(io_vars)）
    syms: list             # 照合に使う記号（S1-r では一部が別の記号）
    texts: list | None     # 説明版の照会テキスト（D1/D0）。記号版は None
    q_in: np.ndarray       # bool: 入力変数か
    q_out: np.ndarray      # bool: 出力変数か
    io_text: str           # f2（SVD）の照会文。case_text(c, io=True) と同じ形


def load_data(setting: str) -> dict:
    """設定 A/B のデータ（external_baselines.load_setting ＋ 分野・変数の出現）."""
    data = load_setting(setting)
    eqs = [e for e in load_equations() if eq_key(e)]
    assert [eq_key(e) for e in eqs] == data["eq_keys"]
    occ_sym, occ_desc, eq_ptr = [], [], [0]
    for e in eqs:
        vd = e.get("variables") or {}
        first = {}
        if isinstance(vd, dict):
            for k, v in vd.items():
                s = norm(k)
                if s and s not in first:
                    first[s] = norm(v) if isinstance(v, str) else ""
        for s in sorted(first):
            occ_sym.append(s)
            occ_desc.append(first[s])
        eq_ptr.append(len(occ_sym))
    eq_ptr = np.asarray(eq_ptr, dtype=np.int64)
    n_vars = np.diff(eq_ptr)
    data.update(
        eqs=eqs,
        eq_by_key=dict(zip(data["eq_keys"], eqs)),
        eq_domains=[norm(e.get("domain") or "") for e in eqs],
        occ_sym=occ_sym, occ_desc=occ_desc, eq_ptr=eq_ptr, n_vars=n_vars,
        occ_eq=np.repeat(np.arange(len(eqs)), n_vars),
    )
    return data


def load_var_descriptions(path: Path = VAR_DESC_JSON) -> dict:
    """make_var_descriptions.py finalize の出力（case_id → 生成結果）."""
    return json.load(open(path, encoding="utf-8"))["cases"]


def rename_subset(case_id: str, syms: list, p: float) -> set:
    """付け替える変数（⌈p × 変数の数⌉ 個）。case_id から決まる乱数の並びの先頭なので、p を変えても入れ子."""
    n = math.ceil(round(p * len(syms), 9))
    if n <= 0:
        return set()
    order = sorted(syms)
    random.Random(int(hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:8], 16)).shuffle(order)
    return set(order[:n])


def d0_description(sym: str, case: dict, eq_by_key: dict) -> str:
    """正解式（correct_model_ids の順）のうち、記号 sym を最初に含む式の説明."""
    for m in case.get("correct_model_ids") or []:
        e = eq_by_key.get(norm(m))
        if not e:
            continue
        for k, v in (e.get("variables") or {}).items():
            if norm(k) == sym:
                return norm(v) if isinstance(v, str) else ""
    raise KeyError(f"{case.get('case_id')}: symbol {sym!r} not in correct equations")


def io_text(ctx: str, case: dict, rep) -> str:
    """case_text(case, io=True) と同じ形で、入出力の各記号を rep(記号) に置き換えた文字列."""
    ins = " ".join(rep(norm(v)) for v in (case.get("input_variables") or []) if norm(v))
    outs = " ".join(rep(norm(v)) for v in (case.get("output_variables") or []) if norm(v))
    return " ".join(filter(None, [ctx, f"INPUT {ins}" if ins else "", f"OUTPUT {outs}" if outs else ""]))


def build_view(cond: str, case: dict, vd_case, eq_by_key: dict, p: float = 0.0) -> CaseView:
    orig = sorted(io_vars(case))
    iv, ov = in_vars(case), out_vars(case)
    q_in = np.array([s in iv for s in orig], dtype=bool)
    q_out = np.array([s in ov for s in orig], dtype=bool)
    if cond == "S0":
        return CaseView(case_text(case), orig, list(orig), None, q_in, q_out, case_text(case, io=True))
    if vd_case is None:
        raise KeyError(f"{case.get('case_id')}: no generated descriptions in var_descriptions.json")
    ctx = norm(vd_case["context_nosym"])
    var = vd_case["variables"]
    if cond == "S1":
        alt = {s: norm(var[s]["alt_symbol"]) for s in rename_subset(case["case_id"], orig, p)}
        rep = lambda s: alt.get(s, s)  # noqa: E731
        return CaseView(ctx, orig, [rep(s) for s in orig], None, q_in, q_out, io_text(ctx, case, rep))
    if cond == "D1":
        para = {s: norm(var[s]["paraphrase"]) for s in orig}
    elif cond == "D0":
        para = {s: d0_description(s, case, eq_by_key) for s in orig}
    else:
        raise ValueError(cond)
    rep = lambda s: para[s]  # noqa: E731
    return CaseView(ctx, orig, list(orig), [para[s] for s in orig], q_in, q_out, io_text(ctx, case, rep))


def build_views(cond: str, data: dict, vd, p: float = 0.0) -> list:
    return [build_view(cond, c, (vd or {}).get(c.get("case_id")), data["eq_by_key"], p)
            for c in data["cases"]]


class SymbolMatcher:
    """記号の完全一致（一致 = 1.0）。τ = SYMBOL_TAU と組で使う."""
    kind = "symbol"

    def __init__(self, data: dict, views: list):
        self.occ = np.asarray(data["occ_sym"], dtype=object)
        self.pos = {}
        for o, s in enumerate(data["occ_sym"]):
            self.pos.setdefault(s, []).append(o)
        self.syms = [list(v.syms) for v in views]

    def q_occ(self, ci: int) -> np.ndarray:
        syms = self.syms[ci]
        out = np.zeros((len(syms), len(self.occ)), dtype=np.float32)
        for i, s in enumerate(syms):
            out[i, self.pos.get(s, [])] = 1.0
        return out

    def occ_occ(self, a, b) -> np.ndarray:
        return (self.occ[a][:, None] == self.occ[b][None, :]).astype(np.float32)


class DescriptionMatcher:
    """説明の E5 埋め込みのコサイン類似度."""
    kind = "description"

    def __init__(self, occ_emb: np.ndarray, q_emb: np.ndarray, q_idx: list):
        self.occ_emb, self.q_emb, self.q_idx = occ_emb, q_emb, q_idx

    def q_occ(self, ci: int) -> np.ndarray:
        return self.q_emb[self.q_idx[ci]] @ self.occ_emb.T

    def occ_occ(self, a, b) -> np.ndarray:
        return self.occ_emb[a] @ self.occ_emb[b].T


def embed_unique(texts: list, kind: str) -> tuple:
    """重複を除いて E5 で埋め込む（接頭辞 'query: '、キャッシュつき）。(行列, テキスト→行)."""
    uniq = sorted(set(texts))
    E = embed_texts(["query: " + t for t in uniq], E5_MODEL, kind)
    return E, {t: i for i, t in enumerate(uniq)}


def occ_embeddings(data: dict) -> np.ndarray:
    """DB の変数の出現ごとの埋め込み (n_occ × 768)。data にキャッシュする."""
    if "_occ_emb" not in data:
        E, idx = embed_unique(data["occ_desc"], "vardesc_db")
        data["_occ_emb"] = E[[idx[d] for d in data["occ_desc"]]]
    return data["_occ_emb"]


def make_matcher(cond: str, data: dict, views: list):
    if cond in ("S0", "S1"):
        return SymbolMatcher(data, views)
    q_emb, idx = embed_unique([t for v in views for t in v.texts], f"vardesc_{cond}")
    return DescriptionMatcher(occ_embeddings(data), q_emb,
                              [np.array([idx[t] for t in v.texts], dtype=np.int64) for v in views])


def calibration_pairs(idx: list, views: list, data: dict) -> dict:
    """τ の決定と事前確認に使う組。正例: 照会の変数 s と、正解式の中の同じ記号 s の説明。
    負例: 同じ正解式の中の別の記号の説明."""
    eq_ptr, occ_sym, cl = data["eq_ptr"], data["occ_sym"], data["correct_lists"]
    rows = []
    for ci in idx:
        for k, s in enumerate(views[ci].orig_syms):
            for j in cl[ci]:
                a, b = int(eq_ptr[j]), int(eq_ptr[j + 1])
                if s not in occ_sym[a:b]:
                    continue
                for o in range(a, b):
                    rows.append((ci, k, o, 1 if occ_sym[o] == s else 0))
    arr = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    return {"case": arr[:, 0], "item": arr[:, 1], "occ": arr[:, 2], "label": arr[:, 3]}


def pair_sims(pairs: dict, matcher) -> np.ndarray:
    """組ごとの類似度（説明版の照合器だけ）."""
    rows = [matcher.q_idx[c][k] for c, k in zip(pairs["case"].tolist(), pairs["item"].tolist())]
    return np.einsum("ij,ij->i", matcher.q_emb[rows], matcher.occ_emb[pairs["occ"]])
