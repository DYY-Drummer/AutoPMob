"""var_conditions.py（条件ごとのケースの見え方・照合器・τ の組）のテスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.1, §3.4, §4 の 1・5
実データを読むテスト（load_data）は 10 秒前後かかる。
"""
import numpy as np
import pytest

from var_conditions import (
    rename_subset, build_view, build_views, load_data, SymbolMatcher, calibration_pairs,
)
from two_stage_query_conditioned import case_text, io_vars, in_vars, out_vars


def test_rename_subset_ceil_nested_deterministic():
    syms = ["A", "B", "C"]
    assert rename_subset("x", syms, 0.0) == set()
    assert rename_subset("x", syms, 1.0) == set(syms)
    half = rename_subset("x", syms, 0.5)
    assert len(half) == 2 and half <= rename_subset("x", syms, 1.0)
    assert half == rename_subset("x", list(reversed(syms)), 0.5)
    assert len(rename_subset("y", ["T"], 0.5)) == 1


CASE = {"case_id": "toy_1", "context": "Reactor sharing $T$.",
        "input_variables": ["T", "F"], "output_variables": ["C_A"],
        "correct_model_ids": ["src__eq_2", "src__eq_1"], "variant_type": "original"}
EQ_BY_KEY = {"src__eq_1": {"variables": {"T": "temperature in the tank [K]", "F": "feed flow"}},
             "src__eq_2": {"variables": {"C_A": "concentration of A", "T": "reactor temperature"}}}
VD = {"context_nosym": "Reactor sharing the temperature.",
      "variables": {
          "T": {"paraphrase": "temperature of the reactor", "alt_symbol": "\\theta", "meaning_conflict": False},
          "F": {"paraphrase": "volumetric feed rate", "alt_symbol": "q", "meaning_conflict": False},
          "C_A": {"paraphrase": "concentration of species A", "alt_symbol": "c_A", "meaning_conflict": False}}}


def test_view_S0_matches_case_text():
    v = build_view("S0", CASE, None, EQ_BY_KEY)
    assert v.ctx == case_text(CASE) and v.io_text == case_text(CASE, io=True)
    assert v.orig_syms == sorted(io_vars(CASE)) == v.syms == ["C_A", "F", "T"]
    assert list(v.q_in) == [False, True, True] and list(v.q_out) == [True, False, False]
    assert v.texts is None


def test_view_S1_rename_all():
    v = build_view("S1", CASE, VD, EQ_BY_KEY, p=1.0)
    assert v.ctx == "Reactor sharing the temperature."
    assert v.syms == ["c_A", "q", "\\theta"] and v.orig_syms == ["C_A", "F", "T"]
    assert v.io_text == "Reactor sharing the temperature. INPUT \\theta q OUTPUT c_A"
    v0 = build_view("S1", CASE, VD, EQ_BY_KEY, p=0.0)
    assert v0.syms == ["C_A", "F", "T"] and v0.io_text == "Reactor sharing the temperature. INPUT T F OUTPUT C_A"


def test_view_D1_and_D0():
    d1 = build_view("D1", CASE, VD, EQ_BY_KEY)
    assert d1.texts == ["concentration of species A", "volumetric feed rate", "temperature of the reactor"]
    assert d1.io_text == ("Reactor sharing the temperature. INPUT temperature of the reactor "
                          "volumetric feed rate OUTPUT concentration of species A")
    d0 = build_view("D0", CASE, VD, EQ_BY_KEY)
    # 正解式の順（src__eq_2 → src__eq_1）で記号を最初に含む式の説明
    assert d0.texts == ["concentration of A", "feed flow", "reactor temperature"]


def test_calibration_pairs_labels():
    d = {"eq_ptr": np.array([0, 2, 4]), "occ_sym": ["F", "T", "C_A", "T"], "correct_lists": [[1, 0]]}
    v = build_view("D1", CASE, VD, EQ_BY_KEY)
    pr = calibration_pairs([0], [v], d)
    got = sorted(zip(pr["item"].tolist(), pr["occ"].tolist(), pr["label"].tolist()))
    # 照会 (C_A, F, T)。式 0 = {F, T}、式 1 = {C_A, T}
    assert got == sorted([(0, 2, 1), (0, 3, 0), (1, 0, 1), (1, 1, 0),
                          (2, 2, 0), (2, 3, 1), (2, 0, 0), (2, 1, 1)])


@pytest.fixture(scope="module")
def data_a():
    return load_data("A")


def test_load_data_occurrences(data_a):
    d = data_a
    assert len(d["cases"]) == 1823 and len(d["eq_keys"]) == 11146
    assert int(d["eq_ptr"][-1]) == len(d["occ_sym"]) == len(d["occ_desc"]) == 58008
    assert int((d["n_vars"] == 0).sum()) == 54
    for j in range(0, 11146, 97):
        a, b = d["eq_ptr"][j], d["eq_ptr"][j + 1]
        assert set(d["occ_sym"][a:b]) == d["eq_vars"][j]


def test_symbol_features_reproduce_existing(data_a):
    """完全一致の照合で、新しい特徴量が set_aware_reranker.compute_features_with_set と一致する（§4 の 1）."""
    from set_aware_reranker import compute_features_with_set
    from var_matching import (SYMBOL_TAU, K_REF, occurrence_incidence, stage1_match_counts,
                              jaccard_from_counts, case_features, domain_flags)
    d = data_a
    views = build_views("S0", d, None)
    sm = SymbolMatcher(d, views)
    inc = occurrence_incidence(d["eq_ptr"])
    n_eq = len(d["eq_keys"])
    ts = np.linspace(0.0, 1.0, n_eq)
    svd = np.linspace(1.0, 0.0, n_eq)
    for ci in list(range(0, len(d["cases"]), 91))[:20]:
        v = views[ci]
        full = sm.q_occ(ci)
        vs = jaccard_from_counts(stage1_match_counts(full >= SYMBOL_TAU, inc), len(v.syms), d["n_vars"])
        cands = np.argsort(-vs)[:50].tolist()
        occ = np.concatenate([np.arange(d["eq_ptr"][j], d["eq_ptr"][j + 1]) for j in cands]).astype(np.int64)
        cptr = np.concatenate([[0], np.cumsum([d["n_vars"][j] for j in cands])]).astype(np.int64)
        ref_end = int(cptr[min(K_REF, len(cands))])
        new, _ = case_features(ts[cands], svd[cands], domain_flags(cands, d["eq_domains"], v.ctx),
                               v.q_in, v.q_out, full[:, occ], sm.occ_occ(occ, occ[:ref_end]),
                               cptr, [d["eq_domains"][j] for j in cands], SYMBOL_TAU)
        c = d["cases"][ci]
        old = compute_features_with_set(cands, ts, io_vars(c), in_vars(c), out_vars(c), svd,
                                        d["eq_vars"], d["eq_domains"], case_text(c), None, io_vars(c),
                                        False, ("Comp", "Coh", "Dom"))
        assert np.array_equal(new, old), c["case_id"]
