"""set_aware_reranker.py に 2026-10-20 に足したオプションのテスト.

- filter_excluded：外すケースの一覧（JSON の excluded_case_ids）でケースを除く
- equiv_ranks：別の出典にある同じ式（正規化した LaTeX が同じ）を正解とみなした順位（1 本の正解式は 1 回だけ）
- run_mode(eval_val=True)：検証用のケースも評価し、テストの結果は変えない
- run_mode(equiv_ids=...)：per-case に ranks_equiv を足す（式が互いに違えば ranks と同じ）
"""
import json

import pytest

from set_aware_reranker import equiv_ranks, filter_excluded, run_mode
from test_stage2_fast import _toy_run_inputs


def test_filter_excluded_drops_listed_case_ids(tmp_path):
    cases = [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"excluded_case_ids": ["b"]}))
    assert [c["case_id"] for c in filter_excluded(cases, str(p))] == ["a", "c"]


def test_equiv_ranks_counts_an_equivalent_equation_and_each_gt_once():
    ids = {10: "A", 11: "B", 12: "A", 13: "C", 14: "B"}
    ranked = [13, 12, 11, 10, 14]          # 12 は 10 と同じ式（"A"）、14 は 11 と同じ式（"B"）
    # 正解 10 は同じ式の 12 が 2 位にある → 2 位。正解 11 は自分が 3 位 → 3 位（14 は数えない）
    assert equiv_ranks(ranked, {10, 11}, ids, miss_rank=10_000) == [2, 3]


def test_equiv_ranks_two_gt_with_the_same_string_use_distinct_positions():
    ids = {1: "A", 2: "A", 3: "A", 4: "Z"}
    ranked = [4, 3, 1]                      # 正解 1・2 はどちらも "A"。並んだ "A" は 3 と 1 の 2 本
    assert sorted(equiv_ranks(ranked, {1, 2}, ids, miss_rank=10_000)) == [2, 3]


def test_equiv_ranks_missing_when_no_equivalent_is_ranked():
    ids = {1: "A", 2: "B", 3: "C"}
    assert equiv_ranks([2, 3], {1}, ids, miss_rank=10_000) == [10_000]


def _common():
    cases, ek, et, ev, ed, es, cl, cs = _toy_run_inputs()
    return dict(seeds=[42], cases=cases, eq_keys_l=ek, eq_texts_l=et, eq_vars_l=ev,
                eq_domains_l=ed, eq_sources_l=es, correct_lists=cl, case_srcs=cs,
                top_k=15, epochs=2, lr=1e-3, save_per_case=True, split_mode="stratified", fast=True)


def test_eval_val_adds_validation_cases_without_changing_test():
    base = run_mode("reranker-10S", **_common())
    both = run_mode("reranker-10S", **_common(), eval_val=True)
    test = [{k: v for k, v in r.items() if k != "split"} for r in both["per_case"] if r["split"] == "test"]
    assert test == base["per_case"]
    assert any(r["split"] == "val" for r in both["per_case"])
    assert both["Recall@20"] == base["Recall@20"]


def test_equiv_ids_identity_gives_same_ranks():
    c = _common()
    ids = list(range(len(c["eq_keys_l"])))   # どの式も互いに違う
    r = run_mode("reranker-10S", **c, equiv_ids=ids)
    assert all(rec["ranks_equiv"] == rec["ranks"] for rec in r["per_case"])


def test_equiv_ranks_ignores_same_source_duplicates_when_sources_given():
    """報告書 §3.5 の 386/3,132 本と同じ定義：同じ式は「別の出典」にあるものだけを数える."""
    ids = {1: "A", 2: "A", 3: "A"}
    src = {1: "s1", 2: "s1", 3: "s2"}
    # 正解 1 と同じ式は 2（同じ出典 s1、数えない）と 3（別の出典 s2、数える）
    assert equiv_ranks([2, 3, 1], {1}, ids, miss_rank=10_000, eq_src=src) == [2]
    assert equiv_ranks([2, 3, 1], {1}, ids, miss_rank=10_000) == [1]
