"""paper/cace/db_counts.py（式データベースから論文に書く数を数え、追跡するファイルに保存する）の単体テスト."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper" / "cace"))
import db_counts as dc  # noqa: E402

EQS = [
    {"source_id": "a.pdf", "eq_id": "eq_1", "domain": "CSTR", "variables": {"k": "rate constant", "T": "temperature"}},
    {"source_id": "a.pdf", "eq_id": "eq_2", "domain": "CSTR", "variables": {"k": "thermal conductivity"}},
    {"source_id": "b.pdf", "eq_id": "eq_1", "domain": "Heat Transfer", "variables": {"T": "temperature", "q": "flux"}},
    {"source_id": "handbook_x", "eq_id": "eq_1", "domain": "Optics", "variables": {"n": "index"}},
]


def test_count_database_counts_sources_symbols_and_labels():
    out = dc.count_database(EQS)
    assert out["n_equations"] == 4
    assert out["n_sources"] == 3 and out["n_pdf"] == 2 and out["n_reference_sets"] == 1
    assert out["n_equations_pdf"] == 3 and out["n_equations_reference"] == 1
    assert out["n_symbols"] == 4                    # k, T, q, n
    assert out["n_symbols_shared"] == 2             # k and T appear in two equations
    assert out["n_symbols_conflict"] == 1           # k has two different definitions; T has one
    assert out["n_labels"] == 3 and out["share_labels_once"] == 2 / 3


def test_synth_one_source_counts_synthesized_cases_that_draw_on_one_source():
    cases = [{"variant_type": "dae_X2", "correct_model_ids": ["a.pdf__eq_1", "a.pdf__eq_2"]},
             {"variant_type": "dae_X2", "correct_model_ids": ["a.pdf__eq_1", "b.pdf__eq_1"]},
             {"variant_type": "original", "correct_model_ids": ["a.pdf__eq_1"]}]
    assert dc.synth_one_source(cases, EQS) == 1
