# tests/test_run_var_description.py
"""run_var_description.py のテスト.

回帰の確認: S0 が既報（w30-70・seed 42）と第 1 段の被覆率を再現する.
仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §4 の 7
実データで学習するので数分かかる。RUN_SLOW=1 のときだけ実行する（slow）。
tau_for の確認（速い）: データは読まず、較正の組・類似度・Youden の τ を差し替えて、有限でない τ を ValueError にする。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import run_var_description as rvd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "experiments/xs1/reranker-10S_w30-70__42.json"
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="RUN_SLOW=1 のときだけ（数分かかる）")


@slow
def test_s0_reproduces_reference(tmp_path):
    out = tmp_path / "s0.json"
    subprocess.run([sys.executable, "run_var_description.py", "--setting", "A", "--conds", "S0",
                    "--seed-list", "42", "--output", str(out)], cwd=ROOT, check=True)
    got = json.load(open(out))["results"]["S0"]["per_seed"][0]
    ref = json.load(open(REF))["results"]["reranker-10S"]["per_seed"][0]
    assert got["n_cases"] == ref["n_cases"] == 423
    for k in ("Recall@K_correct", "MAP", "Recall@20"):
        assert abs(got[k] - ref[k]) <= 1e-4, (k, got[k], ref[k])


@slow
def test_s0_coverage_reproduces_a1(tmp_path):
    out = tmp_path / "cov.json"
    subprocess.run([sys.executable, "run_var_description.py", "--setting", "A", "--conds", "S0",
                    "--coverage", "--output", str(out)], cwd=ROOT, check=True)
    cov = json.load(open(out))["coverage"]["S0"]
    assert abs(cov["w30-70"] - 0.9670508667903073) < 1e-9
    assert abs(cov["w70-30"] - 0.9202866372367196) < 1e-9


# ---- tau_for（データは読まない。calibration_pairs・pair_sims・youden_tau を run_var_description の名前空間で差し替える） ----

class DescMatcher:
    kind = "description"  # tau_for が見るのは kind だけ


def install_fake_calibration(monkeypatch, tau) -> tuple:
    """較正の組・類似度・Youden の τ を偽物にする。(pair_sims の値, 組のラベル, youden_tau が受け取った引数の記録) を返す."""
    sims, labels, calls = np.array([0.9, 0.1]), np.array([1, 0]), []

    def fake_youden(s, y):
        calls.append((s, y))
        return tau

    monkeypatch.setattr(rvd, "calibration_pairs", lambda idx, views, data: {"label": labels})
    monkeypatch.setattr(rvd, "pair_sims", lambda pr, matcher: sims)
    monkeypatch.setattr(rvd, "youden_tau", fake_youden)
    return sims, labels, calls


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"])
def test_tau_for_rejects_a_non_finite_tau(monkeypatch, bad):
    """sklearn の roc_curve は先頭の閾値が inf で、J が全部同じ（退化した組）だとそれが選ばれる。
    そのまま使うと説明版の条件がすべて「一致なし」になり、黙って結果が壊れる."""
    install_fake_calibration(monkeypatch, bad)
    with pytest.raises(ValueError, match=rf"τ.*{bad!r}"):
        rvd.tau_for(DescMatcher(), [], {}, [0, 1])


@pytest.mark.parametrize("delta", [0.0, -0.02, 0.02], ids=["delta-0", "delta-minus", "delta-plus"])
def test_tau_for_returns_a_finite_tau_plus_delta_unchanged(monkeypatch, delta):
    sims, labels, calls = install_fake_calibration(monkeypatch, 0.8123)
    assert rvd.tau_for(DescMatcher(), [], {}, [0, 1], delta) == 0.8123 + delta
    assert len(calls) == 1 and calls[0][0] is sims and calls[0][1] is labels  # pair_sims の値と較正の組のラベルがそのまま渡る


def test_tau_for_symbol_mode_is_symbol_tau_and_does_not_calibrate(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("記号版は較正しない")

    for name in ("calibration_pairs", "pair_sims", "youden_tau"):
        monkeypatch.setattr(rvd, name, forbidden)
    symbol = type("SymbolMatcher", (), {"kind": "symbol"})()
    assert rvd.tau_for(symbol, [], {}, [0, 1], 0.02) == rvd.SYMBOL_TAU == 1.0
