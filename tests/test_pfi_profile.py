import json
import numpy as np
from analyze_pfi_profile import (
    load_per_case, split_dependent_robust, attr_array,
    mannwhitney_effect, standardized_mean_diff, spearman_drop_sep,
)


def _recs():
    # gComp: solved 8件。dep(flip=1) は n_input 大・sep 大、rob(flip=0) は小。
    # dep/rob 各4件（完全分離）→ mannwhitney 両側 p≈0.029 で安定して <0.1。
    out = []
    for flip, ni, sep, drop in [
        (1, 12, 0.50, 0.40), (1, 11, 0.45, 0.35), (1, 13, 0.55, 0.42), (1, 10, 0.48, 0.38),
        (0, 4, 0.05, 0.00), (0, 5, 0.03, 0.00), (0, 3, 0.06, 0.01), (0, 6, 0.04, 0.00),
    ]:
        out.append({"feature": "gComp", "base_R": 1.0, "flip": flip,
                    "n_input": ni, "sep": sep, "drop": drop, "n_correct": 1})
    # base_R<1.0 は依存/頑健から除外される
    out.append({"feature": "gComp", "base_R": 0.5, "flip": 0,
                "n_input": 9, "sep": 0.20, "drop": 0.20, "n_correct": 2})
    return out


def test_split_dependent_robust_conditions_on_solved():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    assert len(dep) == 4 and len(rob) == 4  # base_R<1.0 は除外


def test_mannwhitney_effect_sign_positive_when_dep_larger():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    e = mannwhitney_effect(attr_array(dep, "n_input"), attr_array(rob, "n_input"))
    assert e["mean_dep"] > e["mean_rob"]
    assert e["rank_biserial_r"] > 0.9   # 完全分離に近い
    assert e["mannwhitney_p"] < 0.1


def test_standardized_mean_diff_positive():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    smd = standardized_mean_diff(attr_array(dep, "sep"), attr_array(rob, "sep"))
    assert smd > 1.0


def test_spearman_drop_sep_positive():
    s = spearman_drop_sep(_recs(), "gComp")
    assert s["rho"] > 0.5


def test_load_per_case(tmp_path):
    doc = {"config": {"setting": "A"}, "records": [{"feature": "gComp", "base_R": 1.0}]}
    p = tmp_path / "pc.json"
    p.write_text(json.dumps(doc))
    recs, cfg = load_per_case(p)
    assert cfg["setting"] == "A" and recs[0]["feature"] == "gComp"
