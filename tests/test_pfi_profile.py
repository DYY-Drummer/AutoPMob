import json
import numpy as np
from analyze_pfi_profile import (
    load_per_case, split_dependent_robust, attr_array,
    mannwhitney_effect, standardized_mean_diff, spearman_drop_sep,
    smd_stderr,
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


from analyze_pfi_profile import sep_profile, build_stats


def test_sep_profile_bins_increase():
    recs = [{"feature": "gComp", "base_R": 1.0, "flip": 1,
             "sep": s, "drop": s, "n_input": 5, "n_correct": 1,
             "n_output": 1, "n_sources": 1}
            for s in np.linspace(0, 1, 40)]
    xs, ys, es = sep_profile(recs, "gComp", n_bins=4)
    assert len(xs) == 4
    assert ys[-1] > ys[0]            # sep が大きいビンほど drop 大
    assert all(e >= 0 for e in es)


def test_build_stats_has_features_and_attrs():
    recs = _recs()
    st = build_stats(recs)
    assert "gComp" in st["features"]
    g = st["features"]["gComp"]
    assert "n_input" in g["attrs"] and "sep" in g["attrs"]
    assert "spearman_drop_sep" in g
    assert g["n_dependent"] == 4 and g["n_robust"] == 4


def test_smd_stderr_matches_closed_form():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    a, b = attr_array(dep, "n_input"), attr_array(rob, "n_input")
    d = standardized_mean_diff(a, b)
    n1, n2 = a.size, b.size
    expected = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    assert abs(smd_stderr(a, b) - expected) < 1e-12
    assert smd_stderr(a, b) > 0


def test_smd_stderr_nan_when_group_too_small():
    import math
    one = np.array([1.0])
    assert math.isnan(smd_stderr(one, np.array([2.0, 3.0])))
