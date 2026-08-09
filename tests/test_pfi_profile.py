import json
import math
import numpy as np
from analyze_pfi_profile import (
    load_per_case, collapse_to_cases, split_dependent_robust, attr_array,
    mannwhitney_effect, standardized_mean_diff, spearman_drop_sep,
    smd_stderr, sep_profile, cancellation_check, build_stats,
)


def _recs():
    # gComp: solved 8件（各1 case_id、seed重複なし）。dep(flip=1) は n_input 大・sep 大、rob(flip=0) は小。
    # dep/rob 各4件（完全分離）→ mannwhitney 両側 p≈0.029 で安定して <0.1。
    out = []
    for i, (flip, ni, sep, drop) in enumerate([
        (1, 12, 0.50, 0.40), (1, 11, 0.45, 0.35), (1, 13, 0.55, 0.42), (1, 10, 0.48, 0.38),
        (0, 4, 0.05, 0.00), (0, 5, 0.03, 0.00), (0, 3, 0.06, 0.01), (0, 6, 0.04, 0.00),
    ]):
        out.append({"feature": "gComp", "case_id": f"case{i}", "base_R": 1.0, "flip": flip,
                    "n_input": ni, "sep": sep, "drop": drop, "n_correct": 1,
                    "n_output": 1, "n_sources": 1, "variant": "original"})
    # base_R<1.0 は依存/頑健から除外される
    out.append({"feature": "gComp", "case_id": "case_unsolved", "base_R": 0.5, "flip": 0,
                "n_input": 9, "sep": 0.20, "drop": 0.20, "n_correct": 2,
                "n_output": 1, "n_sources": 1, "variant": "original"})
    return out


def _seed_repeat_recs():
    # 同一 case_id が複数 seed（重複）で異なる sep/drop/flip を持つ状況を模す。
    # case "A": seed1 flip=1(sep=0.9,drop=0.5), seed2 flip=0(sep=0.1,drop=0.0)
    #   → 平均 sep=0.5, 平均 drop=0.25, flip_frac=0.5 → flip=1（>=0.5 は反転扱い、境界値）
    # case "B": 2 seed とも flip=0 → 頑健
    # case "C": base_R<1.0（未解決）→ solved_only=True では除外される
    return [
        {"feature": "gComp", "case_id": "A", "base_R": 1.0, "flip": 1,
         "sep": 0.9, "drop": 0.5, "n_correct": 1, "n_input": 10, "n_output": 1,
         "n_sources": 1, "variant": "original"},
        {"feature": "gComp", "case_id": "A", "base_R": 1.0, "flip": 0,
         "sep": 0.1, "drop": 0.0, "n_correct": 1, "n_input": 10, "n_output": 1,
         "n_sources": 1, "variant": "original"},
        {"feature": "gComp", "case_id": "B", "base_R": 1.0, "flip": 0,
         "sep": 0.05, "drop": 0.0, "n_correct": 2, "n_input": 4, "n_output": 1,
         "n_sources": 1, "variant": "original"},
        {"feature": "gComp", "case_id": "B", "base_R": 1.0, "flip": 0,
         "sep": 0.03, "drop": 0.01, "n_correct": 2, "n_input": 4, "n_output": 1,
         "n_sources": 1, "variant": "original"},
        {"feature": "gComp", "case_id": "C", "base_R": 0.5, "flip": 0,
         "sep": 0.2, "drop": 0.2, "n_correct": 2, "n_input": 9, "n_output": 1,
         "n_sources": 1, "variant": "dae_x"},
    ]


def test_load_per_case(tmp_path):
    doc = {"config": {"setting": "A"}, "n_seeds": 10,
           "records": [{"feature": "gComp", "base_R": 1.0}]}
    p = tmp_path / "pc.json"
    p.write_text(json.dumps(doc))
    recs, cfg, n_seeds = load_per_case(p)
    assert cfg["setting"] == "A" and recs[0]["feature"] == "gComp"
    assert n_seeds == 10


# --- collapse_to_cases（FIX C1） -------------------------------------------

def test_collapse_to_cases_averages_seed_repeats_and_filters_unsolved():
    cases = collapse_to_cases(_seed_repeat_recs(), "gComp", solved_only=True)
    by_id = {c["case_id"]: c for c in cases}
    assert set(by_id) == {"A", "B"}          # C は base_R<1.0 で除外
    assert by_id["A"]["n_obs"] == 2
    assert abs(by_id["A"]["sep"] - 0.5) < 1e-9
    assert abs(by_id["A"]["drop"] - 0.25) < 1e-9
    assert abs(by_id["A"]["flip_frac"] - 0.5) < 1e-9
    assert by_id["A"]["flip"] == 1           # ちょうど0.5は反転扱い（演算子は >=）
    assert by_id["B"]["flip"] == 0
    # 属性は seed 間で不変という前提なので先頭レコードの値をそのまま使う
    assert by_id["A"]["n_input"] == 10
    assert by_id["B"]["n_correct"] == 2


def test_collapse_to_cases_solved_only_false_includes_unsolved():
    cases = collapse_to_cases(_seed_repeat_recs(), "gComp", solved_only=False)
    assert "C" in {c["case_id"] for c in cases}


# --- split_dependent_robust（FIX C1） --------------------------------------

def test_split_dependent_robust_conditions_on_solved():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    assert len(dep) == 4 and len(rob) == 4  # base_R<1.0 は除外


def test_split_dependent_robust_no_case_overlap_after_collapse():
    dep, rob = split_dependent_robust(_seed_repeat_recs(), "gComp")
    dep_ids = {c["case_id"] for c in dep}
    rob_ids = {c["case_id"] for c in rob}
    assert dep_ids.isdisjoint(rob_ids)       # 収集前は "A" が両方の記録を持っていた
    assert dep_ids == {"A"}
    assert rob_ids == {"B"}


# --- attr_array -------------------------------------------------------------

def test_attr_array_drops_non_finite_values():
    recs = [{"n_input": 5.0}, {"n_input": float("nan")},
            {"n_input": float("inf")}, {"n_input": 7.0}]
    arr = attr_array(recs, "n_input")
    assert sorted(arr.tolist()) == [5.0, 7.0]


# --- mannwhitney_effect ------------------------------------------------------

def test_mannwhitney_effect_sign_positive_when_dep_larger():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    e = mannwhitney_effect(attr_array(dep, "n_input"), attr_array(rob, "n_input"))
    assert e["mean_dep"] > e["mean_rob"]
    assert e["rank_biserial_r"] > 0.9   # 完全分離に近い
    assert e["mannwhitney_p"] < 0.1


def test_mannwhitney_effect_sem_nan_for_empty_group():
    e = mannwhitney_effect(np.array([]), np.array([1.0, 2.0, 3.0]))
    assert math.isnan(e["sem_dep"])
    assert math.isnan(e["mean_dep"])


def test_mannwhitney_effect_sem_zero_for_singleton_group():
    e = mannwhitney_effect(np.array([5.0]), np.array([1.0, 2.0, 3.0]))
    assert e["sem_dep"] == 0.0
    assert e["mean_dep"] == 5.0


# --- standardized_mean_diff / smd_stderr ------------------------------------

def test_standardized_mean_diff_positive():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    smd = standardized_mean_diff(attr_array(dep, "sep"), attr_array(rob, "sep"))
    assert smd > 1.0


def test_standardized_mean_diff_nan_when_pooled_sd_zero():
    # 完全分離した定数群（群内分散0）→ 真の SMD は無限大。0.0 ではなく nan を返す。
    dep = np.array([5.0, 5.0, 5.0])
    rob = np.array([1.0, 1.0, 1.0])
    assert math.isnan(standardized_mean_diff(dep, rob))


def test_smd_stderr_matches_closed_form():
    dep, rob = split_dependent_robust(_recs(), "gComp")
    a, b = attr_array(dep, "n_input"), attr_array(rob, "n_input")
    d = standardized_mean_diff(a, b)
    n1, n2 = a.size, b.size
    expected = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    assert abs(smd_stderr(a, b) - expected) < 1e-12
    assert smd_stderr(a, b) > 0


def test_smd_stderr_nan_when_group_too_small():
    one = np.array([1.0])
    assert math.isnan(smd_stderr(one, np.array([2.0, 3.0])))


# --- spearman_drop_sep（FIX C1） ---------------------------------------------

def test_spearman_drop_sep_positive():
    s = spearman_drop_sep(_recs(), "gComp")
    assert s["rho"] > 0.5


def test_spearman_drop_sep_excludes_unsolved_cases():
    # _recs() は base_R==1.0 が8件 + base_R==0.5 が1件。仕様上、後者は除外される。
    s = spearman_drop_sep(_recs(), "gComp")
    assert s["n"] == 8


def test_spearman_drop_sep_counts_cases_not_records():
    # A・B が各2レコード（seed重複）でも n はレコード数(4)ではなくケース数(2)。
    s = spearman_drop_sep(_seed_repeat_recs(), "gComp")
    assert s["n"] == 2


# --- sep_profile（FIX C2 / I2） ----------------------------------------------

def test_sep_profile_bins_increase():
    recs = [{"feature": "gComp", "case_id": f"c{i}", "base_R": 1.0, "flip": 1,
             "sep": s, "drop": s, "n_input": 5, "n_correct": 1,
             "n_output": 1, "n_sources": 1, "variant": "original"}
            for i, s in enumerate(np.linspace(0, 1, 40))]
    xs, ys, es = sep_profile(recs, "gComp", n_bins=4)
    assert len(xs) == 4
    assert ys[-1] > ys[0]            # sep が大きいビンほど drop 大
    assert all(e >= 0 for e in es)


def test_sep_profile_x_uses_bin_median_not_interval_midpoint():
    # 上位ビンに外れ値 (10.0) が1件あるだけで、区間中点は5付近まで歪む。
    # 中央値ならビン内の実測値（0.05〜0.08と10.0の中央=0.07）に留まる。
    seps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 10.0]
    recs = [{"feature": "gComp", "case_id": f"c{i}", "base_R": 1.0, "flip": 0,
             "sep": s, "drop": 0.1, "n_input": 5, "n_correct": 1,
             "n_output": 1, "n_sources": 1, "variant": "original"}
            for i, s in enumerate(seps)]
    xs, ys, es = sep_profile(recs, "gComp", n_bins=2)
    assert len(xs) == 2
    assert abs(xs[1] - 0.07) < 1e-9       # ビン内実測値の中央値
    assert xs[1] < 1.0                    # 区間中点（約5.02）には引きずられない


def test_sep_profile_excludes_unsolved_like_spearman():
    # base_R<1.0 のケースは spearman_drop_sep と同じ母集団になるよう除外される（FIX C2）。
    # 極端な sep/drop を持つ未解決ケースを混ぜても出力が変化しないことを確認する。
    clean = [{"feature": "gComp", "case_id": f"c{i}", "base_R": 1.0, "flip": 0,
              "sep": s, "drop": s, "n_input": 5, "n_correct": 1,
              "n_output": 1, "n_sources": 1, "variant": "original"}
             for i, s in enumerate(np.linspace(0, 1, 10))]
    unsolved = {"feature": "gComp", "case_id": "unsolved", "base_R": 0.4, "flip": 0,
                "sep": 999.0, "drop": -999.0, "n_input": 5, "n_correct": 1,
                "n_output": 1, "n_sources": 1, "variant": "original"}
    xs_a, ys_a, es_a = sep_profile(clean, "gComp", n_bins=2)
    xs_b, ys_b, es_b = sep_profile(clean + [unsolved], "gComp", n_bins=2)
    assert xs_a == xs_b and ys_a == ys_b and es_a == es_b


# --- cancellation_check（FIX C3） ---------------------------------------------

def test_cancellation_check_splits_by_sep_sign():
    recs = [
        {"feature": "gDom", "case_id": "p1", "base_R": 1.0, "flip": 1,
         "sep": 0.4, "drop": 0.30, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "p2", "base_R": 1.0, "flip": 1,
         "sep": 0.5, "drop": 0.34, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "p3", "base_R": 1.0, "flip": 0,
         "sep": 0.2, "drop": 0.26, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "n1", "base_R": 1.0, "flip": 0,
         "sep": -0.4, "drop": -0.20, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "n2", "base_R": 1.0, "flip": 0,
         "sep": -0.3, "drop": -0.24, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "n3", "base_R": 1.0, "flip": 0,
         "sep": -0.5, "drop": -0.28, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
    ]
    res = cancellation_check(recs, "gDom")
    assert res["n_sep_pos"] == 3
    assert res["n_sep_neg"] == 3
    assert res["mean_drop_sep_pos"] > 0
    assert res["mean_drop_sep_neg"] < 0
    assert res["sem_drop_sep_pos"] >= 0
    assert res["sem_drop_sep_neg"] >= 0
    assert 0.0 <= res["mannwhitney_p"] <= 1.0


def test_cancellation_check_nan_p_when_one_side_empty():
    recs = [{"feature": "gDom", "case_id": f"p{i}", "base_R": 1.0, "flip": 0,
             "sep": 0.1 + 0.01 * i, "drop": 0.05, "n_correct": 1, "n_input": 5,
             "n_output": 1, "n_sources": 1, "variant": "original"}
            for i in range(4)]
    res = cancellation_check(recs, "gDom")
    assert res["n_sep_pos"] == 4
    assert res["n_sep_neg"] == 0
    assert math.isnan(res["mannwhitney_p"])
    assert math.isnan(res["mean_drop_sep_neg"])


def _cancellation_populations_recs():
    # solved_only=True と False とで sep<0 群の平均 drop が変わることを示す最小構成。
    # s1: 解決済み(base_R=1.0)・逆分離(sep<0)。実データ同様、drop は非負（ここでは 0.02）。
    # u1: 未解決(base_R<1.0)・逆分離(sep<0)。perm が baseline を上回りうるため drop は
    #     負になれる（ここでは直接 -0.5 を与える）。solved_only=True では除外される。
    # p1: sep>0 側の最小限のダミー（本テストの主眼は neg 側の母集団差）。
    return [
        {"feature": "gDom", "case_id": "s1", "base_R": 1.0, "flip": 0,
         "sep": -0.3, "drop": 0.02, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "u1", "base_R": 0.4, "flip": 0,
         "sep": -0.2, "drop": -0.5, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
        {"feature": "gDom", "case_id": "p1", "base_R": 1.0, "flip": 1,
         "sep": 0.3, "drop": 0.10, "n_correct": 1, "n_input": 5,
         "n_output": 1, "n_sources": 1, "variant": "original"},
    ]


def test_cancellation_check_solved_only_excludes_negative_drop_that_all_cases_includes():
    recs = _cancellation_populations_recs()
    solved = cancellation_check(recs, "gDom", solved_only=True)
    allc = cancellation_check(recs, "gDom", solved_only=False)

    # solved_only=True: u1（未解決）は除外される → sep<0 群は s1 のみ、drop は非負のまま。
    assert solved["n_sep_neg"] == 1
    assert solved["mean_drop_sep_neg"] == 0.02
    assert solved["mean_drop_sep_neg"] >= 0

    # solved_only=False: u1 も含まれる → sep<0 群は s1+u1 の平均になり、負に振れる。
    assert allc["n_sep_neg"] == 2
    assert allc["mean_drop_sep_neg"] == (0.02 + (-0.5)) / 2
    assert allc["mean_drop_sep_neg"] < 0

    # 母集団の選択（solved_only）そのものが結果を変えることを直接示す。
    assert solved["mean_drop_sep_neg"] != allc["mean_drop_sep_neg"]


# --- build_stats ---------------------------------------------------------------

def test_build_stats_has_features_and_attrs():
    recs = _recs()
    st = build_stats(recs, n_seeds=7)
    assert "gComp" in st["features"]
    g = st["features"]["gComp"]
    assert "n_input" in g["attrs"] and "sep" in g["attrs"]
    assert "spearman_drop_sep" in g
    assert g["n_dependent"] == 4 and g["n_robust"] == 4
    assert g["n_cases"] == 8
    assert "cancellation" in g
    assert "cancellation_all_cases" in g
    assert st["n_seeds"] == 7
