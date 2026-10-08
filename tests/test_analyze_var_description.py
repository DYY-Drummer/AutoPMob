"""analyze_var_description.py のテスト.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.3, §5
補助関数（quantiles・ci95・verdict・paired_with_ci など）と cmd_precheck・cmd_stats・cmd_figure を、
tmp_path に作った合成データで確かめる。実データ・E5 の埋め込みには触れない。
"""
import json
import math
import statistics
import sys
import warnings

import numpy as np
import pytest

import analyze_var_description as avd
from analyze_var_description import AUC_GATE, quantiles

T975_DF9 = 2.2621571628  # t 分布の上側 2.5% 点（自由度 9）。n = 10 の信頼区間の係数（手計算の照合用）


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """どのテストも repo の experiments/・figures/ に書かない（出力先を tmp_path に向ける）."""
    monkeypatch.setattr(avd, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(avd, "FIG_DIR", tmp_path / "figures")


def test_quantiles_and_gate():
    q = quantiles(np.arange(101, dtype=float))
    assert q["mean"] == 50.0 and q["p5"] == 5.0 and q["p50"] == 50.0 and q["p95"] == 95.0
    assert AUC_GATE == 0.80


from analyze_var_description import ci95, verdict, family


def test_ci95_and_verdict():
    d = [0.01, -0.01, 0.0, 0.005, -0.005, 0.0, 0.01, -0.01, 0.0, 0.0]
    lo, hi = ci95(d)
    assert -0.02 <= lo < 0 < hi <= 0.02
    assert verdict({"p_ttest": 0.5, "p_wilcoxon": 0.5, "mean_delta": 0.0}, [lo, hi]).startswith("equivalent")
    assert verdict({"p_ttest": 0.01, "p_wilcoxon": 0.002, "mean_delta": -0.05}, [-0.07, -0.03]) == "a lower (significant)"
    assert verdict({"p_ttest": 0.01, "p_wilcoxon": 0.002, "mean_delta": 0.05}, [0.03, 0.07]) == "a higher (significant)"
    assert verdict({"p_ttest": 0.2, "p_wilcoxon": 0.1, "mean_delta": -0.03}, [-0.08, 0.02]) == "inconclusive"


def test_family():
    assert family("original") == "single"
    assert family("multisource_v3") == "cross" and family("multisource_random_io") == "cross"
    assert family("dae_X3") == "synth"


# ---- ci95・verdict ----

def test_ci95_exact_interval():
    d = [0.01, -0.01, 0.0, 0.005, -0.005, 0.0, 0.01, -0.01, 0.0, 0.0]
    assert ci95(d) == [-0.0051, 0.0051]


def test_ci95_without_digits_is_not_rounded():
    d = [0.01, -0.01, 0.0, 0.005, -0.005, 0.0, 0.01, -0.01, 0.0, 0.0]  # 平均 0、標本 SD √5e-5
    half = T975_DF9 * math.sqrt(5e-5) / math.sqrt(10)  # 手計算の半幅 0.00505834…
    lo, hi = ci95(d, digits=None)
    assert hi == pytest.approx(half, rel=1e-9) and lo == pytest.approx(-half, rel=1e-9)
    assert [lo, hi] != ci95(d)  # 既定の 4 桁（[-0.0051, 0.0051]）には丸めない
    assert ci95(d, digits=6) == [-0.005058, 0.005058]
    assert ci95([0.1], digits=None) == [None, None]  # 1 件では定義できない


def test_verdict_needs_both_tests():
    ci = [-0.07, -0.03]
    assert verdict({"p_ttest": 0.01, "p_wilcoxon": 0.2, "mean_delta": -0.05}, ci) == "inconclusive"
    assert verdict({"p_ttest": 0.2, "p_wilcoxon": 0.01, "mean_delta": -0.05}, ci) == "inconclusive"


def test_verdict_adds_significance_to_equivalent():
    lower = verdict({"p_ttest": 0.01, "p_wilcoxon": 0.002, "mean_delta": -0.01}, [-0.015, -0.005])
    assert lower == "equivalent: 95% CI within ±0.02; significant (a lower)"
    higher = verdict({"p_ttest": 0.01, "p_wilcoxon": 0.002, "mean_delta": 0.01}, [0.005, 0.015])
    assert higher == "equivalent: 95% CI within ±0.02; significant (a higher)"
    # 片方の検定だけが p < 0.05 なら有意とは書かない
    one = verdict({"p_ttest": 0.01, "p_wilcoxon": 0.2, "mean_delta": -0.01}, [-0.015, -0.005])
    assert one == "equivalent: 95% CI within ±0.02"


def test_verdict_when_tests_are_undefined():
    st = {"p_ttest": None, "p_wilcoxon": None, "mean_delta": 0.03}
    assert verdict(st, [0.03, 0.03]) == "inconclusive (tests undefined)"
    assert verdict({**st, "mean_delta": 0.0}, [0.0, 0.0]) == "equivalent: 95% CI within ±0.02"
    # 区間も定義できない（1 組だけ）ときも、検定がないので結論は出さない
    assert verdict(st, [None, None]) == "inconclusive (tests undefined)"


def test_verdict_equivalence_margin_is_inclusive_at_exactly_0_02():
    st = {"p_ttest": 0.5, "p_wilcoxon": 0.5, "mean_delta": 0.0}
    assert verdict(st, [-0.02, 0.02]) == "equivalent: 95% CI within ±0.02"  # 丸める前の区間がちょうど ±0.02 なら収まる
    assert verdict(st, [-0.02, 0.0200001]) == "inconclusive"  # 上端が 1e-7 超えれば収まらない
    assert verdict(st, [-0.0200001, 0.02]) == "inconclusive"  # 下端も同じ


# ---- per_seed・case_seed_means ----

def test_per_seed_returns_seeds_sorted_by_the_data():
    res = {"X": {"per_seed": [{"seed": 9, "MAP": 0.3}, {"seed": 1, "MAP": 0.1}, {"seed": 5, "MAP": 0.2}]}}
    assert avd.per_seed(res, "X", "MAP") == ([1, 5, 9], [0.1, 0.2, 0.3])  # SEEDS にない seed でも並べる


def test_per_seed_rejects_duplicate_seeds():
    res = {"X": {"per_seed": [{"seed": 1, "MAP": 0.1}, {"seed": 1, "MAP": 0.2}]}}
    with pytest.raises(ValueError, match="重複"):
        avd.per_seed(res, "X", "MAP")


CASES = [
    {"seed": 1, "case_id": "a", "variant": "original", "Recall@K_correct": 0.2},
    {"seed": 1, "case_id": "b", "variant": "dae_X3", "Recall@K_correct": 0.6},
    {"seed": 2, "case_id": "a", "variant": "original", "Recall@K_correct": 0.4},
]


def test_case_seed_means_filters_by_exclusion_and_family():
    res = {"X": {"per_case": CASES}}
    assert avd.case_seed_means(res, "X") == pytest.approx({1: 0.4, 2: 0.4})
    assert avd.case_seed_means(res, "X", exclude={"a"}) == pytest.approx({1: 0.6})
    assert avd.case_seed_means(res, "X", fam="single") == pytest.approx({1: 0.2, 2: 0.4})
    fm = avd.family_means(res, "X")
    assert sorted(fm) == ["single", "synth"]
    assert fm["single"] == pytest.approx({1: 0.2, 2: 0.4}) and fm["synth"] == pytest.approx({1: 0.6})
    assert sorted(avd.family_means(res, "X", exclude={"b"})) == ["single"]  # 族のケースが全部落ちたら族ごと消える


def test_case_seed_means_needs_per_case():
    with pytest.raises(ValueError, match="per_case"):
        avd.case_seed_means({"X": {"per_seed": []}}, "X")


# ---- paired_with_ci ----

def test_paired_with_ci_direction_and_interval():
    vb = [0.70 + 0.01 * i for i in range(10)]
    jitter = [0.001, -0.001, 0.002, -0.002, 0.0015, -0.0015, 0.0005, -0.0005, 0.0025, -0.0025]
    va = [b + 0.03 + j for b, j in zip(vb, jitter)]
    r = avd.paired_with_ci(va, vb)  # 差は a − b
    assert r["mean_delta"] == pytest.approx(0.03, abs=1e-4)
    lo, hi = r["ci95"]
    assert lo < 0.03 < hi and (lo + hi) / 2 == pytest.approx(0.03, abs=1e-4)
    assert r["significant"] is True and r["verdict"] == "a higher (significant)"
    rev = avd.paired_with_ci(vb, va)
    assert rev["mean_delta"] == pytest.approx(-0.03, abs=1e-4)
    assert rev["significant"] is True and rev["verdict"] == "a lower (significant)"


def test_paired_with_ci_marks_non_significant_as_false():
    vb = [0.70 + 0.01 * i for i in range(10)]
    va = [b + d for b, d in zip(vb, [0.004, -0.003, 0.002, -0.004, 0.003, -0.002, 0.001, -0.001, 0.0035, -0.0025])]
    r = avd.paired_with_ci(va, vb)
    assert r["significant"] is False and r["verdict"].startswith("equivalent")
    assert "significant" not in r["verdict"]


@pytest.mark.parametrize("sign", [1, -1], ids=["a-higher", "a-lower"])
@pytest.mark.parametrize("half, inside", [(0.00503, False), (0.00497, True)],
                         ids=["exact-edge-0.02003-outside", "exact-edge-0.01997-inside"])
def test_verdict_is_decided_on_the_unrounded_ci(sign, half, inside):
    """差の平均 ±0.015、95% 信頼区間の半幅 half のデータ。

    厳密な端は ±0.02003（外）か ±0.01997（内）。4 桁に丸めるとどちらも ±0.02 になるので、
    丸めた区間で判定すると、外のデータまで「同程度」になってしまう."""
    n = 10
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)  # 平均 0、標本 SD 1
    d = sign * (0.015 + half * math.sqrt(n) / T975_DF9 * z)  # 半幅 = t × SD / √n が half になる SD
    vb = [0.70 + 0.01 * i for i in range(n)]
    va = [b + x for b, x in zip(vb, d)]
    r = avd.paired_with_ci(va, vb)
    exact = ci95(np.array(va) - np.array(vb), digits=None)
    assert exact == pytest.approx(sorted([sign * 0.015 - half, sign * 0.015 + half]), abs=1e-9)  # データの作りの確認
    assert (max(abs(x) for x in exact) <= 0.02) == inside  # 厳密な区間は、外のデータでは ±0.02 を超える
    assert r["ci95"] == sorted([sign * 0.01, sign * 0.02])  # 保存するのは丸めた区間（どちらも ±0.02 に収まって見える）
    direction = "a higher" if sign > 0 else "a lower"
    expected = f"equivalent: 95% CI within ±0.02; significant ({direction})" if inside else f"{direction} (significant)"
    assert r["verdict"] == expected


@pytest.mark.parametrize("shift", [0.0, 0.03])
def test_paired_with_ci_identical_differences_give_none_not_nan(shift):
    vb = [0.70 + 0.01 * i for i in range(10)]
    r = avd.paired_with_ci([b + shift for b in vb], vb)
    assert r["p_ttest"] is None and r["p_wilcoxon"] is None and r["cohen_dz"] is None
    assert r["std_delta"] == 0.0 and r["significant"] is None
    assert "undefined" in r["note"]
    assert r["mean_delta"] == pytest.approx(shift, abs=1e-9) and r["n"] == 10
    json.dumps(r, allow_nan=False)  # NaN・inf が残っていれば ValueError
    assert r["verdict"] == ("equivalent: 95% CI within ±0.02" if shift == 0.0 else "inconclusive (tests undefined)")


def test_differences_equal_up_to_float_noise_count_as_identical_without_warnings():
    vb = [0.3 + 0.1 * i for i in range(10)]
    va = [x + 0.03 for x in vb]
    d = [a - b for a, b in zip(va, vb)]
    assert 0 < max(d) - min(d) < 1e-12  # 厳密には同じでない（浮動小数点の丸め ~1e-16 だけ違う）
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # scipy の「ほぼ同一のデータ」の RuntimeWarning が出れば失敗
        r = avd.paired_with_ci(va, vb)
    assert r["p_ttest"] is None and r["p_wilcoxon"] is None and r["cohen_dz"] is None
    assert r["significant"] is None and "undefined" in r["note"]
    assert r["mean_delta"] == pytest.approx(0.03, abs=1e-9) and r["verdict"] == "inconclusive (tests undefined)"
    json.dumps(r, allow_nan=False)


def test_differences_that_vary_beyond_float_noise_are_still_tested():
    vb = [0.3 + 0.1 * i for i in range(10)]
    va = [x + 0.03 + (1e-6 if i == 4 else 0.0) for i, x in enumerate(vb)]  # 1 組だけ 1e-6 違う = 丸めでなく本物の差
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = avd.paired_with_ci(va, vb)
    assert r["p_ttest"] is not None and r["p_wilcoxon"] is not None and "note" not in r
    assert r["significant"] is True and r["verdict"] == "a higher (significant)"


def test_paired_with_ci_single_pair_has_no_nan():
    r = avd.paired_with_ci([0.72], [0.70])
    assert r["p_ttest"] is None and r["p_wilcoxon"] is None and r["cohen_dz"] is None
    assert r["ci95"] == [None, None] and r["verdict"] == "inconclusive (tests undefined)"
    json.dumps(r, allow_nan=False)


def test_safe_paired_turns_leftover_nan_and_inf_into_none(monkeypatch):
    import analyze_significance
    fake = {"n": 3, "mean_base": 0.1, "mean_rer": 0.2, "mean_delta": 0.1, "std_delta": float("nan"),
            "p_ttest": float("nan"), "p_wilcoxon": 0.25, "cohen_dz": float("inf")}
    monkeypatch.setattr(analyze_significance, "paired_stats", lambda base, rer: dict(fake))
    st = avd.safe_paired([0.1, 0.2, 0.4], [0.2, 0.2, 0.5])
    assert st["std_delta"] is None and st["p_ttest"] is None and st["cohen_dz"] is None
    assert st["p_wilcoxon"] == 0.25 and st["mean_delta"] == 0.1 and st["n"] == 3
    json.dumps(st, allow_nan=False)


def test_write_json_is_strict_and_leaves_no_partial_file(tmp_path):
    ok = tmp_path / "ok.json"
    avd.write_json(ok, {"x": [1, None], "label": "±"})
    assert json.loads(ok.read_text(encoding="utf-8")) == {"x": [1, None], "label": "±"}
    bad = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        avd.write_json(bad, {"x": {"p": float("nan")}})
    assert not bad.exists()  # 書き始める前に失敗するので、途中までのファイルが残らない


def test_paired_with_ci_rejects_broken_inputs():
    with pytest.raises(ValueError):
        avd.paired_with_ci([0.7, 0.8], [0.7])  # 組の数が違う
    with pytest.raises(ValueError):
        avd.paired_with_ci([], [])
    with pytest.raises(ValueError, match="NaN"):  # NaN を None に直して黙って通さない
        avd.paired_with_ci([0.7, float("nan"), 0.8], [0.7, 0.7, 0.7])


# ---- 合成データ（設定 A・B の結果ファイル） ----

SEEDS10 = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
LABELS = ["S0", "S1", "S1-r50", "S1-r100", "D1", "D0"]
# R@K の水準: D1 は S1 より 0.005 低い（同程度）、S1-r は記号を替えるほど下がる、D0 は漏れで高い
LEVEL = {"S0": 0.740, "S1": 0.742, "S1-r50": 0.700, "S1-r100": 0.660, "D1": 0.737, "D0": 0.770}
AMP = {"S0": 0.003, "S1": 0.003, "S1-r50": 0.002, "S1-r100": 0.002, "D1": 0.001, "D0": 0.002}
VARIANT = ["original", "original", "multisource_v3", "multisource_v3", "multisource_random_io",
           "dae_X3", "dae_X3", "dae_X1"]
SETTING_SHIFT = {"A": 0.0, "B": -0.03}  # 設定 B は A より R@K が 0.03 低い（ファイルの取り違えが分かる）


def synthetic_results(shift: float = 0.0) -> dict:
    """results[label] = {per_seed（10 seed）, per_case（seed ごとに 5 ケース）}。乱数は使わず三角関数でゆらがせる."""
    out = {}
    for li, lab in enumerate(LABELS):
        per_seed, per_case = [], []
        for si, seed in enumerate(SEEDS10):
            rk = LEVEL[lab] + shift + 0.01 * math.sin(si + 1) + AMP[lab] * math.sin(2.1 * si + 1.7 * li)
            per_seed.append({"seed": seed, "Recall@K_correct": rk,
                             "MAP": rk - 0.20 + 0.002 * math.cos(1.1 * si + li),
                             "Recall@20": rk + 0.12 + 0.002 * math.cos(0.9 * si + 2 * li)})
            for j in range(5):
                k = (si + j) % 8
                v = rk + 0.05 * math.sin(1.9 * k + 0.8 * si + 2.3 * li)
                per_case.append({"seed": seed, "case_id": f"case_{k}", "variant": VARIANT[k],
                                 "Recall@K_correct": min(1.0, max(0.0, v))})
        out[lab] = {"mode": lab, "per_seed": per_seed, "per_case": per_case}
    return out


def dump(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def load_strict(path) -> dict:
    """NaN・Infinity を含む JSON は受け付けない読み込み."""
    def reject(const):
        raise ValueError(f"not strict JSON: {const}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_experiment_files(exp, drop_b=(), conflict_ratio=0.1, qc_failed=()) -> None:
    """stats・figure が読むファイルを exp に書く（設定 A・B、qc、説明、τ 感度の 2 本）.

    qc_failed: 説明ファイルで qc_failed を立てる case_id。本物の finalize と同じく、どのレコードにも qc_failed を書く."""
    main = ["S0", "S1", "D1", "D0"]
    for setting, shift in SETTING_SHIFT.items():
        res = synthetic_results(shift)
        for lab in drop_b if setting == "B" else ():
            res.pop(lab)
        dump(exp / f"var_desc_{setting}.json", {
            "config": {"setting": setting},
            "tau": {lab: {str(s): 1.0 if lab in ("S0", "S1") else 0.82 for s in SEEDS10} for lab in main},
            "approx_mismatch": {lab: {str(s): 0.01 for s in SEEDS10} for lab in main},
            "results": res})
    dump(exp / "var_descriptions_qc.json", {"meaning_conflict_ratio": conflict_ratio})
    variables = {"T": {"paraphrase": "temperature", "alt_symbol": "q", "meaning_conflict": False}}
    cases = {"case_0": {"context_nosym": "x", "variables": variables},
             "case_3": {"context_nosym": "x", "variables": {
                 **variables, "F": {"paraphrase": "flow", "alt_symbol": "u", "meaning_conflict": True}}}}
    for cid in qc_failed:  # per_case にない case_id（学習側だけのケース）も書ける
        cases.setdefault(cid, {"context_nosym": "x", "variables": variables})
    for cid, c in cases.items():
        c["qc_failed"] = cid in qc_failed
    dump(exp / "var_descriptions.json", {"meta": {}, "cases": cases})
    for d, rk in (("-0.02", 0.731), ("+0.02", 0.735)):
        dump(exp / f"var_desc_tau{d}.json", {"results": {f"D1-tau{d}": {"per_seed": [
            {"seed": 42, "Recall@K_correct": rk, "MAP": rk - 0.2, "Recall@20": rk + 0.1}]}}})


# ---- excluding_conflict・family_tests ----

def test_excluding_conflict_counts_the_cases_actually_dropped():
    res = synthetic_results()
    conflict = {"case_3", "train_only_1", "train_only_2"}  # 学習側にしかないケースは落ちない
    out = avd.excluding_conflict(res, conflict)
    seeds_with_case_3 = {r["seed"] for r in res["D1"]["per_case"] if r["case_id"] == "case_3"}
    assert 0 < len(seeds_with_case_3) < len(SEEDS10)  # 落ちる seed と落ちない seed がある
    assert out["n_excluded_cases"] == 1 != len(conflict)
    assert out["n_excluded_cases_per_seed"] == {str(s): int(s in seeds_with_case_3) for s in SEEDS10}
    st = out["D1_vs_S1"]
    assert st["n"] == 10 and st["verdict"] and st["significant"] in (True, False)


def seed_means(res, label, skip=frozenset(), variants=None):
    """label の per_case を seed ごとに平均した R@K のリスト（seed の昇順）.

    skip の case_id と、variants にない variant は除く。数え方は本体（case_seed_means・family）とは別."""
    by = {}
    for r in res[label]["per_case"]:
        if r["case_id"] in skip or (variants is not None and r["variant"] not in variants):
            continue
        by.setdefault(r["seed"], []).append(r["Recall@K_correct"])
    return [statistics.fmean(by[s]) for s in sorted(by)]


def test_excluding_conflict_statistics_match_an_independent_computation():
    res = synthetic_results()
    conflict = {"case_3", "train_only_1"}
    st = avd.excluding_conflict(res, conflict)["D1_vs_S1"]
    d1, s1 = seed_means(res, "D1", conflict), seed_means(res, "S1", conflict)
    assert st["n"] == len(d1) == len(s1) == 10
    assert st["mean_base"] == pytest.approx(statistics.fmean(s1), abs=1e-4)  # base は S1
    assert st["mean_rer"] == pytest.approx(statistics.fmean(d1), abs=1e-4)  # rer は D1
    assert st["mean_delta"] == pytest.approx(statistics.fmean(d1) - statistics.fmean(s1), abs=1e-4)  # 差は D1 − S1
    # 除外が効いている（除かない平均とは 5e-4 以上違う）ので、除き忘れは許容誤差 1e-4 で見分けられる
    for label, kept in (("D1", d1), ("S1", s1)):
        assert abs(statistics.fmean(seed_means(res, label)) - statistics.fmean(kept)) > 5e-4


def test_excluding_conflict_names_the_reason_in_the_seed_mismatch_error():
    res = synthetic_results()
    res["D1"]["per_case"] = [r for r in res["D1"]["per_case"] if r["seed"] != 42]  # D1 だけ seed 42 がない
    with pytest.raises(ValueError, match=r"D1 vs S1 \(meaning_conflict を除く\)"):  # 既定は従来どおり
        avd.excluding_conflict(res, {"case_3"})
    with pytest.raises(ValueError, match=r"D1 vs S1 \(qc_failed を除く\)"):
        avd.excluding_conflict(res, {"case_3"}, reason="qc_failed")
    ok = synthetic_results()
    assert avd.excluding_conflict(ok, {"case_3"}, reason="qc_failed") == avd.excluding_conflict(ok, {"case_3"})  # 値は変わらない


def test_family_tests_have_verdicts_and_keep_each_family_separate():
    out = avd.family_tests(synthetic_results())["D1_vs_S1"]
    assert sorted(out) == ["cross", "single", "synth"]
    for st in out.values():
        assert st["verdict"] and "ci95" in st and st["significant"] in (True, False)
    assert out["single"]["n"] == 8 and out["cross"]["n"] == 9  # その族のテストケースがある seed だけ


@pytest.mark.parametrize("label, fam, variants", [
    ("D1", "single", {"original"}),
    ("S1", "cross", {"multisource_v3", "multisource_random_io"}),
    ("D0", "synth", {"dae_X3", "dae_X1"}),
], ids=["D1-single", "S1-cross", "D0-synth"])
def test_family_table_has_mean_sd_and_seed_count_per_family(label, fam, variants):
    res = synthetic_results()
    tab = avd.family_table(res)
    assert sorted(tab) == sorted(LABELS) and sorted(tab[label]) == ["cross", "single", "synth"]
    v = seed_means(res, label, variants=variants)  # 族のテストケースがある seed ごとの平均
    assert abs(statistics.pstdev(v) - statistics.stdev(v)) > 1e-3  # ddof の取り違えを許容誤差 1e-4 で見分けられる
    assert tab[label][fam]["n_seeds"] == len(v) and len(v) < len(SEEDS10)  # テストケースがない seed は数えない
    assert tab[label][fam]["mean"] == pytest.approx(statistics.fmean(v), abs=1e-4)
    assert tab[label][fam]["sd"] == pytest.approx(statistics.stdev(v), abs=1e-4)  # 標本 SD（ddof=1）


def test_family_table_sd_is_zero_for_a_single_seed():
    res = {"X": {"per_case": [{"seed": 1, "case_id": "a", "variant": "original", "Recall@K_correct": 0.2}]}}
    assert avd.family_table(res) == {"X": {"single": {"mean": 0.2, "sd": 0.0, "n_seeds": 1}}}  # NaN にしない


def test_family_tests_reject_mismatched_seed_sets():
    res = synthetic_results()
    res["D1"]["per_case"] = [r for r in res["D1"]["per_case"] if r["seed"] != 42]
    with pytest.raises(ValueError, match="seed の集合"):
        avd.family_tests(res)


def test_compare_raises_when_seed_sets_differ():
    res = synthetic_results()
    res["D1"]["per_seed"] = res["D1"]["per_seed"][:9]   # 42〜7777
    res["S1"]["per_seed"] = res["S1"]["per_seed"][1:]   # 123〜9999: 件数は同じでも seed が違う
    with pytest.raises(ValueError, match="seed の集合"):
        avd.compare(res, "D1", "S1", "x")


# ---- cmd_stats（var_descriptions.json は tmp_path に書いた本物の形式のファイルを、本体の load_json で読む） ----

def test_cmd_stats_end_to_end(capsys):
    write_experiment_files(avd.EXP)
    avd.cmd_stats(None)
    printed = capsys.readouterr().out
    stats = load_strict(avd.EXP / "var_desc_stats.json")  # NaN があればここで落ちる

    keys = [f"{a}_vs_{b}" for a, b, _ in avd.COMPARISONS]
    assert len(keys) == 6
    for setting in "AB":
        comps = stats[setting]["comparisons"]
        assert list(comps) == keys
        for (a, b, _), key in zip(avd.COMPARISONS, keys):
            assert f"[{setting}] {a} − {b}:" in printed  # 比較ごとに 1 行
            for m in avd.METRICS:
                assert comps[key][m]["verdict"] and comps[key][m]["significant"] in (True, False)
                assert len(comps[key][m]["ci95"]) == 2
        # 合成データに仕込んだ効果の向き
        v = {k: comps[k]["Recall@K_correct"]["verdict"] for k in keys}
        assert v["D1_vs_S1"] == "equivalent: 95% CI within ±0.02; significant (a lower)"
        assert v["S1-r50_vs_S1"] == "a lower (significant)"
        assert v["S1-r100_vs_S1"] == "a lower (significant)"
        assert v["D1_vs_S1-r100"] == "a higher (significant)"
        assert v["D0_vs_D1"] == "a higher (significant)"
        assert v["S1_vs_S0"] == "equivalent: 95% CI within ±0.02"
        assert set(stats[setting]["family_tests"]["D1_vs_S1"]) == {"single", "cross", "synth"}
        assert all(t["verdict"] for t in stats[setting]["family_tests"]["D1_vs_S1"].values())

    # 設定 B は A より R@K が 0.03 低い（ファイルの取り違えがあれば平均が合わない）
    base = {s: stats[s]["comparisons"]["D1_vs_S1"]["Recall@K_correct"]["mean_base"] for s in "AB"}
    assert base["A"] - base["B"] == pytest.approx(0.03, abs=1e-3)
    assert stats["meaning_conflict"] == {"ratio": 0.1, "n_cases_with_conflict": 1, "exclusion_applied": True}
    dropped = {r["seed"] for r in synthetic_results()["D1"]["per_case"] if r["case_id"] == "case_3"}
    for setting in "AB":
        ex = stats[setting]["excluding_meaning_conflict"]
        assert ex["n_excluded_cases"] == 1
        assert ex["n_excluded_cases_per_seed"] == {str(s): int(s in dropped) for s in SEEDS10}
        assert ex["D1_vs_S1"]["verdict"]
    assert set(stats["A"]["tau_sensitivity_seed42"]) == {"-0.02", "+0.02", "0"}
    assert stats["A"]["tau_sensitivity_seed42"]["-0.02"]["Recall@K_correct"] == 0.731
    assert stats["A"]["tau_sensitivity_seed42"]["0"]["MAP"] == pytest.approx(
        synthetic_results()["D1"]["per_seed"][0]["MAP"])


@pytest.mark.parametrize("ratio, applied", [(0.03, False), (0.05, False), (0.0501, True)],
                         ids=["ratio-0.03", "ratio-exactly-0.05", "ratio-0.0501"])
def test_cmd_stats_excludes_only_when_the_conflict_ratio_is_above_5_percent(ratio, applied):
    write_experiment_files(avd.EXP, conflict_ratio=ratio)
    avd.cmd_stats(None)
    stats = load_strict(avd.EXP / "var_desc_stats.json")
    assert stats["meaning_conflict"]["ratio"] == ratio  # ちょうど 0.05 は「超えていない」
    assert stats["meaning_conflict"]["exclusion_applied"] is applied
    for setting in "AB":
        assert ("excluding_meaning_conflict" in stats[setting]) is applied


def test_cmd_stats_reports_a_missing_label(capsys):
    write_experiment_files(avd.EXP, drop_b=("S1-r50",))
    avd.cmd_stats(None)
    printed = capsys.readouterr().out
    assert "[B] S1-r50 − S1: missing label" in printed
    assert "[A] S1-r50 − S1: missing label" not in printed
    assert "[B] D1 − S1:" in printed  # ほかの比較は続ける
    entry = load_strict(avd.EXP / "var_desc_stats.json")["B"]["comparisons"]["S1-r50_vs_S1"]
    assert "missing label" in entry["error"] and "S1-r50" in entry["error"]


def test_cmd_stats_writes_null_for_a_zero_variance_difference(capsys):
    write_experiment_files(avd.EXP)
    path = avd.EXP / "var_desc_B.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["results"]["S1-r50"]["per_seed"] = doc["results"]["S1"]["per_seed"]  # S1-r50 − S1 は全 seed で差が 0
    path.write_text(json.dumps(doc), encoding="utf-8")
    avd.cmd_stats(None)  # 以前は NaN が出て、厳密な JSON として書けなかった
    printed = capsys.readouterr().out
    assert "[B] S1-r50 − S1: Δ=+0.0000 CI=[0.0, 0.0] p_t=n/a p_w=n/a → equivalent: 95% CI within ±0.02" in printed
    entry = load_strict(avd.EXP / "var_desc_stats.json")["B"]["comparisons"]["S1-r50_vs_S1"]
    for m in avd.METRICS:
        st = entry[m]
        assert st["p_ttest"] is None and st["p_wilcoxon"] is None and st["cohen_dz"] is None
        assert st["std_delta"] == 0.0 and st["significant"] is None and "undefined" in st["note"]


def test_cmd_stats_without_d1_still_reports_the_rest(capsys):
    write_experiment_files(avd.EXP)
    path = avd.EXP / "var_desc_B.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["results"]["D1"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    avd.cmd_stats(None)
    printed = capsys.readouterr().out
    for a, b in (("D1", "S1"), ("D1", "S1-r100"), ("D0", "D1")):
        assert f"[B] {a} − {b}: missing label (D1)" in printed
    assert "[B] S1-r50 − S1: Δ=" in printed
    assert "[B] D1 − S1 の族別・meaning_conflict を除いた比較: missing label" in printed
    stats = load_strict(avd.EXP / "var_desc_stats.json")
    assert stats["B"]["family_tests"] == {} and "excluding_meaning_conflict" not in stats["B"]
    assert stats["A"]["family_tests"] and "excluding_meaning_conflict" in stats["A"]


def test_cmd_stats_says_when_a_tau_run_is_missing(capsys):
    write_experiment_files(avd.EXP)
    (avd.EXP / "var_desc_tau+0.02.json").unlink()
    avd.cmd_stats(None)
    assert "[A] tau+0.02: var_desc_tau+0.02.json がない" in capsys.readouterr().out  # 黙って省かない
    assert set(load_strict(avd.EXP / "var_desc_stats.json")["A"]["tau_sensitivity_seed42"]) == {"-0.02", "0"}


def test_cmd_stats_does_not_swallow_a_seed_mismatch():
    write_experiment_files(avd.EXP)
    path = avd.EXP / "var_desc_B.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["results"]["D1"]["per_seed"][0]["seed"] = 43
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="seed の集合"):
        avd.cmd_stats(None)
    assert not (avd.EXP / "var_desc_stats.json").exists()  # 壊れた入力から出力を作らない


# ---- cmd_stats: QC を通らなかったケース（qc_failed）を除いた比較 ----

QC_PAIRS = [("D1", "S1"), ("S1-r50", "S1"), ("S1-r100", "S1"), ("D1", "S1-r100")]
QC_FAILED_IDS = ("case_1", "case_4", "case_6", "train_only_9")  # 最後の 1 件はテストケースにない（学習側だけ）
QC_IN_TEST = {"case_1", "case_4", "case_6"}  # 上のうち per_case にあるもの。meaning_conflict の case_3 とは別


@pytest.mark.parametrize("ratio", [0.0, 0.05], ids=["ratio-0", "ratio-exactly-0.05"])
def test_cmd_stats_always_reports_excluding_qc_failed(ratio, capsys):
    """meaning_conflict の比率が 5% 以下でその除外を出さないときも、qc_failed の除外は出す（比率のゲートがない）."""
    write_experiment_files(avd.EXP, conflict_ratio=ratio, qc_failed=QC_FAILED_IDS)
    avd.cmd_stats(None)
    printed = capsys.readouterr().out
    stats = load_strict(avd.EXP / "var_desc_stats.json")
    assert stats["meaning_conflict"]["exclusion_applied"] is False
    assert stats["qc_failed"] == {"n_cases": len(QC_FAILED_IDS)}  # 説明ファイルの qc_failed の数（テストケースにない 1 件も数える）
    keys = [f"{a}_vs_{b}" for a, b in QC_PAIRS]
    for setting in "AB":
        assert "excluding_meaning_conflict" not in stats[setting]
        ex = stats[setting]["excluding_qc_failed"]
        assert list(ex) == keys
        for (a, b), key in zip(QC_PAIRS, keys):
            assert ex[key]["n_excluded_cases"] == len(QC_IN_TEST) < len(QC_FAILED_IDS)  # 実際に落ちた数
            assert ex[key][key]["verdict"] and len(ex[key][key]["ci95"]) == 2
            assert f"[{setting}] {a} − {b} excluding qc_failed (n_excluded_cases={len(QC_IN_TEST)}):" in printed


def test_cmd_stats_excluding_qc_failed_matches_an_independent_computation():
    from scipy import stats as sps
    write_experiment_files(avd.EXP, qc_failed=QC_FAILED_IDS)  # 既定の conflict_ratio=0.1 なので meaning_conflict の除外も同時に走る
    avd.cmd_stats(None)
    stats = load_strict(avd.EXP / "var_desc_stats.json")
    for setting, shift in SETTING_SHIFT.items():
        res = synthetic_results(shift)
        for a, b in QC_PAIRS:
            key = f"{a}_vs_{b}"
            ex = stats[setting]["excluding_qc_failed"][key]
            da, db = seed_means(res, a, QC_IN_TEST), seed_means(res, b, QC_IN_TEST)  # 両ラベルの per_case から落としてから seed ごとに平均
            st = ex[key]
            assert st["n"] == len(da) == len(db) == 10
            assert st["mean_rer"] == pytest.approx(statistics.fmean(da), abs=1e-4)  # rer は a
            assert st["mean_base"] == pytest.approx(statistics.fmean(db), abs=1e-4)  # base は b
            assert st["mean_delta"] == pytest.approx(statistics.fmean(da) - statistics.fmean(db), abs=1e-4)  # 差は a − b
            assert st["p_ttest"] == pytest.approx(sps.ttest_rel(da, db).pvalue, abs=1e-5)
            # 除外が効いている（除かない平均・meaning_conflict だけを除いた平均とは 4e-3 以上違う）ので、取り違えは許容誤差 1e-4 で見分けられる
            for label, kept in ((a, da), (b, db)):
                assert abs(statistics.fmean(seed_means(res, label)) - statistics.fmean(kept)) > 4e-3
                assert abs(statistics.fmean(seed_means(res, label, {"case_3"})) - statistics.fmean(kept)) > 4e-3
            per_seed = {str(s): len({r["case_id"] for r in res[a]["per_case"] if r["seed"] == s} & QC_IN_TEST)
                        for s in SEEDS10}
            assert ex["n_excluded_cases_per_seed"] == per_seed and ex["n_excluded_cases"] == len(QC_IN_TEST)


def test_cmd_stats_excluding_qc_failed_reports_a_missing_label(capsys):
    write_experiment_files(avd.EXP, drop_b=("S1-r100",), qc_failed=QC_FAILED_IDS)
    avd.cmd_stats(None)  # 例外にしない
    printed = capsys.readouterr().out
    stats = load_strict(avd.EXP / "var_desc_stats.json")
    ex = stats["B"]["excluding_qc_failed"]
    assert ex["S1-r100_vs_S1"] == {"error": "missing label: S1-r100"}
    assert ex["D1_vs_S1-r100"] == {"error": "missing label: S1-r100"}
    for key in ("D1_vs_S1", "S1-r50_vs_S1"):  # ほかの対は続ける
        assert ex[key]["n_excluded_cases"] == len(QC_IN_TEST) and ex[key][key]["verdict"]
    assert "[B] S1-r100 − S1 excluding qc_failed: missing label (S1-r100)" in printed
    assert "[B] D1 − S1-r100 excluding qc_failed: missing label (S1-r100)" in printed
    assert f"[B] D1 − S1 excluding qc_failed (n_excluded_cases={len(QC_IN_TEST)}):" in printed
    assert all("error" not in v for v in stats["A"]["excluding_qc_failed"].values())  # 設定 A は全部そろっている


def test_cmd_stats_excluding_qc_failed_is_reported_even_without_d1():
    write_experiment_files(avd.EXP, qc_failed=QC_FAILED_IDS)
    path = avd.EXP / "var_desc_B.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["results"]["D1"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    avd.cmd_stats(None)
    ex = load_strict(avd.EXP / "var_desc_stats.json")["B"]["excluding_qc_failed"]
    assert ex["D1_vs_S1"] == {"error": "missing label: D1"} and ex["D1_vs_S1-r100"] == {"error": "missing label: D1"}
    for key in ("S1-r50_vs_S1", "S1-r100_vs_S1"):  # D1 がなくても、S1-r の比較は出る
        assert ex[key]["n_excluded_cases"] == len(QC_IN_TEST) and ex[key][key]["verdict"]


def test_cmd_stats_without_qc_failed_cases_excludes_nothing():
    write_experiment_files(avd.EXP)  # 既定: どのレコードも qc_failed = False
    avd.cmd_stats(None)
    stats = load_strict(avd.EXP / "var_desc_stats.json")
    assert stats["qc_failed"] == {"n_cases": 0}
    for setting in "AB":
        for key, ex in stats[setting]["excluding_qc_failed"].items():
            assert ex["n_excluded_cases"] == 0 and set(ex["n_excluded_cases_per_seed"].values()) == {0}


def test_cmd_stats_requires_the_qc_failed_flag_in_every_description_record():
    write_experiment_files(avd.EXP, qc_failed=QC_FAILED_IDS)
    path = avd.EXP / "var_descriptions.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["cases"]["case_0"]["qc_failed"]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(KeyError, match="qc_failed"):  # 黙って「失敗なし」にして、感度の確認を空にしない
        avd.cmd_stats(None)
    assert not (avd.EXP / "var_desc_stats.json").exists()


# ---- cmd_figure ----

@pytest.fixture
def drawn():
    """合成データで cmd_figure を実行し、(返された Figure, [設定 A の合成結果, 設定 B の合成結果]) を返す."""
    write_experiment_files(avd.EXP)
    fig = avd.cmd_figure(None)
    return fig, [synthetic_results(SETTING_SHIFT[s]) for s in "AB"]


def mean_sd(res, label):
    """label の seed ごとの R@K の（平均, 標本 SD（ddof=1））。statistics で数える（本体は numpy）."""
    v = [p["Recall@K_correct"] for p in res[label]["per_seed"]]
    return statistics.fmean(v), statistics.stdev(v)


def series_of(ax):
    """軸から (D1 の線分, D0 の線分, S1 の errorbar コンテナ) を取り出す。凡例の文言で見分ける."""
    from matplotlib.container import ErrorbarContainer
    (d1,) = [ln for ln in ax.lines if ln.get_label() == "Matching by descriptions (D1, uses no symbols)"]
    (d0,) = [ln for ln in ax.lines if ln.get_label() == "Copied source descriptions (D0, reference)"]
    (s1,) = [c for c in ax.containers
             if isinstance(c, ErrorbarContainer) and c.get_label() == "Matching by symbols (S1)"]
    return d1, d0, s1


def test_cmd_figure_writes_pdf_and_png():
    write_experiment_files(avd.EXP)
    fig = avd.cmd_figure(None)
    assert len(fig.axes) == 2  # 戻り値は Figure（設定 A・B の 2 枚）
    pdf, png = (avd.FIG_DIR / f"fig_var_description.{ext}" for ext in ("pdf", "png"))
    assert pdf.read_bytes()[:5] == b"%PDF-" and len(pdf.read_bytes()) > 5000
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and len(png.read_bytes()) > 5000


def test_main_figure_subcommand_still_returns_nothing(monkeypatch):
    """cmd_figure が Figure を返しても、main() は何も返さず、CLI の動作（図を保存する）は変わらない."""
    write_experiment_files(avd.EXP)
    monkeypatch.setattr(sys, "argv", ["analyze_var_description.py", "figure"])
    assert avd.main() is None
    assert (avd.FIG_DIR / "fig_var_description.pdf").exists() and (avd.FIG_DIR / "fig_var_description.png").exists()


def display_extent(coll):
    """PolyCollection・LineCollection が描かれる範囲（ピクセル）."""
    from matplotlib.transforms import Bbox
    return Bbox.union([p.get_extents(coll.get_transform()) for p in coll.get_paths()])


def test_cmd_figure_direct_labels_are_beyond_the_segment_ends(drawn):
    fig, _ = drawn
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        assert sorted(t.get_text() for t in ax.texts) == ["D0", "D1", "S1"]
        segments = [ln for ln in ax.lines if list(ln.get_xdata()) == [0, 100]]
        assert len(segments) == 2  # D1 と D0 は x ∈ [0, 100] の線分
        for t in ax.texts:
            assert t.xy[0] == 100 and t.xyann == (6, 0) and t.anncoords == "offset points"
            assert t.get_va() == "center"
            box = t.get_window_extent(renderer)
            for ln in ax.lines:  # 線がラベルを貫いていない
                path = ln.get_transform().transform_path(ln.get_path())
                assert not path.intersects_bbox(box, filled=False), (t.get_text(), ln.get_label())
            for coll in ax.collections:  # 帯・誤差棒（x ≤ 100）は、ラベルより左で終わる
                assert display_extent(coll).x1 < box.x0, t.get_text()
    _, labels = fig.axes[0].get_legend_handles_labels()
    assert "Matching by descriptions (D1, uses no symbols)" in labels
    assert fig.axes[0].get_ylabel() == "Recall@K (mean over 10 seeds; bars and band: ± SD)"
    colors = {ln.get_label(): ln.get_color() for ln in fig.axes[0].lines}
    assert colors["Matching by descriptions (D1, uses no symbols)"] == avd.C_DESC == "#2a78d6"
    assert avd.C_SYM == "#eb6834" and avd.C_COPY == "#1baf7a"


def test_cmd_figure_plots_the_seed_means_in_the_designed_colors(drawn):
    from matplotlib.colors import to_hex
    fig, results = drawn
    assert len(fig.axes) == len(results) == 2
    for ax, res in zip(fig.axes, results):
        d1, d0, s1 = series_of(ax)
        s1_line = s1.lines[0]  # errorbar のデータ線（記号版を 0%・50%・100% 替えた点）
        m = {lab: mean_sd(res, lab)[0] for lab in ("D1", "D0", "S1", "S1-r50", "S1-r100")}
        # 描かれた点 = seed ごとの R@K の平均
        assert list(d1.get_xdata()) == list(d0.get_xdata()) == [0, 100]
        assert list(s1_line.get_xdata()) == [0, 50, 100]
        assert list(d1.get_ydata()) == pytest.approx([m["D1"]] * 2, abs=1e-12)
        assert list(d0.get_ydata()) == pytest.approx([m["D0"]] * 2, abs=1e-12)
        assert list(s1_line.get_ydata()) == pytest.approx([m["S1"], m["S1-r50"], m["S1-r100"]], abs=1e-12)
        # 色（D1 青・S1 橙・D0 緑）は、モジュールの定数でなく描かれた線の色で確かめる
        assert [to_hex(ln.get_color()) for ln in (d1, s1_line, d0)] == ["#2a78d6", "#eb6834", "#1baf7a"]
        # 直接ラベルの y = 描かれた系列の平均（D1・D0 は線分、S1 は x = 100%）
        labels = {t.get_text(): t.xy for t in ax.texts}
        assert sorted(labels) == ["D0", "D1", "S1"]
        for name, line, expected in (("D1", d1, m["D1"]), ("D0", d0, m["D0"]), ("S1", s1_line, m["S1-r100"])):
            x, y = labels[name]
            assert x == line.get_xdata()[-1] == 100  # 線分・点列の右端（x = 100%）
            assert y == pytest.approx(expected, abs=1e-12)  # 独立に数えた平均
            assert y == pytest.approx(line.get_ydata()[-1], abs=1e-12)  # 描かれた系列の値


def test_cmd_figure_error_bars_and_band_are_the_per_seed_sd(drawn):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_hex
    fig, results = drawn
    for ax, res in zip(fig.axes, results):
        _, _, s1 = series_of(ax)
        # S1 の誤差棒: 点（0・50・100%）の平均から上下に、seed ごとの R@K の標本 SD（ddof=1）
        (bars,) = s1.lines[2]
        segments = bars.get_segments()
        assert [float(seg[0][0]) for seg in segments] == [0, 50, 100]
        for seg, label in zip(segments, ("S1", "S1-r50", "S1-r100")):
            m, sd = mean_sd(res, label)
            v = [p["Recall@K_correct"] for p in res[label]["per_seed"]]
            assert sd > 1e-3 and abs(statistics.pstdev(v) - sd) > 1e-4  # 合成データの SD は 0 でなく、ddof=0 とも違う
            (_, y_lo), (_, y_hi) = seg
            assert float(y_lo) == pytest.approx(m - sd, abs=1e-12)
            assert float(y_hi) == pytest.approx(m + sd, abs=1e-12)
        assert to_hex(bars.get_color()[0]) == "#eb6834"
        # D1 の帯: fill_between（PolyCollection）で x = 0〜100、y = 平均 ± SD。axhspan（パッチ）は使わない
        (band,) = [c for c in ax.collections if isinstance(c, PolyCollection)]
        vertices = band.get_paths()[0].vertices
        m, sd = mean_sd(res, "D1")
        assert (vertices[:, 0].min(), vertices[:, 0].max()) == (0, 100)
        assert vertices[:, 1].min() == pytest.approx(m - sd, abs=1e-12)
        assert vertices[:, 1].max() == pytest.approx(m + sd, abs=1e-12)
        assert to_hex(band.get_facecolor()[0][:3]) == "#2a78d6"
        assert len(ax.patches) == 0


# ---- cmd_precheck（E5 は呼ばず、var_conditions の関数を偽物に差し替える） ----

def install_fake_precheck(monkeypatch, sims_by_cond: dict, labels) -> None:
    import var_conditions

    class View:
        orig_syms = ["T"]
        texts = ["reactor temperature"]

    n = len(labels)
    data = {"cases": [{"case_id": "c0"}], "correct_lists": [[0]], "occ_sym": ["T"], "occ_desc": ["temp"],
            "eq_keys": ["src__eq_1"], "occ_eq": np.array([0])}
    pairs = {"case": np.zeros(n, dtype=int), "item": np.zeros(n, dtype=int),
             "occ": np.zeros(n, dtype=int), "label": labels}
    monkeypatch.setattr(var_conditions, "load_data", lambda setting: data)
    monkeypatch.setattr(var_conditions, "load_var_descriptions", lambda *a, **k: {})
    monkeypatch.setattr(var_conditions, "build_views", lambda cond, d, vd: [View()])
    monkeypatch.setattr(var_conditions, "make_matcher", lambda cond, d, views: cond)
    monkeypatch.setattr(var_conditions, "calibration_pairs", lambda idx, views, d: pairs)
    monkeypatch.setattr(var_conditions, "pair_sims", lambda pr, matcher: sims_by_cond[matcher])


def test_precheck_gate_uses_the_unrounded_auc(monkeypatch):
    # 正例 1 件・負例 25,000 件のうち 19,999 件より上 → AUC = 0.79996（4 桁に丸めると 0.8）
    n_neg, below = 25000, 19999
    labels = np.r_[1, np.zeros(n_neg, dtype=int)]
    d1 = np.r_[0.5, np.linspace(0.0, 0.49, below), np.linspace(0.51, 1.0, n_neg - below)]
    d0 = np.r_[1.0, np.linspace(0.0, 0.9, n_neg)]
    install_fake_precheck(monkeypatch, {"D1": d1, "D0": d0}, labels)
    avd.EXP.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        avd.cmd_precheck(None)
    assert exc.value.code == 2
    out = load_strict(avd.EXP / "var_desc_precheck.json")
    assert out["D1"]["auc"] == 0.8  # JSON には丸めた値
    assert out["gate"] == {"metric": "D1 AUC", "threshold": 0.8, "pass": False}  # 判定は丸める前の値
    assert out["D0"]["auc"] == 1.0 and out["D1"]["n_pos"] == 1 and out["D1"]["n_neg"] == n_neg


def test_precheck_passes_when_the_auc_is_exactly_the_gate(monkeypatch):
    from sklearn.metrics import roc_auc_score
    labels = np.r_[1, np.zeros(5, dtype=int)]
    sims = np.array([0.5, 0.1, 0.2, 0.3, 0.4, 0.9])  # 負例 5 件のうち 4 件が正例より下 → AUC = 4/5
    assert roc_auc_score(labels, sims) == AUC_GATE == 0.8  # 前提: 浮動小数点でもちょうど 0.80
    install_fake_precheck(monkeypatch, {"D1": sims, "D0": sims}, labels)
    avd.EXP.mkdir(parents=True)
    avd.cmd_precheck(None)  # ちょうど 0.80 は通る（SystemExit が出ない）
    out = load_strict(avd.EXP / "var_desc_precheck.json")
    assert out["D1"]["auc"] == 0.8
    assert out["gate"] == {"metric": "D1 AUC", "threshold": 0.8, "pass": True}


def test_precheck_passes_when_the_auc_is_above_the_gate(monkeypatch):
    labels = np.r_[1, np.zeros(1000, dtype=int)]
    sims = np.r_[0.9, np.linspace(0.0, 0.5, 1000)]
    install_fake_precheck(monkeypatch, {"D1": sims, "D0": sims}, labels)
    avd.EXP.mkdir(parents=True)
    avd.cmd_precheck(None)  # 終了しない
    out = load_strict(avd.EXP / "var_desc_precheck.json")
    assert out["gate"]["pass"] is True and out["D1"]["auc"] == 1.0
    assert out["D1"]["hardest_negatives"][0]["query_text"] == "reactor temperature"



def test_verdict_direction_from_unrounded_mean_not_rounded_delta():
    """差が小さく（約 3e-5）すべての seed で正だが、丸めると 0 になる場合、
    verdict の向きは丸める前の平均差で決まることを確認。
    
    丸めた mean_delta=0.0 は「a lower」を指すが、実際の CI は正なので
    verdict は「significant (a higher)」と判定されるべき。"""
    # 10 組のペア、差はすべて約 3e-5 で正、適度なばらつき
    vb = [0.70 + 0.01 * i for i in range(10)]
    small_diff = 3e-5
    # ばらつきは小さい（±2e-5 スケール）で、t 検定が有意（p<0.05）になる
    small_noise = np.array([0.2e-4, -0.15e-4, 0.25e-4, -0.1e-4, 0.2e-4, -0.15e-4, 0.1e-4, -0.2e-4, 0.15e-4, -0.08e-4])
    va = [b + small_diff + n for b, n in zip(vb, small_noise)]
    
    r = avd.paired_with_ci(va, vb)  # 差は a − b（正）
    
    # 保存される mean_delta は丸めた値（0.0 になる）
    assert r["mean_delta"] == 0.0
    
    # 両検定で p < 0.05（有意）
    assert r["p_ttest"] is not None and r["p_ttest"] < 0.05
    assert r["p_wilcoxon"] is not None and r["p_wilcoxon"] < 0.05
    assert r["significant"] is True
    
    # 未丸めの実際の差は正
    unrounded_mean_diff = float(np.mean(np.array(va) - np.array(vb)))
    assert unrounded_mean_diff > 0  # 実際の平均差は正
    
    # CI は丸めた値だが ±0.02 に収まる
    lo, hi = r["ci95"]
    assert -0.02 <= lo <= hi <= 0.02
    
    # 最重要: verdict は「significant (a higher)」を含む
    # （丸めた delta=0.0 でなく、実際の平均差の符号を使う）
    assert "significant (a higher)" in r["verdict"]
    json.dumps(r, allow_nan=False)  # NaN・inf が残っていなければ OK

