import subprocess
from pathlib import Path

import kato1021_counts as C
import kato1021_mapping as M

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "experiments" / "kato1020" / "excluded_cases.json"
V3 = ROOT / "experiments" / "kato1020" / "excluded_cases_v3.json"


def test_counts_reproduce_report_db21007_for_the_91_case_list():
    d = C.counts(V1)
    assert d["n_cases"] == 1732
    assert d["req3_trivial_no_intermediate"] == 1725
    assert d["same_symbol_different_definition_cases"] == 1300
    assert d["llm_meaning_conflict_cases"] == 596
    assert (d["setting_A_mean_K"], d["setting_A_share_K_ge_2"]) == (3.99, 0.904)
    assert d["table2"]["出典横断"] == {"n": 640, "mean_K": 2.01, "max_K": 3, "mean_sources": 2.01}
    assert (d["gt_distinct_equations"], d["gt_identical_latex_in_other_source"]) == (3132, 386)
    assert d["cross_source_no_shared_symbol_by_variant"] == {"multisource_original": 24, "multisource_random_io": 81}
    # 報告書 §3.2「92 件のうち 91 件は 1 つの出典」「92 件中 67 件は 1 式、61 件は基礎式集の式だけ」
    assert d["single_source_breakdown"] == {"n": 92, "K1": 67, "handbook_only": 61, "one_source": 91}


def test_counts_v3_drop_only_core_019_v1_from_the_single_source_cases():
    d = C.counts(V3)
    assert d["n_cases"] == 1731
    assert d["single_source_breakdown"] == {"n": 91, "K1": 67, "handbook_only": 61, "one_source": 91}
    assert d["table2"]["単一出典"] == {"n": 91, "mean_K": 1.33, "max_K": 4, "mean_sources": 1.0}
    assert (d["req3_trivial_no_intermediate"], d["same_symbol_different_definition_cases"]) == (1724, 1299)


def test_context_changes_since_db21007_are_only_the_113_rewritten_contexts():
    cc = C.context_changes(V1)
    assert cc["n_changed"] == 113 and cc["fields"] == {"context": 113}
    assert cc["by_variant"] == {"multisource_original": 2, "multisource_random_io": 111}
    assert cc["n_changed_excluded"] == 0 and cc["n_changed_setting_B"] == 0


def test_rng_and_wcount_formats():
    assert M.rng([2000, 1000, 2000]) == "1,000〜2,000"
    assert M.rng([2000, 2000]) == "2,000"
    assert M.wcount(["w30-70", "w40-60", "w30-70"]) == "(0.3, 0.7) 2、(0.4, 0.6) 1"


def test_check_flags_numbers_absent_from_report():
    D = M.Doc(None, None, r"Recall@20 を $0.906$ から 0.981 に（$+0.075$）、$-0.015$、1{,}732 件")
    D.check("x", "0.906 → 0.981（+0.075）、−0.015、1,732 件")
    assert D.miss == []
    D.check("y", "0.907、1,733 件")
    assert [t for _, _, t in D.miss] == ["0.907", "1,733"]


def test_check_requires_the_sign_of_a_difference():
    D = M.Doc(None, None, r"Recall@20 は $+0.001$（3 通り）、$-0.0005$")
    D.check("x", "−0.0005（5/10）、+0.001（3/10）")
    assert D.miss == []
    D.check("y", "−0.001（5/10）")
    assert [t for _, _, t in D.miss] == ["−0.001"]


def test_small_differences_are_written_to_four_decimals():
    assert M.sdx(-0.000513) == "−0.0005"
    assert M.sdx(0.0123) == "+0.012"


def test_old_values_all_appear_in_report_db21007_and_claims_match_themselves():
    text = subprocess.run(["git", "show", f"{M.REPORT_REV}:{M.REPORT}"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout
    o = M.Run(M.K20, "v1_check", V1)
    n = M.Run(M.K20, "v1_check", V1)   # 旧値どうしで作る：照合の漏れがなく、主張の比較がすべて「同じ」になる
    md, miss, claims = M.build(o, n, text, M.REPORT_REV)
    assert miss == []
    assert claims and all(a == b for _, _, a, b in claims)
    assert "（未計算）" not in md
