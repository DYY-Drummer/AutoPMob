"""doc_split_rerun.py（同じ文献のファイルをまとめ、文献単位の分割で提案手法を学習し直す）の単体テスト."""
import pytest

from doc_split_rerun import (
    BOOK_GROUPS, doc_labels, group_sources, llm_case_means, mean_sd_over_seeds, source_stem,
)


# ---- ファイル名から文献の名前（頭の番号と .pdf を除く） ----------------------------------

@pytest.mark.parametrize("name, stem", [
    ("01_Theory_of_Star_Formation_2007.pdf", "theory_of_star_formation_2007"),
    ("FB02_Theory_of_Star_Formation_2007.pdf", "theory_of_star_formation_2007"),
    ("FB2_10_Review_liquid_phase_sintering_2008.pdf", "review_liquid_phase_sintering_2008"),
    ("FB3_01_Machine_Learning_for_Fluid_Mechanics_2019.pdf", "machine_learning_for_fluid_mechanics_2019"),
    # 年が頭に付くファイル名（初期の収集）は、年を後ろに回して自動収集の名前とそろえる
    ("2022_A_study_of_triple-mass_diffusion.pdf", "a_study_of_triple-mass_diffusion_2022"),
    # 教科書の名前は年を回さない（別の規則でまとめる）
    ("2018_Book_ProcessControl_Chapter1.pdf", "2018_book_processcontrol_chapter1"),
    ("handbook_chemeng", "handbook_chemeng"),
    ("ORNL/SPR-2018/864", "ornl/spr-2018/864"),
])
def test_source_stem_removes_collection_prefix_and_extension(name, stem):
    assert source_stem(name) == stem


# ---- 同じ文献のファイルをまとめる ------------------------------------------------------

def test_group_sources_joins_files_with_the_same_stem():
    files = ["01_Theory_of_Star_Formation_2007.pdf", "FB02_Theory_of_Star_Formation_2007.pdf", "038.pdf"]
    docs = group_sources(files, doi_groups=[], book_groups={})
    assert docs["theory_of_star_formation_2007"] == ["01_Theory_of_Star_Formation_2007.pdf",
                                                     "FB02_Theory_of_Star_Formation_2007.pdf"]
    assert docs["038.pdf"] == ["038.pdf"]          # 1 ファイルの文献はファイル名のまま


def test_group_sources_joins_files_with_the_same_doi_even_if_the_names_differ():
    files = ["A_paper_2010.pdf", "FB01_B_paper_2010.pdf", "C.pdf"]
    docs = group_sources(files, doi_groups=[["A_paper_2010.pdf", "FB01_B_paper_2010.pdf", "not_in_db.pdf"]],
                         book_groups={})
    (name, members), = [(k, v) for k, v in docs.items() if len(v) > 1]
    assert members == ["A_paper_2010.pdf", "FB01_B_paper_2010.pdf"]
    assert name in members or name == source_stem(members[0])
    assert "not_in_db.pdf" not in sum(docs.values(), [])


def test_group_sources_uses_the_book_name_for_hand_listed_books():
    files = ["SEBORG_3rd_Edition_Process_Dynamics_and.pdf", "SEBORG_Process_Dynamic_and_Control_Chap_2.pdf", "x.pdf"]
    docs = group_sources(files, doi_groups=[], book_groups=BOOK_GROUPS)
    assert docs["Seborg_Process_Dynamics_and_Control"] == files[:2]
    assert docs["x.pdf"] == ["x.pdf"]


def test_group_sources_merges_transitively_across_rules():
    # 名前で A-B、DOI で B-C がつながれば A-B-C は 1 つの文献
    files = ["01_Same_title_2000.pdf", "02_Same_title_2000.pdf", "Other_name.pdf"]
    docs = group_sources(files, doi_groups=[["02_Same_title_2000.pdf", "Other_name.pdf"]], book_groups={})
    assert len(docs) == 1
    assert sorted(sum(docs.values(), [])) == sorted(files)


def test_book_groups_cover_the_two_duplicated_textbooks_and_marlin():
    assert set(BOOK_GROUPS["Seborg_Process_Dynamics_and_Control"]) == {
        "SEBORG_3rd_Edition_Process_Dynamics_and.pdf", "SEBORG_Process_Dynamic_and_Control_Chap_2.pdf"}
    assert set(BOOK_GROUPS["Corriou_Process_Control"]) == {
        "2018_Book_ProcessControl.pdf", "2018_Book_ProcessControl_Chapter1.pdf",
        "2018_Book_ProcessControl_Chapter1_6.pdf", "9781852337766-c1.pdf"}
    assert set(BOOK_GROUPS["Marlin_Process_Control"]) == {
        "Process_Control_Designing_Processes_and_Control_Systems_for_Dynamic_Performance.pdf",
        "Process_Control_Chapter3.pdf", "Process_Control_Chapter4.pdf"}


# ---- ケースの出典（ファイル）を文献に置き換える ---------------------------------------------

def test_doc_labels_maps_each_case_source_to_its_document_and_keeps_unknown_files():
    file_to_doc = {"a.pdf": "doc1", "b.pdf": "doc1", "c.pdf": "c.pdf"}
    assert doc_labels(["a.pdf", "b.pdf", "c.pdf", "unknown"], file_to_doc) == ["doc1", "doc1", "c.pdf", "unknown"]


# ---- 集計 ----------------------------------------------------------------------------

def test_llm_case_means_averages_the_runs_of_each_case():
    per_case = [{"case_id": "x", "run": 1, "recall@20": 1.0, "AP@20": 1.0},
                {"case_id": "x", "run": 2, "recall@20": 0.5, "AP@20": 0.25},
                {"case_id": "y", "run": 1, "recall@20": 0.0, "AP@20": 0.0}]
    means = llm_case_means(per_case)
    assert means["x"] == {"n_runs": 2, "recall@20": 0.75, "AP@20": 0.625}
    assert means["y"] == {"n_runs": 1, "recall@20": 0.0, "AP@20": 0.0}


def test_mean_sd_over_seeds_uses_the_sample_standard_deviation():
    out = mean_sd_over_seeds({"42": 0.9, "123": 1.0, "456": 0.8})
    assert out["n"] == 3
    assert out["mean"] == pytest.approx(0.9)
    assert out["sd"] == pytest.approx(0.1)
    assert mean_sd_over_seeds({"42": 0.9})["sd"] == 0.0


def test_mean_sd_over_seeds_with_no_seeds_reports_zero_and_no_values():
    # 設定 B には単一出典・出典横断のケースがないので、種類ごとの集計が空になる
    out = mean_sd_over_seeds({})
    assert out["n"] == 0
    assert out["mean"] is None and out["sd"] is None and out["min"] is None and out["max"] is None
    assert out["per_seed"] == {}
