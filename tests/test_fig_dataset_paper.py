"""generate_figures_thesis.dataset：論文用に、拡張ケースの棒を除いた図を別の場所へ書き出せること."""
import generate_figures_thesis as gft


def test_dataset_figure_without_augmented_cases_goes_to_the_given_directory(tmp_path):
    labels = gft.dataset(augmented=False, out_dir=tmp_path)
    assert labels == ["Single-source", "Cross-source", "Synthesized"]
    assert (tmp_path / "fig_dataset.pdf").exists()
    assert (tmp_path / "fig_dataset.png").exists()


def test_split_balance_for_the_paper_shows_the_recorded_split(tmp_path):
    """論文用の図：種 42 の分割が、実験で使った分割（splits.json）と同じ件数で、軸の名前が本文の呼び名であること."""
    import json
    rec = json.load(open(gft.ROOT / "experiments" / "llm_db_baseline" / "splits.json"))["splits"]["A"]["42"]
    info = gft.split_balance(seed=42, out_dir=tmp_path, paper=True)
    assert info["n_train"] == len(rec["train"])
    assert info["n_test"] == len(rec["test"])
    assert info["titles"] == ["Ground-truth equations", "Input variables", "Output variables", "Sources spanned"]
    assert info["legend"] == ["Training", "Test"]
    assert (tmp_path / "fig_split_balance.pdf").exists()


def test_method_figures_keep_the_thesis_labels_by_default(tmp_path):
    labels = ["Baseline (classical IR)", "Proposed, static reranking", "Proposed, learned greedy"]
    assert gft.method_comparison(out_dir=tmp_path) == labels
    assert gft.characteristic(out_dir=tmp_path) == labels


def test_method_figures_for_the_paper_carry_the_name_of_the_method(tmp_path):
    """論文用の図：手法の名称（2026-11-18 に先生の案から選んだ TRACE）を凡例に出せること."""
    labels = ["Baseline (classical IR)", "TRACE, static reranking", "TRACE, learned greedy"]
    assert gft.method_comparison(method_name="TRACE", out_dir=tmp_path) == labels
    assert gft.characteristic(method_name="TRACE", out_dir=tmp_path) == labels
    assert (tmp_path / "fig_method_comparison.pdf").exists() and (tmp_path / "fig_characteristic.pdf").exists()
