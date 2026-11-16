"""generate_figures_thesis.dataset：論文用に、拡張ケースの棒を除いた図を別の場所へ書き出せること."""
import generate_figures_thesis as gft


def test_dataset_figure_without_augmented_cases_goes_to_the_given_directory(tmp_path):
    labels = gft.dataset(augmented=False, out_dir=tmp_path)
    assert labels == ["Single-source", "Cross-source", "Synthesized"]
    assert (tmp_path / "fig_dataset.pdf").exists()
    assert (tmp_path / "fig_dataset.png").exists()
