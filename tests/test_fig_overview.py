"""図 1（AutoPMoB の全体像と本研究の位置づけ）の生成テスト.

仕様: docs/superpowers/specs/2026-09-07-w2-introduction-design.md §3
- PDF/PNG が生成される
- 自己完結に必要なラベル（本研究の段階・4 段階名・DB 規模・CSTR の 3 式・自由度）が図中にある
"""
import matplotlib

matplotlib.use("Agg")

import generate_figures_thesis as G


def _all_text(fig) -> str:
    parts = []
    for ax in fig.axes:
        parts.extend(t.get_text() for t in ax.texts)
    return " ".join(parts).replace("\n", " ")  # 折り返しの改行は空白に正規化


def test_overview_generates_files_and_labels():
    fig = G.overview()
    texts = _all_text(fig)
    for needle in [
        "this study", "Equation-set retrieval", "Documents", "Extraction", "Equivalence",
        "Physical model", "11,146 equations", "361 sources",
        r"r_A = k", r"\exp(-E/RT)", "degrees of freedom",
    ]:
        assert needle in texts, needle
    for ext in ("pdf", "png"):
        assert (G.FIG / f"fig_autopmob_overview.{ext}").exists()


def test_overview_edges_match_shared_variables():
    """辺ラベルは共有変数（ODE-rate law: r_A, rate law-Arrhenius: k）に厳密対応する."""
    fig = G.overview()
    labels = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert G.OVERVIEW_EDGE_LABELS == [r"$r_A$", r"$k$"]
    for lab in G.OVERVIEW_EDGE_LABELS:
        assert lab in labels
