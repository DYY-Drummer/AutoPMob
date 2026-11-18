"""thesis/master_thesis/check_thesis.py（修士論文の点検）の単体テスト."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "thesis" / "master_thesis"))
import check_thesis as ct  # noqa: E402


def test_body_pages_reads_the_page_of_the_end_of_the_conclusion():
    aux = "\n".join([r"\newlabel{sec:conclusion}{{9}{31}{}{chapter.9}{}}",
                     r"\newlabel{thesis:endbody}{{9}{31}{}{chapter.9}{}}",
                     r"\newlabel{app:hyper}{{Appendix A}{36}{}{appendix.A}{}}"])
    assert ct.body_pages(aux) == 31


def test_body_pages_reports_a_thesis_that_was_not_built():
    with pytest.raises(ValueError, match="thesis:endbody"):
        ct.body_pages(r"\newlabel{sec:intro}{{1}{1}}")


def test_unresolved_lists_undefined_references_and_citations():
    log = "\n".join(["LaTeX Warning: Reference `tab:gone' on page 3 undefined on input line 12.",
                     "Package natbib Warning: Citation `Nobody2020' on page 5 undefined on input line 40.",
                     "LaTeX Warning: Reference `tab:gone' on page 9 undefined on input line 80.",
                     "Overfull \\hbox (3.0pt too wide) in paragraph at lines 4--5"])
    assert ct.unresolved(log) == ["Nobody2020", "tab:gone"]
    assert ct.unresolved("Output written on main.dvi (102 pages).") == []


def test_paper_sections_lists_the_section_files_in_order_and_skips_comments():
    tex = "\n".join([r"\input{../../paper/cace/sections/abstract}",
                     r"% \input{../../paper/cace/sections/highlights}",
                     r"\input{../../paper/cace/sections/introduction}",
                     r"\input{sections/related}",
                     r"\input{../../paper/cace/appendix}"])
    assert ct.paper_sections(tex) == ["abstract", "introduction", "related"]


def test_check_reports_a_body_over_the_limit(tmp_path):
    thesis, paper = tmp_path / "thesis", tmp_path / "paper"
    thesis.mkdir()
    paper.mkdir()
    body = "".join(f"\\input{{../../paper/cace/sections/{s}}}\n" for s in ("abstract", "introduction", "conclusion"))
    (thesis / "main.tex").write_text(body + "\\input{../../paper/cace/appendix}\n", encoding="utf-8")
    (paper / "main.tex").write_text("".join(f"\\input{{sections/{s}}}\n" for s in
                                            ("abstract", "highlights", "keywords", "introduction", "conclusion", "backmatter"))
                                    + "\\input{appendix}\n", encoding="utf-8")
    (thesis / "main.aux").write_text(r"\newlabel{thesis:endbody}{{9}{41}{}{chapter.9}{}}", encoding="utf-8")
    (thesis / "main.log").write_text("", encoding="utf-8")
    failures = ct.check(thesis, paper)
    assert failures == ["the main text has 41 pages, over the limit of 40"]
    (thesis / "main.aux").write_text(r"\newlabel{thesis:endbody}{{9}{40}{}{chapter.9}{}}", encoding="utf-8")
    assert ct.check(thesis, paper) == []


def test_check_reports_a_section_of_the_paper_that_the_thesis_does_not_read(tmp_path):
    thesis, paper = tmp_path / "thesis", tmp_path / "paper"
    thesis.mkdir()
    paper.mkdir()
    (thesis / "main.tex").write_text("\\input{../../paper/cace/sections/abstract}\n\\input{../../paper/cace/sections/introduction}\n"
                                     "\\input{../../paper/cace/appendix}\n", encoding="utf-8")
    (paper / "main.tex").write_text("\\input{sections/abstract}\n\\input{sections/introduction}\n\\input{sections/results}\n",
                                    encoding="utf-8")
    (thesis / "main.aux").write_text(r"\newlabel{thesis:endbody}{{9}{30}{}{chapter.9}{}}", encoding="utf-8")
    (thesis / "main.log").write_text("", encoding="utf-8")
    assert ct.check(thesis, paper) == ["the thesis reads the sections ['abstract', 'introduction'], the paper has "
                                       "['abstract', 'introduction', 'results']"]


def test_the_thesis_reads_every_section_of_the_paper():
    """実物：修論の main.tex が、論文の本文の節（Abstract から Conclusion まで）を同じ順に読み込み、付録も読み込むこと."""
    thesis = (ct.HERE / "main.tex").read_text(encoding="utf-8")
    paper = (ct.PAPER / "main.tex").read_text(encoding="utf-8")
    assert ct.paper_sections(thesis) == [s for s in ct.paper_sections(paper) if s not in ct.PAPER_ONLY]
    assert "\\input{../../paper/cace/appendix}" in thesis
    assert "\\input{../../paper/cace/numbers}" in thesis
