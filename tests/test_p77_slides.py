"""生成された pptx が仕様を満たすことの検証（生成物を開き直して確認）。"""
import sys
from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "seminar_p77"))

import content_p77 as C
from build_p77_slides import build


@pytest.fixture(scope="module")
def prs(tmp_path_factory):
    out = tmp_path_factory.mktemp("deck") / "p77_test.pptx"
    build(str(out))
    return Presentation(str(out))


def iter_runs(slide):
    """スライド上の全 run（テキストボックス＋表セル）を列挙。"""
    for sh in slide.shapes:
        if sh.has_text_frame:
            for para in sh.text_frame.paragraphs:
                yield from para.runs
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for cell in row.cells:
                    for para in cell.text_frame.paragraphs:
                        yield from para.runs


def slide_text(slide):
    texts = []
    for sh in slide.shapes:
        if sh.has_text_frame:
            texts.append(sh.text_frame.text)
        if getattr(sh, "has_table", False) and sh.has_table:
            for row in sh.table.rows:
                for cell in row.cells:
                    texts.append(cell.text_frame.text)
    return "\n".join(texts)


def test_slide_count(prs):
    assert len(prs.slides._sldIdLst) == 8


def test_canvas_is_16_9(prs):
    assert prs.slide_width == 12192000 and prs.slide_height == 6858000


def test_all_runs_at_least_20pt_and_explicit(prs):
    for i, slide in enumerate(prs.slides):
        for run in iter_runs(slide):
            if not run.text.strip():
                continue
            assert run.font.size is not None, \
                f"slide {i+1}: サイズ未指定の run 「{run.text[:20]}」"
            assert run.font.size >= Pt(20), \
                f"slide {i+1}: {run.font.size.pt}pt < 20pt 「{run.text[:20]}」"


def test_fonts_are_ibm_plex(prs):
    for i, slide in enumerate(prs.slides):
        for run in iter_runs(slide):
            if not run.text.strip():
                continue
            assert run.font.name and run.font.name.startswith("IBM Plex"), \
                f"slide {i+1}: フォント {run.font.name} 「{run.text[:20]}」"


def test_slide_texts_match_content_module(prs):
    for i, slide in enumerate(prs.slides):
        built = slide_text(slide)
        assert C.SLIDES[i]["headline"] in built, f"slide {i+1}: 見出しが描画されていない"


def test_key_strings_rendered(prs):
    slides = list(prs.slides)
    assert "Model Context Protocol" in slide_text(slides[2])
    assert "60" in slide_text(slides[4])
    assert "backup ファイル" in slide_text(slides[5])
    assert "抄録のみ" in slide_text(slides[7])


def test_notes_attached_verbatim(prs):
    for i, slide in enumerate(prs.slides):
        notes = slide.notes_slide.notes_text_frame.text
        assert notes == C.SLIDES[i]["notes"], f"slide {i+1}: 台本がノート欄と不一致"
