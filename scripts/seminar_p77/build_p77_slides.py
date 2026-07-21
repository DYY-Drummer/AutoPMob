# -*- coding: utf-8 -*-
"""MCP 活用研究 紹介スライド（v2・10 枚）のビルダー。

テンプレ（PSE Asia 2026 発表スライド）を複製し、既存スライドを全削除して
content_p77.SLIDES の 10 枚を描画する。
使い方: python3 scripts/seminar_p77/build_p77_slides.py [出力パス]
"""
import shutil
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_p77 as C

TEMPLATE = "/Users/kazuhiromiyamura/Desktop/学会/PSE ASIAN2026/PSEAsia2026_slide_v2.pptx"
DEFAULT_OUT = "/Users/kazuhiromiyamura/Desktop/ゼミ/論文紹介_LLM_MCP_シミュレータ操作_かずひろ.pptx"

NAVY = RGBColor(0x16, 0x30, 0x4F)
TEXT = RGBColor(0x22, 0x22, 0x22)
BLUE = RGBColor(0x2E, 0x86, 0xC1)
GREEN = RGBColor(0x2E, 0x88, 0x51)
GRAY = RGBColor(0x8C, 0x8C, 0x8C)
LIGHT = RGBColor(0xEA, 0xF3, 0xFB)
LIGHT2 = RGBColor(0xDC, 0xEA, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

BODY_FONT = "IBM Plex Sans"
HEAD_FONT = "IBM Plex Sans SemiBold"


def _style(run, size, color, bold, font):
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = font  # 欧文フォント（<a:latin>）
    # IBM Plex に和文グリフは無いため、和文フォント（<a:ea>）を明示指定する
    rPr = run.font._rPr
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", "Hiragino Kaku Gothic ProN")


def add_text(slide, x, y, w, h, lines, size=20, color=TEXT, bold=False,
             font=BODY_FONT, align=PP_ALIGN.LEFT, spacing=1.15):
    """テキストボックスを置く。lines は str または list[str]。"""
    if isinstance(lines, str):
        lines = [lines]
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        run = para.add_run()
        run.text = line
        _style(run, size, color, bold, font)
    return box


def add_bullets(slide, x, y, w, h, items, size=20, spacing=1.2):
    """箇条書き。item が (lead, rest) の tuple なら lead を太字 NAVY にする。"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.line_spacing = spacing
        para.space_after = Pt(8)
        bullet = para.add_run()
        bullet.text = "・"
        _style(bullet, size, TEXT, False, BODY_FONT)
        if isinstance(item, tuple):
            lead, rest = item
            r1 = para.add_run()
            r1.text = lead
            _style(r1, size, NAVY, True, HEAD_FONT)
            r2 = para.add_run()
            r2.text = rest
            _style(r2, size, TEXT, False, BODY_FONT)
        else:
            r = para.add_run()
            r.text = item
            _style(r, size, TEXT, False, BODY_FONT)
    return box


def add_box(slide, x, y, w, h, lines, fill, text_color=WHITE, size=20,
            bold=False, font=BODY_FONT, outline=None):
    """角丸四角＋中央揃えテキスト。"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if outline is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = outline
        shape.line.width = Pt(1.5)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    if isinstance(lines, str):
        lines = lines.split("\n")
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = line
        _style(run, size, text_color, bold, font)
    return shape


def add_arrow(slide, x, y, w, h, label=None, fill=GRAY, left=False):
    """矢印（既定は右向き）。label があれば矢印の上にテキストを置く。"""
    kind = MSO_SHAPE.LEFT_ARROW if left else MSO_SHAPE.RIGHT_ARROW
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    if label:
        add_text(slide, x - 0.3, y - 0.55, w + 0.6, 0.5, label, size=20,
                 color=NAVY, align=PP_ALIGN.CENTER)
    return shape


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_headline(slide, text, sub=None):
    add_text(slide, 0.55, 0.3, 12.25, 1.05, text, size=28, color=NAVY,
             bold=True, font=HEAD_FONT)
    if sub:
        add_text(slide, 0.55, 1.28, 12.25, 0.6, sub, size=22, color=BLUE,
                 font=HEAD_FONT, bold=True)


# ---- kind 別レンダラ --------------------------------------------------

def render_title(slide, c):
    add_text(slide, 0.8, 0.9, 11.7, 1.5, c["headline"], size=36, color=NAVY,
             bold=True, font=HEAD_FONT)
    add_text(slide, 0.8, 2.25, 11.7, 0.85, c["sub"], size=24, color=BLUE,
             bold=True, font=HEAD_FONT)
    add_text(slide, 0.8, 3.2, 11.7, 2.3, c["body"], size=22, color=TEXT, spacing=1.3)
    add_text(slide, 0.8, 5.6, 11.7, 0.6, c["explain"], size=20, color=GRAY)
    add_text(slide, 0.8, 6.5, 11.7, 0.55, c["footer"], size=22, color=TEXT)


def render_table(slide, c):
    add_headline(slide, c["headline"])
    add_text(slide, 0.55, 1.35, 12.25, 0.55, c["lead"], size=22, color=TEXT, bold=True)
    n_rows = len(c["rows"]) + 1
    widths = c.get("col_widths", [4.9, 1.1, 6.25])
    table_shape = slide.shapes.add_table(
        n_rows, len(widths), Inches(0.55), Inches(2.0), Inches(12.25),
        Inches(c.get("table_h", 4.15)))
    table = table_shape.table
    for j, w in enumerate(widths):
        table.columns[j].width = Inches(w)
    for j, head in enumerate(c["header"]):
        _fill_cell(table.cell(0, j), head, bold=True, color=WHITE, fill=NAVY)
    for i, row in enumerate(c["rows"], start=1):
        fill = LIGHT if i % 2 == 1 else WHITE  # 既定の表スタイルを上書きして縞にする
        for j, val in enumerate(row):
            _fill_cell(table.cell(i, j), val, fill=fill,
                       align=PP_ALIGN.CENTER if j == 1 else PP_ALIGN.LEFT)
    # 表の高さが小さいスライドでは注記を表の直下に寄せる（間延び防止）
    note_y = min(6.30, 2.0 + c.get("table_h", 4.15) + 0.2)
    add_text(slide, 0.55, note_y, 12.25, 0.5, c["note"], size=20, color=BLUE, bold=True)
    add_text(slide, 0.55, note_y + 0.55, 12.25, 0.45, c["source"], size=20, color=GRAY)


def _fill_cell(cell, text, bold=False, color=TEXT, fill=None, align=PP_ALIGN.LEFT):
    cell.margin_top = cell.margin_bottom = Pt(2)
    if fill is not None:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    tf = cell.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    _style(run, 20, color, bold, BODY_FONT)


def render_mcp_fig(slide, c):
    add_headline(slide, c["headline"])
    add_bullets(slide, 0.55, 1.5, 12.25, 2.4, c["bullets"], size=22)
    fig = c["fig"]
    add_box(slide, 0.7, 4.55, 3.4, 1.6, fig["llm"], NAVY, WHITE, size=22,
            bold=True, font=HEAD_FONT)
    add_arrow(slide, 4.25, 5.05, 0.75, 0.55)
    add_box(slide, 5.15, 4.7, 2.7, 1.3, fig["mcp"], BLUE, WHITE, size=20, bold=True)
    add_arrow(slide, 8.0, 5.05, 0.75, 0.55)
    add_box(slide, 8.9, 4.3, 3.7, 2.1,
            [fig["tools_title"]] + fig["tools"].split("\n"),
            LIGHT, TEXT, size=20)


def render_arch_fig(slide, c):
    add_headline(slide, c["headline"])
    # 幅は 20pt 太字（Hiragino W6/IBM Plex SemiBold）の実測折り返し幅から逆算。
    # 必要 usable 幅: b2"Claude Desktop"2.38 / b3 2.15 / b4"Python API"1.65 / b5"AVEVA Process"2.33
    xs = [(0.55, 1.95), (2.68, 2.62), (5.48, 2.44), (8.10, 1.95), (10.23, 2.55)]
    fills = [LIGHT2, NAVY, BLUE, LIGHT2, GREEN]
    colors = [TEXT, WHITE, WHITE, TEXT, WHITE]
    for (x, w), fill, col, label in zip(xs, fills, colors, c["boxes"]):
        add_box(slide, x, 1.85, w, 1.35, label, fill, col, size=20, bold=True)
    for gap_x in (2.50, 5.30, 7.92, 10.05):
        add_arrow(slide, gap_x, 2.32, 0.18, 0.42)
    add_arrow(slide, 3.0, 3.5, 8.5, 0.5, left=True)
    add_text(slide, 3.0, 4.02, 8.5, 0.45, c["arrow_back"], size=20, color=GRAY,
             align=PP_ALIGN.CENTER)
    add_bullets(slide, 0.55, 4.55, 12.25, 2.5, c["bullets"], size=20)


def render_two_col(slide, c):
    add_headline(slide, c["headline"])
    add_box(slide, 0.55, 1.6, 5.95, 4.55, [""], LIGHT2)
    add_box(slide, 6.85, 1.6, 5.95, 4.55, [""], LIGHT)
    add_text(slide, 0.85, 1.8, 5.4, 0.55, c["left_title"], size=24, color=GRAY,
             bold=True, font=HEAD_FONT)
    add_bullets(slide, 0.85, 2.5, 5.45, 3.4, c["left_items"], size=20)
    add_text(slide, 7.15, 1.8, 5.4, 0.55, c["right_title"], size=24, color=GREEN,
             bold=True, font=HEAD_FONT)
    add_bullets(slide, 7.15, 2.5, 5.45, 3.4, c["right_items"], size=20)
    bar = add_box(slide, 0.55, 6.4, 12.25, 0.65, c["bottom"], NAVY, WHITE,
                  size=22, bold=True, font=HEAD_FONT)
    return bar


def render_bullets_box(slide, c):
    add_headline(slide, c["headline"])
    add_bullets(slide, 0.55, 1.7, 12.25, 3.4, c["bullets"], size=22, spacing=1.3)
    add_box(slide, 0.55, 5.5, 12.25, 1.1, c["box"], LIGHT, NAVY, size=24,
            bold=True, font=HEAD_FONT, outline=BLUE)


def render_summary(slide, c):
    add_headline(slide, c["headline"])
    items = [f"{i+1}.  {t}" for i, t in enumerate(c["items"])]
    add_text(slide, 0.8, 1.8, 11.75, 4.0, items, size=22, spacing=1.5)
    add_text(slide, 0.8, 6.3, 11.75, 0.6, c["note"], size=20, color=GRAY)


RENDERERS = {
    "title": render_title,
    "table": render_table,
    "mcp_fig": render_mcp_fig,
    "arch_fig": render_arch_fig,
    "two_col": render_two_col,
    "bullets_box": render_bullets_box,
    "summary": render_summary,
}


def build(out_path: str) -> None:
    import copy

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, out)          # マスター・テーマを丸ごと引き継ぐ
    prs = Presentation(str(out))

    # Get reference to first slide's notes structure before deletion
    first_slide = next(iter(prs.slides))
    reference_notes_element = first_slide.notes_slide._element

    sld_list = prs.slides._sldIdLst      # 既存14枚を全削除
    for sld in list(sld_list):
        # rel も切断しないと旧スライドパーツがパッケージ内に残り、
        # 保存時に slide1..8 のパート名衝突（Duplicate name 警告）を起こす。
        prs.part.drop_rel(sld.get(qn("r:id")))
        sld_list.remove(sld)

    for c in C.SLIDES:
        slide = prs.slides.add_slide(prs.slide_layouts[0])  # 'DEFAULT'

        # Clone notes structure so notes_placeholder exists
        notes_slide = slide.notes_slide
        reference_cSld = reference_notes_element.find(
            "{http://schemas.openxmlformats.org/presentationml/2006/main}cSld")
        if reference_cSld is not None:
            new_cSld_elem = copy.deepcopy(reference_cSld)
            existing_cSld = notes_slide._element.find(
                "{http://schemas.openxmlformats.org/presentationml/2006/main}cSld")
            if existing_cSld is not None:
                notes_slide._element.remove(existing_cSld)
            notes_slide._element.insert(0, new_cSld_elem)

        RENDERERS[c["kind"]](slide, c)
        set_notes(slide, c["notes"])
    # テンプレ由来の文書メタデータを本デッキ用に上書き（作成者はそのまま）
    prs.core_properties.title = "論文紹介: LLMにシミュレータを操作させる研究の構造（DTU論文中心）"
    prs.core_properties.subject = "研究室ゼミ（2026-07）"
    prs.save(str(out))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    build(target)
    print(f"wrote {target}")
