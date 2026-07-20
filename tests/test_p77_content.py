"""P-77紹介スライドの本文データが仕様書のルールを満たすことの検証。

仕様: docs/superpowers/specs/2026-07-21-p77-seminar-slides-design.md
- 略称はスライドごとに正式名称併記 / 禁止語なし / 台本250-500字・合計2300-4000字
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "seminar_p77"))

import content_p77 as C


def test_eight_slides():
    assert len(C.SLIDES) == 8


def test_every_slide_has_kind_and_notes():
    for i, s in enumerate(C.SLIDES):
        assert "kind" in s, f"slide {i+1}: kind がない"
        assert isinstance(s["notes"], str) and s["notes"], f"slide {i+1}: notes がない"


def test_notes_length_per_slide():
    for i, s in enumerate(C.SLIDES):
        n = len(s["notes"])
        assert 250 <= n <= 500, f"slide {i+1}: 台本 {n} 字（250-500字の範囲外）"


def test_notes_total_length():
    total = sum(len(s["notes"]) for s in C.SLIDES)
    assert 2300 <= total <= 4000, f"台本合計 {total} 字（約12分=2300-4000字の範囲外）"


def test_abbreviations_have_full_names_on_each_slide():
    for i in range(8):
        text = C.slide_all_text(i)
        for abbr, full in C.ABBREV_FULL.items():
            if abbr in text:
                assert full in text, f"slide {i+1}: 略称 {abbr} があるのに正式名称 {full} がない"


def test_no_forbidden_words():
    for i in range(8):
        text = C.slide_all_text(i) + C.SLIDES[i]["notes"]
        for w in C.FORBIDDEN_WORDS:
            assert w not in text, f"slide {i+1}: 禁止語 {w} が含まれる"


def test_required_explanation_pairs():
    for idx, term, expl in C.REQUIRED_PAIRS:
        text = C.slide_all_text(idx)
        assert term in text, f"slide {idx+1}: 用語 {term} が見つからない（REQUIRED_PAIRS と本文の不整合）"
        assert expl in text, f"slide {idx+1}: {term} の説明 {expl} がない"


def test_title_slide_mentions_p77_and_authors():
    text = C.slide_all_text(0)
    assert "P-77" in text
    assert "Chiang" in text and "国立台湾科技大学" in text


def test_summary_slide_declares_abstract_only_basis():
    assert "抄録のみ" in C.slide_all_text(7)
