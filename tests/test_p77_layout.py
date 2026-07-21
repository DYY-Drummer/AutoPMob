"""レイアウト監査：全スライドで「はみ出し・重なりゼロ」を実フォント計測で検証。

textmetrics.audit_presentation が生成物 pptx を開き直し、
(a) キャンバス下端はみ出し (b) 図形内テキストあふれ (c) テキスト矩形の重なり
を検査する。ユーザー指摘（2026-07-21、S1 の本文と注記の重なり等）の再発防止。
"""
import sys
from pathlib import Path

import pytest
from pptx import Presentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "seminar_p77"))

from build_p77_slides import build
from textmetrics import audit_presentation


@pytest.fixture(scope="module")
def prs(tmp_path_factory):
    out = tmp_path_factory.mktemp("deck_layout") / "p77_layout.pptx"
    build(str(out))
    return Presentation(str(out))


def test_no_overflow_or_overlap(prs):
    problems = audit_presentation(prs)
    assert not problems, "レイアウト違反:\n" + "\n".join(problems)
