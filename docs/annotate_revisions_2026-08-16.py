"""加藤先生に再確認いただくための注釈版 PDF を作る.

progress_report_2026-08-09_SK.pdf の注釈を受けて直した箇所だけを
本文中でハイライトし、右余白に一言の修正理由を付ける。

出典 : docs/progress_report_2026-08-09_topic.pdf
出力 : docs/progress_report_2026-08-09_topic_revised_annotated.pdf
"""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).parent
SRC = ROOT / "progress_report_2026-08-09_topic.pdf"
OUT = ROOT / "progress_report_2026-08-09_topic_revised_annotated.pdf"

COLOR_HL = (1.0, 0.85, 0.2)      # ハイライト（黄）
COLOR_NOTE = (1.0, 0.6, 0.0)     # 付箋（橙）

# (アンカー, 一言の修正理由)
REVISIONS = [
    ("互いに無関係な組（出典文献が重ならず",
     "比較の基準を、無作為な組から無関係な組に変えました。"),

    ("276 組すべてが0.213 を上回り",
     "入出力だけ変えた組は説明文が同じで当然なので、根拠を言い換えの組だけにしました。"),

    ("無関係な組でも類似度が高くなる場合はある",
     "無関係でも類似度が高くなる場合を実測して追加しました。"),

    ("AUC は次の手順で求める",
     "AUC の説明を手順の形に書き直しました。"),

    ("意味の近さという特徴量",
     "何の AUC かを明示しました。"),

    ("文章類似度のAUC は0.437",
     "数値が AUC の値であることを明記しました。"),

    ("手順は次のとお",
     "シャッフルと乱数の関係が分かるよう、手順を明記しました。"),
]


def place_note(page, anchor_rect, body):
    """右余白に付箋（テキスト注釈）を置く."""
    x = page.rect.width - 26
    y = max(28, min(anchor_rect.y0, page.rect.height - 40))
    annot = page.add_text_annot(fitz.Point(x, y), body, icon="Comment")
    annot.set_info(content=body)
    annot.set_colors(stroke=COLOR_NOTE)
    annot.update()


def main() -> int:
    doc = fitz.open(SRC)
    placed, missed = 0, []

    for i, (anchor, reason) in enumerate(REVISIONS, 1):
        hit = None
        for pno in range(len(doc)):
            rects = doc[pno].search_for(anchor)
            if rects:
                hit = (pno, rects)
                break
        if hit is None:
            missed.append((i, anchor))
            continue

        pno, rects = hit
        page = doc[pno]
        for r in rects:
            a = page.add_highlight_annot(r)
            a.set_colors(stroke=COLOR_HL)
            a.update()
        place_note(page, rects[0], reason)
        placed += 1
        print(f"  #{i:02d} p{pno+1} OK  {anchor[:32]}")

    for i, anchor in missed:
        print(f"  #{i:02d} --  MISS {anchor[:32]}")

    doc.save(OUT)
    print(f"\n注釈 {placed}/{len(REVISIONS)} 件を配置 → {OUT.name}")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
