"""加藤先生に再確認いただくための注釈版 PDF を作る.

progress_report_2026-08-09_SK.pdf（加藤先生の注釈 7 件）以降に直した箇所を
本文中でハイライトし、右余白に「指摘 → 対応」の付箋を付ける。

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

# (アンカー, 出所, 指摘・要望, 対応)
#   出所 "加藤" = SK.pdf の注釈、"一洋" = 本人からの指摘
REVISIONS = [
    ("それぞれ0.7 と0.3 の重みを掛けて", "一洋",
     "第 1 段の文章類似度の重みは 0.7 のはずでは。",
     "ご指摘のとおりでした。実装は 0.7*文章類似度 + 0.3*変数の一致度 です。"
     "旧版の「7 対 3 の重み」「重み 7」は値と比が紛れるため、実際の値 0.7 と 0.3 に改めました。"),

    ("10 個の特徴量の意味と計算方法を表", "一洋",
     "言及した各特徴量の意味と計算方法をまとめる表を入れてほしい。",
     "表 1 を新設し、10 個の特徴量を「基本の 7 個」と「集合の 3 個」に分けて、"
     "意味と計算式を示しました。集合特徴量の参照集合が第 1 段の上位 5 件であることも明記しています。"),

    ("互いに無関係な組（出典文献が重ならず", "加藤",
     "比較には、無作為な 2 ケースではなく、無関係な 2 ケースを使うべきでは？",
     "比較の基準を「無関係な組」に置き換えました。"
     "出典文献が重ならず、同じ物理モデルにも由来しない組（191,722 組）に限定した定義です。"
     "中央値は 0.213 で、無作為な組の 0.218 とほぼ同じでしたので、結論は変わっていません。"),

    ("276 組すべてが0.213 を上回り", "加藤",
     "計算に使用した文は変えてないから当たり前では？（入出力だけ変えた組が 1.000 に集まる件）",
     "そのとおりでしたので、1.000 の組を根拠にする記述を削除しました。"
     "内容が同じことの根拠は、説明文を書き換えても高い値を保つ「言い換えの組」だけに絞っています。"),

    ("無関係な組でも類似度が高くなる場合はある", "加藤",
     "異なるケースでも類似度が高くなる場合がない？あるなら、それはどのようなときか？",
     "実測して項目を追加しました。0.5 を超えるのは 3.9% で、"
     "同じ種類の装置・現象を扱う組（CSTR どうしで 0.70）がこれに当たります。"
     "話題が実際に近い場合なので、ベクトルの欠陥ではないと考えています。"),

    ("AUC は次の手順で求める", "加藤",
     "意味がわからない。（AUC の説明）",
     "手順の形に書き直しました。正解式 1 本と不正解式 1 本の組をすべて作り、"
     "正解式のほうが値が大きい組の割合を数える、という手続きを明示しています。"),

    ("意味の近さという特徴量", "加藤",
     "これは何？（「意味の近さ 1 つの AUC」「同じ意味の近さの AUC」）",
     "「意味の近さという特徴量 1 つだけで正解式を並べたときの AUC」と展開しました。"
     "候補側も「同じ計算を、不正解式の範囲を候補 50 件に変えて行なうと」と書き直しています。"),

    ("文章類似度のAUC は0.437", "加藤",
     "これ以降の文の意味がわからない。類似度 0.437 って何のこと？",
     "0.437 は文章類似度そのものではなく AUC の値でしたので、"
     "「◯◯の AUC は…」と全箇所で明記しました。表 2 のキャプションにも「列は不正解式の範囲の違い」を追記しています。"),

    ("手順は次のとお", "加藤",
     "乱数とシャッフルの関係がわからない。同手順？",
     "手順を分けて書きました。(1) 対象の特徴量だけを候補式の間で入れ替える（他の 9 個は元のまま）、"
     "(2) 入れ替え方を変えて 20 回繰り返し平均、(3) さらにデータの分割と訓練をやり直した 10 個のモデルで"
     "全体を繰り返して平均と標準偏差を取る、という関係です。"),

    ("が肩代わりする余地がない", "一洋",
     "「訓練し直さないので、モデルが実際にその特徴量へどれだけ頼っているかを直接測れる」の意味が分からない。",
     "前回のアブレーションとの違いを明示しました。訓練し直すと他の特徴量が肩代わりできるため、"
     "そちらは「その特徴量が無くても課題が解けるか」を測ります。PFI はモデルを固定するので"
     "「今あるモデルがその特徴量をどれだけ当てにしているか」を測ります。"
     "式の特化度で PFI +0.617 とアブレーション +0.090 が食い違うのは、この違いによるものです。"),
]


def place_note(page, anchor_rect, title, body):
    """右余白に付箋（テキスト注釈）を置く."""
    x = page.rect.width - 26
    y = max(28, min(anchor_rect.y0, page.rect.height - 40))
    annot = page.add_text_annot(fitz.Point(x, y), body, icon="Comment")
    annot.set_info(title=title, content=body)
    annot.set_colors(stroke=COLOR_NOTE)
    annot.update()


def main() -> int:
    doc = fitz.open(SRC)
    placed, missed = 0, []

    for i, (anchor, who, comment, fix) in enumerate(REVISIONS, 1):
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

        src = "加藤先生ご指摘" if who == "加藤" else "一洋の見直し"
        title = f"修正 #{i:02d}（{src}）"
        body = f"{title}\n\n■ ご指摘・要望\n{comment}\n\n■ 修正内容と理由\n{fix}"
        place_note(page, rects[0], title, body)
        placed += 1
        print(f"  #{i:02d} p{pno+1} OK  {anchor[:32]}")

    for i, anchor in missed:
        print(f"  #{i:02d} --  MISS {anchor[:32]}")

    doc.save(OUT)
    print(f"\n注釈 {placed}/{len(REVISIONS)} 件を配置 → {OUT.name}")
    return 1 if missed else 0


if __name__ == "__main__":
    raise SystemExit(main())
