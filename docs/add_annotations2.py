"""Fix failed annotations with shorter search strings."""
import fitz

PDF_PATH = "/Users/kazuhiromiyamura/Desktop/AutoPMob/docs/progress_report_2026-04-13_v2_annotated.pdf"

ANNOTATIONS = [
    ("Semantic Scholar APIにより自動取得した論文群",
     "追加（コメント対応）: #15-20が個別記載でない理由を注記。次回報告で正式な著者名・タイトルを記載する旨を明記。",
     0),
    ("Round 1で未カバー",
     "追加（コメント対応）: Round 2が「応用式」である理由を説明。Round 1の基礎に対し、未カバー分野+応用方程式の追加。",
     0),
    ("著作権リスク",
     "追加（コメント対応）: 著作権のあるPDFのアップロードにおけるリスクを明記。",
     0),
    ("出力結果を研究に使用",
     "追加（コメント対応）: Googleの出力を研究使用することの可否を明確化。",
     0),
    ("著作権のあるPDF",
     "追加（コメント対応）: PDF抽出時の著作権リスク対策（有料枠使用）を注記。",
     1),  # 2nd occurrence (first is in 2.4)
    ("851K文字",
     "修正（コメント対応）: 「キャラクター」→「文字」に修正。",
     0),
    ("swap_io",
     "追加（コメント対応）: swap_ioは入出力変数数が等しいケースのみ適用する制約を明記。",
     2),  # footnote occurrence
    ("random_io",
     "追加（コメント対応）: random_ioの求解可能性が未検証である問題を明記。自由度解析の導入を今後の課題に。",
     2),  # footnote occurrence
    ("cosine類似度",
     "追加（コメント対応）: 式(1)の各項の意味を詳細に説明。",
     0),
    ("予備実験",
     "修正（フィードバック対応）: 旧4.2の(A)(B)を「予備実験」に移動。結果表の3手法と混同しない構成に変更。",
     0),
]

doc = fitz.open(PDF_PATH)
applied = 0
failed = []

for search_text, comment, occurrence in ANNOTATIONS:
    found_count = 0
    target_found = False
    for page_num in range(len(doc)):
        page = doc[page_num]
        instances = page.search_for(search_text)
        for inst in instances:
            if found_count == occurrence:
                annot = page.add_highlight_annot(inst)
                annot.set_info(content=comment, title="revision")
                annot.set_colors(stroke=(1.0, 0.8, 0.0))
                annot.update()
                applied += 1
                target_found = True
                break
            found_count += 1
        if target_found:
            break
    if not target_found:
        failed.append(search_text[:50])

doc.save(PDF_PATH, incremental=True, encryption=0)
doc.close()

print(f"Applied {applied} additional annotations, {len(failed)} failed")
if failed:
    print("Failed:")
    for f in failed:
        print(f"  - {f}")
