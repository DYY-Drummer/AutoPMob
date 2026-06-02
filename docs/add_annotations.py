"""Add highlight annotations to the revised PDF to mark modified sections."""
import fitz  # pymupdf

PDF_PATH = "/Users/kazuhiromiyamura/Desktop/AutoPMob/docs/progress_report_2026-04-13_v2.pdf"
OUTPUT_PATH = "/Users/kazuhiromiyamura/Desktop/AutoPMob/docs/progress_report_2026-04-13_v2_annotated.pdf"

# (search_text, comment, match_occurrence) - match_occurrence 0=first, 1=second, etc.
ANNOTATIONS = [
    # --- Section 2.1: 変数表記揺れ ---
    (
        "変数の表記揺れへの対処",
        "追加（コメント対応）: 変数の表記揺れ（例：c_p vs c_w）の扱い方針を追記。完全一致で判定し、誤った統合を回避する安全側の設計としている旨を説明。",
        0,
    ),
    # --- Section 2.2: 参考文献 ---
    (
        "同一書籍を複数行に分けて",
        "追加（コメント対応）: #1と#2、#8と#9が同一書籍の分割チャンクであることを明記。",
        0,
    ),
    (
        "同上，Chap. 2",
        "修正（コメント対応）: #2が#1の一部（別チャンク）であることを明示。",
        0,
    ),
    (
        "同上，Chap. 1/1",
        "修正（コメント対応）: #9が#8の一部（別チャンク）であることを明示。",
        0,
    ),
    (
        "#15--20はSemantic Scholar API",
        "追加（コメント対応）: #15-20が個別記載でない理由を注記。次回報告で正式な著者名・タイトルを記載する旨を明記。",
        0,
    ),
    # --- Section 2.3: ハンドブック式 ---
    (
        "「ハンドブック式」の定義",
        "追加（コメント対応）: ハンドブック式とPDFソース数式の違いを明確化。Perry's Handbook等の普遍的工学知識であり、分野網羅性の補強が目的。",
        0,
    ),
    (
        "ドメインの選択基準",
        "追加（コメント対応）: 15/16ドメインの選択基準を追記。化学工学カリキュラムとPerry's Handbookの章構成を参考。",
        0,
    ),
    (
        "Round 1で未カバーの応用分野",
        "追加（コメント対応）: Round 2が「応用式」である理由を説明。Round 1の基礎に対し、未カバー分野+応用方程式の追加。",
        0,
    ),
    # --- Section 2.4: API利用 ---
    (
        "有料枠を使用する",
        "修正（コメント対応）: 「確認すべき事項」から「対応方針」に変更。有料枠使用を明確に方針化。",
        0,
    ),
    (
        "PDFアップロード時の著作権リスク",
        "追加（コメント対応）: 著作権のあるPDFのアップロードにおけるリスクを明記。無料枠での使用は著作権問題になりうる。",
        0,
    ),
    (
        "APIの出力結果を研究に使用すること自体は禁止されていない",
        "追加（コメント対応）: Googleの出力を研究使用することの可否を明確化。",
        0,
    ),
    # --- Section 3.1: 著作権注意 ---
    (
        "著作権のあるPDFのアップロードに際しては",
        "追加（コメント対応）: PDF抽出時の著作権リスク対策（有料枠使用）を注記。",
        0,
    ),
    # --- Section 3.2: ドメイン準備 ---
    (
        "ドメイン名・トピックリストの準備方法",
        "追加（コメント対応）: ドメイン名・トピックリストの準備方法を説明。現時点では手動設計、将来は自動推定パイプライン構築が課題。",
        0,
    ),
    # --- Section 3.3: 検索クエリ ---
    (
        "検索カテゴリの設計基準",
        "追加（コメント対応）: 検索カテゴリの設計基準を追記。プロセス制御偏りを補うための分野選定。カテゴリ数・件数は経験的設定。",
        0,
    ),
    # --- Section 3.4.1: 用語修正 ---
    (
        "851K文字",
        "修正（コメント対応）: 「キャラクター」→「文字」に修正。",
        0,
    ),
    (
        "数式一覧（model_id",
        "修正（コメント対応）: 「カタログ」→「数式一覧」に修正。",
        0,
    ),
    # --- Section 3.4.2: swap_io/random_io ---
    (
        "swap_ioの制約",
        "追加（コメント対応）: swap_ioは入出力変数数が等しいケースのみ適用。不一致の場合は除外している旨を明記。",
        0,
    ),
    (
        "random_ioの制約",
        "追加（コメント対応）: random_ioの求解可能性が未検証である問題を明記。自由度解析の導入を今後の課題に。",
        0,
    ),
    # --- Section 4: 全面再構成 ---
    (
        "結果表での名称",
        "修正（フィードバック対応）: セクション4を全面再構成。結果表の3手法(baseline/reranker-7/reranker-10)との対応を冒頭に明示する表を追加。",
        0,
    ),
    (
        "用語の説明",
        "追加（コメント対応）: TF-IDF, SVD, Jaccard係数, L2正規化, cosine類似度の説明を新設。特にTF-IDFではtermとdocumentが何に対応するかを明記。",
        0,
    ),
    (
        "グラフの役割",
        "追加（コメント対応）: 二部グラフがどの手法で・どのように使われるかを明記（reranker-10のグラフ特徴量計算にのみ使用）。",
        0,
    ),
    (
        "候補抽出（全手法共通）",
        "修正（フィードバック対応）: baselineが何かを明確化。Stage 1のスコアをそのまま最終ランキングとして使う手法がbaseline。",
        0,
    ),
    (
        "cosine類似度，Jaccard",
        "追加（コメント対応）: 式(1)の各項が何を計算しているか詳細に説明。",
        0,
    ),
    (
        "訓練データにおける正例・負例",
        "追加（コメント対応）: 正例・負例の定義、選択方法、負例数min(8,...)が少数にならない理由を説明。",
        0,
    ),
    (
        "予備実験：直接GNN学習",
        "修正（フィードバック対応）: 旧4.2の(A) GCN Retriever, (B) 残差GCN SVDを「予備実験」セクションに移動。結果表と混同しない構成に変更。各アプローチの入出力・失敗原因を説明。",
        0,
    ),
    # --- Section 5: 太字修正 ---
    (
        "0.949",
        "修正（コメント対応）: 太字を最良性能のreranker-10に統一（旧版ではreranker-7の0.937が太字だった）。",
        0,
    ),
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
                # Add highlight annotation
                annot = page.add_highlight_annot(inst)
                annot.set_info(content=comment, title="revision")
                annot.set_colors(stroke=(1.0, 0.8, 0.0))  # yellow
                annot.update()
                applied += 1
                target_found = True
                break
            found_count += 1
        if target_found:
            break
    if not target_found:
        failed.append(search_text[:40])

doc.save(OUTPUT_PATH)
doc.close()

print(f"Applied {applied} annotations, {len(failed)} failed")
if failed:
    print("Failed searches:")
    for f in failed:
        print(f"  - {f}")
