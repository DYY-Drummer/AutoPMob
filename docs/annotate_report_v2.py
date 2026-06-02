"""
進捗レポート v2 に対して、提出版 PDF と v2 PDF を直接比較し、
実際に変更/追加された箇所のみをハイライトする。

比較方法:
  単語レベル（pymupdf の get_text("words")）の SequenceMatcher 差分を取り、
  「提出版になく、v2 にある」単語のみを変更・追加とみなしてハイライトする。

加えて、各修正内容を説明するスティッキーノート（付箋）を、
該当セクション付近に配置する。

入力:
  docs/4.13データ改善による性能向上レポート.pdf (加藤先生に提出済み)
  docs/progress_report_2026-04-13_v2.pdf (改訂版)

出力:
  docs/progress_report_2026-04-13_v2_annotated.pdf
"""
import difflib
import fitz  # pymupdf
import re

SUBMITTED_PDF = "/Users/kazuhiromiyamura/Desktop/AutoPMob/docs/4.13データ改善による性能向上レポート.pdf"
V2_PDF = "/Users/kazuhiromiyamura/Desktop/AutoPMob/docs/progress_report_2026-04-13_v2.pdf"
OUT_PDF = "/Users/kazuhiromiyamura/Desktop/AutoPMob/docs/progress_report_2026-04-13_v2_annotated.pdf"

# ハイライト色
COLOR_HIGHLIGHT = (1.0, 0.85, 0.2)  # 濃い黄色（追加・変更を統一色）


import unicodedata

# 表記揺れを吸収する文字置換マップ
# （提出版と v2 の間で LaTeX 設定変更があった場合に false diff を防ぐ）
_NOTATION_MAP = {
    # 各種ダッシュ/ハイフンを ASCII ハイフンに統一
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash  (LaTeX -- の結果)
    "\u2014": "-",  # em dash  (LaTeX --- の結果)
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
    "\uff0d": "-",  # fullwidth minus
    # 各種引用符を ASCII に統一
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    # 各種スペース類を通常スペースに（\s+ で後段除去されるが念のため）
    "\u00a0": " ",  # NBSP
    "\u3000": " ",  # 全角スペース
    "\u2002": " ", "\u2003": " ", "\u2009": " ", "\u200a": " ",
    # 全角句読点 → 半角（LaTeX 設定で揺れる場合の保険）
    "，": ",", "．": ".",
    "（": "(", "）": ")",
    "：": ":", "；": ";",
    "【": "[", "】": "]",
}


def normalize_word(w):
    """単語の表記揺れを吸収する正規化.

    - Unicode NFC 正規化（結合文字の差を吸収）
    - ダッシュ類・引用符類・各種空白類を正規形に統一
    - 全角句読点 → 半角（設定揺れ対策）
    - 内部空白の全除去
    """
    if w is None:
        return ""
    # NFC 正規化
    w = unicodedata.normalize("NFC", w)
    # 表記揺れの置換
    w = w.translate(str.maketrans(_NOTATION_MAP))
    # 内部空白を全て除去
    return re.sub(r"\s+", "", w)


def _is_page_number(text, y0, page_height, bottom_margin=80):
    """下余白にある純数字（ページ番号）かどうかを判定."""
    return (
        text.isdigit()
        and len(text) <= 3
        and y0 > page_height - bottom_margin
    )


def extract_word_sequence(pdf_path):
    """PDF から単語列と位置情報を取り出す.

    ページ番号（下余白の純数字）は diff 対象・ハイライト対象から除外する。

    Returns:
        words_norm: list of normalized word strings
        word_info:  list of (page_num, fitz.Rect, original_word, line_id) tuples
                    line_id は (page_num, block_no, line_no) の三つ組で、
                    pymupdf が識別する視覚的な同一行を一意に表す。
    """
    doc = fitz.open(pdf_path)
    words_norm = []
    word_info = []
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        for w in page.get_text("words"):
            # w: (x0, y0, x1, y1, text, block_no, line_no, word_no)
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            block_no = w[5] if len(w) > 5 else 0
            line_no = w[6] if len(w) > 6 else 0
            # ページ番号を除外
            if _is_page_number(text, y0, page_height):
                continue
            norm = normalize_word(text)
            if not norm:
                continue
            words_norm.append(norm)
            line_id = (page_num, block_no, line_no)
            word_info.append((page_num, fitz.Rect(x0, y0, x1, y1), text, line_id))
    doc.close()
    return words_norm, word_info


def find_modified_rects(submitted_pdf, v2_pdf):
    """提出版と v2 の単語列を比較し、v2 側で変更/追加された単語の rect を返す.

    変更されなかった単語（equal）が隣り合う「変更単語」の間に無い場合、
    連続する変更単語は同一行でマージ済みの大きな rect として返す。
    これによりハイライトが視覚的に途切れない。

    Returns:
        modified: list of (page_num, fitz.Rect) for lines/runs to highlight
    """
    sub_words, _ = extract_word_sequence(submitted_pdf)
    v2_words, v2_info = extract_word_sequence(v2_pdf)
    print(f"Submitted words: {len(sub_words)}")
    print(f"v2 words:        {len(v2_words)}")

    matcher = difflib.SequenceMatcher(a=sub_words, b=v2_words, autojunk=False)

    # v2 側の各インデックスが「変更されたか」のフラグを作る
    n_v2 = len(v2_words)
    is_modified = [False] * n_v2
    summary = {"equal": 0, "insert": 0, "replace": 0, "delete": 0}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        summary[tag] = summary.get(tag, 0) + (j2 - j1 if tag in ("insert", "replace", "equal") else 0)
        if tag in ("insert", "replace"):
            for idx in range(j1, j2):
                is_modified[idx] = True
    print(f"Diff opcodes: equal={summary['equal']}, "
          f"insert={summary['insert']}, replace={summary['replace']}")

    # 行単位で集計し、「変更率が高い行」は丸ごとハイライトする。
    # 変更率が低い行は変更単語のみをハイライト（精度優先）。
    LINE_COVERAGE_THRESHOLD = 0.5     # 行内の変更単語比率がこの値以上なら行丸ごとハイライト

    # Step 1: 単語を line_id でグループ化
    #   line_id は pymupdf の (page, block_no, line_no) で、視覚的な同一行を一意に識別。
    #   これにより日本語ベースラインと Latin/monospace ベースラインの差で
    #   同じ行の単語が別行扱いされる問題を回避する。
    lines: dict[tuple, list[tuple[int, bool]]] = {}  # line_id -> [(v2_idx, is_modified)]
    line_order: list[tuple] = []  # 挿入順保持
    for idx in range(n_v2):
        _, _, _, line_id = v2_info[idx]
        if line_id not in lines:
            lines[line_id] = []
            line_order.append(line_id)
        lines[line_id].append((idx, is_modified[idx]))

    modified: list[tuple[int, fitz.Rect]] = []

    for line_id in line_order:
        word_indices = lines[line_id]
        page_num = line_id[0]
        n_total = len(word_indices)
        n_mod = sum(1 for _, m in word_indices if m)
        if n_mod == 0:
            continue
        coverage = n_mod / n_total

        # 行内の単語の中央値 y 範囲を計算し、外れ値（数式・下付き文字等）が
        # 次の行に食い込むのを防ぐ
        all_rects = [v2_info[i][1] for i, _ in word_indices]
        import statistics
        median_y0 = statistics.median(r.y0 for r in all_rects)
        median_y1 = statistics.median(r.y1 for r in all_rects)

        def bounded_rect(rects_subset):
            """y 範囲をメディアンで揃え、行間オーバーラップを防ぐ.

            y1 を 1.5pt 縮めることで、隣接行のディセンダー/アセンダー部分との
            微小オーバーラップを排除する（見た目は変わらず重なりのみ消える）。
            """
            LINE_SHRINK = 1.5
            return fitz.Rect(
                min(r.x0 for r in rects_subset),
                median_y0,
                max(r.x1 for r in rects_subset),
                median_y1 - LINE_SHRINK,
            )

        if coverage >= LINE_COVERAGE_THRESHOLD:
            # 行全体をハイライト
            modified.append((page_num, bounded_rect(all_rects)))
        else:
            # 変更単語のみをハイライト（連続する変更単語はまとめる）
            current_run: list[fitz.Rect] = []

            def flush_run():
                if current_run:
                    modified.append((page_num, bounded_rect(current_run)))

            for idx, is_mod in word_indices:
                rect = v2_info[idx][1]
                if is_mod:
                    current_run.append(rect)
                else:
                    flush_run()
                    current_run = []
            flush_run()

    return modified


# =============================================================================
# スティッキーノート定義（= 加藤先生コメントへの対応内容）
# =============================================================================
# 各ノートは:
#   section: セクション番号
#   title:   短いタイトル
#   comment: 本文（ノート本体に表示）
#   anchor:  付箋の配置を決めるためのキーワード（該当セクション内のユニークな文字列）
#
# anchor キーワードは、v2 PDF 内でそのセクションを特定できるユニークな短い文字列を選ぶ。
# (ハイライトには使わない。付箋の配置位置決定のみに使用)
# =============================================================================

NOTES = [
    {
        "section": "2.1",
        "title": "変数表記揺れへの対処方針（拡充）",
        "anchor": "変数の表記揺れへの対処方針",
        "comment": (
            "加藤先生への前回約束事項『変数表記の正規化』について、\n"
            "方針変更の説明です。実データ精査のうえ、全面正規化は行わない\n"
            "判断に変更した経緯と根拠を本文に記載しました。"
        ),
    },
    {
        "section": "2.2",
        "title": "分割抽出に関する注記（追加）",
        "anchor": "同一書籍を複数行に分けて記載しているものは",
        "comment": "加藤先生指摘『同一書籍が複数行に分かれている理由が不明』への回答です。",
    },
    {
        "section": "2.2",
        "title": "参考文献 #2/#9 の表記修正",
        "anchor": "別チャンクで分割抽出",
        "comment": "加藤先生指摘『#2/#9 が #1/#8 の分割抽出チャンクであることの明示』への対応です。",
    },
    {
        "section": "2.2",
        "title": "#15-20 に※マーカー＋脚注を追加",
        "anchor": "Semantic Scholar APIにより自動取得",
        "comment": (
            "加藤先生指摘『参考文献は個別に記載すべき』への暫定対応です。\n"
            "次回報告で正式な著者名・タイトルを記載する旨を脚注で約束しました。"
        ),
    },
    {
        "section": "2.2",
        "title": "燃焼論文が3本の理由（†脚注）",
        "anchor": "燃焼論文が3本である理由",
        "comment": (
            "加藤先生指摘『なぜ燃焼関連論文だけ3本選んだ？』への回答です。\n"
            "件数は事前指定ではなく、Semantic Scholar API検索（20-30件）の後に\n"
            "Open-access・数式含有量でフィルタした結果であることを†脚注で説明しました。"
        ),
    },
    {
        "section": "2.2",
        "title": "#20「その他」を個別10論文に展開",
        "anchor": "applsci-10-00992-v2",
        "comment": (
            "加藤先生指摘『その他も一つ一つ書くべき』への対応です。\n"
            "旧「#20 その他（50式）」を10論文（#20–#29、計140式）に個別展開しました。\n"
            "3論文は書誌情報が未特定のため脚注でその旨を明記しています。"
        ),
    },
    {
        "section": "2.3",
        "title": "「ハンドブック式」の定義を追加",
        "anchor": "「ハンドブック式」の定義",
        "comment": "加藤先生指摘『ハンドブック式の定義が不明確』への回答です。",
    },
    {
        "section": "2.3",
        "title": "ドメイン選択基準を追加",
        "anchor": "ドメインの選択基準",
        "comment": "加藤先生指摘『15/16 ドメインをどう選んだかの根拠が不明』への回答です。",
    },
    {
        "section": "2.3",
        "title": "Round 2 の位置づけ明確化",
        "anchor": "Round 2（16ドメイン，応用式）",
        "comment": "加藤先生指摘『Round 2 の位置づけが不明』への回答です。",
    },
    {
        "section": "2.4",
        "title": "API 利用を「確認事項」→「対応方針」に明確化",
        "anchor": "対応方針",
        "comment": "加藤先生指摘『無料枠のリスクへの対応方針を明確化すべき』への対応です。",
    },
    {
        "section": "2.4",
        "title": "API 利用 4 項目の全面書き換え",
        "anchor": "PDFアップロード時の著作権リスク",
        "comment": (
            "加藤先生指摘『API 利用時のリスクと対応方針の明確化』への回答です。\n"
            "曖昧な 3 項目を明確な対応方針 4 項目に再構成しました。"
        ),
    },
    {
        "section": "3.1",
        "title": "PDF 抽出時の著作権注記（追加）",
        "anchor": "著作権のあるPDFのアップロードに際しては",
        "comment": "加藤先生指摘『PDF アップロード時の著作権リスク』への対応です。",
    },
    {
        "section": "3.2",
        "title": "Gemini API 制約のタイトル修正",
        "anchor": "Gemini APIの制約と対処",
        "comment": "有料枠方針化（2.4 節）に合わせたタイトル表現の統一です。",
    },
    {
        "section": "3.2",
        "title": "ページ上限処理の詳細化",
        "anchor": "大型テキストブック",
        "comment": "加藤先生指摘『ページ上限処理の具体的な流れが不明』への回答です。",
    },
    {
        "section": "3.2",
        "title": "ドメイン/トピックリストの準備方法（追加）",
        "anchor": "ドメイン名・トピックリストの準備方法",
        "comment": "加藤先生指摘『ドメイン名・トピックリストの準備方法が不明』への回答です。",
    },
    {
        "section": "3.3",
        "title": "検索カテゴリ設計基準の明記",
        "anchor": "検索カテゴリの設計基準",
        "comment": "加藤先生指摘『検索カテゴリ・件数の設計根拠が不明』への回答です。",
    },
    {
        "section": "3.4.1",
        "title": "用語修正: キャラクター/カタログ",
        "anchor": "数式一覧",
        "comment": "加藤先生指摘『カタカナ英語を避ける』への対応です。",
    },
    {
        "section": "3.4.2",
        "title": "swap_io / random_io の制約脚注（追加）",
        "anchor": "swap\\_ioの制約",
        "comment": "加藤先生指摘『swap_io／random_io の妥当性（本当に解けるのか）』への回答です。",
    },
    {
        "section": "4",
        "title": "章タイトル変更",
        "anchor": "検索手法の設計",
        "comment": (
            "加藤先生フィードバック対応：旧タイトルだと GNN がメイン手法と\n"
            "誤解されるため、実態に合わせた章タイトルに変更しました。"
        ),
    },
    {
        "section": "4.1",
        "title": "3 手法と結果表の対応表を新設",
        "anchor": "結果表での名称",
        "comment": "加藤先生指摘『手法セクションと結果表の対応が不明瞭』への回答です。",
    },
    {
        "section": "4.2",
        "title": "用語説明セクション新設",
        "anchor": "用語の説明",
        "comment": (
            "加藤先生指摘『TF-IDF／SVD／Jaccard 等の用語説明が不足』への回答です。\n"
            "特に TF-IDF については本研究での Term／Document の対応を明記しました。"
        ),
    },
    {
        "section": "4.3",
        "title": "グラフの使用範囲を明記",
        "anchor": "グラフの役割",
        "comment": "加藤先生指摘『二部グラフがどの手法で・どう使われるのか不明』への回答です。",
    },
    {
        "section": "4.4",
        "title": "Stage 1 候補抽出の明示",
        "anchor": "Stage 1: 候補抽出（全手法共通）",
        "comment": "加藤先生指摘『baseline の定義・Stage 1 スコアの意味が不明』への回答です。",
    },
    {
        "section": "4.5",
        "title": "正例・負例の定義と min(8,...) の理由",
        "anchor": "訓練データにおける正例・負例",
        "comment": "加藤先生指摘『正例・負例の定義と負例数 min(8,...) の理由』への回答です。",
    },
    {
        "section": "4.6",
        "title": "旧 (A)(B) を「予備実験」に移動",
        "anchor": "予備実験：直接GNN学習によるアプローチ",
        "comment": (
            "加藤先生フィードバック対応：結果表の 3 手法と混同を招く (A)(B) を\n"
            "「予備実験」セクションに分離し、主要比較対象から明確に区別しました。"
        ),
    },
    {
        "section": "5",
        "title": "結果表の太字を最良性能に統一",
        "anchor": "3,244式において，reranker-10",
        "comment": "加藤先生指摘『最良値のみ太字にすべき（太字の基準が不統一）』への修正です。",
    },
]


# =============================================================================
# ハイライト rect の後処理（同一行内で連続する rect をマージして見栄えを良くする）
# =============================================================================

def merge_adjacent_rects(rects_by_page):
    """同一ページの同一行で水平方向に近接する rect をマージする.

    Args:
        rects_by_page: dict of page_num -> list of fitz.Rect
    Returns:
        dict of page_num -> list of merged fitz.Rect
    """
    merged_all = {}
    for pno, rects in rects_by_page.items():
        if not rects:
            merged_all[pno] = []
            continue
        # y 座標でグループ化（行ごと）
        rects_sorted = sorted(rects, key=lambda r: (round(r.y0, 1), r.x0))
        merged = []
        current = rects_sorted[0]
        LINE_TOL = 3.0  # 同一行とみなす y 差の許容値
        # 水平ギャップ許容値: 浮動小数点精度の問題や微小な単語間隔のみ埋める。
        # 日本語1文字の幅(~10pt)以下に抑えることで、未変更単語を挟む場合には
        # マージしない（その場合は先に find_modified_rects で分割済み）
        GAP_TOL = 9.0
        for r in rects_sorted[1:]:
            same_line = abs(r.y0 - current.y0) < LINE_TOL and abs(r.y1 - current.y1) < LINE_TOL
            if same_line and (r.x0 - current.x1) < GAP_TOL:
                # マージ
                current = fitz.Rect(current.x0, min(current.y0, r.y0),
                                    max(current.x1, r.x1), max(current.y1, r.y1))
            else:
                merged.append(current)
                current = r
        merged.append(current)
        merged_all[pno] = merged
    return merged_all


# =============================================================================
# スティッキーノートの配置
# =============================================================================

def _generate_anchor_variants(s):
    """PDF レンダリング時のゆらぎを吸収する多様なバリアントを生成."""
    # 1) LaTeX マークアップ由来のエスケープを解除
    s = s.replace("\\#", "#").replace("\\_", "_").replace("\\&", "&")
    base = {s}
    for x in list(base):
        if "~" in x:
            base.add(x.replace("~", " "))
        if "--" in x:
            base.add(x.replace("--", "\u2013"))
    # 2) 両方向の変換
    for x in list(base):
        if "~" in x:
            base.add(x.replace("~", " "))
        if "--" in x:
            base.add(x.replace("--", "\u2013"))
    # 3) ASCII↔JA 境界のスペース挿入（片方向・両方向）
    ja_re = r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]'
    ascii_re = r'[A-Za-z0-9/()\-#]'
    expanded = set(base)
    for x in base:
        v1 = re.sub(f'({ascii_re})({ja_re})', r'\1 \2', x)
        v2 = re.sub(f'({ja_re})({ascii_re})', r'\1 \2', x)
        v3 = re.sub(f'({ja_re})({ascii_re})', r'\1 \2', v1)
        expanded.update([v1, v2, v3])
    return [v for v in expanded if v]


def find_anchor_rect(doc, anchor_text):
    """anchor_text を含む最初のページ上の矩形を探す（付箋配置用）."""
    variants = _generate_anchor_variants(anchor_text)
    for page_num, page in enumerate(doc):
        for v in variants:
            try:
                hits = page.search_for(v)
            except Exception:
                continue
            if hits:
                return page_num, hits[0]
    return None, None


def place_sticky_note(page, anchor_rect, title, body):
    """付箋を anchor_rect の右側マージンに配置."""
    page_rect = page.rect
    note_x = page_rect.width - 30
    note_y = max(30, min(anchor_rect.y0, page_rect.height - 30))
    point = fitz.Point(note_x, note_y)
    annot = page.add_text_annot(point, body, icon="Comment")
    annot.set_info(title=title, content=body)
    annot.set_colors(stroke=(1.0, 0.6, 0.0))
    annot.update()


# =============================================================================
# メイン処理
# =============================================================================

def main():
    print("=== Step 1: 単語レベル差分で変更箇所を検出 ===")
    modified = find_modified_rects(SUBMITTED_PDF, V2_PDF)
    print(f"Modified words:  {len(modified)}")

    # ページ別にグルーピング → 隣接 rect をマージ
    rects_by_page = {}
    for pno, rect in modified:
        rects_by_page.setdefault(pno, []).append(rect)
    merged = merge_adjacent_rects(rects_by_page)
    total_merged = sum(len(v) for v in merged.values())
    print(f"Merged rects:    {total_merged}")

    print()
    print("=== Step 2: ハイライト適用 ===")
    doc = fitz.open(V2_PDF)
    highlight_count = 0
    for pno, rects in merged.items():
        page = doc[pno]
        for rect in rects:
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=COLOR_HIGHLIGHT)
            annot.update()
            highlight_count += 1
    print(f"Highlights:      {highlight_count}")

    print()
    print("=== Step 3: スティッキーノート配置 ===")
    placed = 0
    for i, note in enumerate(NOTES, start=1):
        title = note['title']
        body = note['comment']
        pno, rect = find_anchor_rect(doc, note["anchor"])
        if rect is not None:
            place_sticky_note(doc[pno], rect, title, body)
            placed += 1
            print(f"  #{i:02d} §{note['section']} {note['title']}: p{pno+1} OK")
        else:
            print(f"  #{i:02d} §{note['section']} {note['title']}: ANCHOR NOT FOUND "
                  f"[{note['anchor'][:30]}]")

    doc.save(OUT_PDF)
    doc.close()

    print()
    print(f"=== Summary ===")
    print(f"Highlights:     {highlight_count}")
    print(f"Sticky notes:   {placed}/{len(NOTES)}")
    print(f"Saved:          {OUT_PDF}")


if __name__ == "__main__":
    main()
