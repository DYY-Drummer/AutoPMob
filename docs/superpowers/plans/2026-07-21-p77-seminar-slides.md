# P-77 紹介スライド（研究室ゼミ・8枚PPTX）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PSE Asia 2026 のポスター P-77 を研究室ゼミで紹介する日本語スライド8枚（ノート欄に台本つき）を、既存の学会テンプレを流用して python-pptx で生成する。

**Architecture:** 本文・台本・検証ルールをデータモジュール `content_p77.py` に分離し、`build_p77_slides.py` がテンプレ pptx を複製→既存14枚を全削除→8枚を描画して出力する。pytest が生成物を開き直して仕様（20pt・略称併記・禁止語・台本分量）を機械検証する。

**Tech Stack:** Python 3 / python-pptx 1.0.2 / pytest 9.0.2（いずれも導入済み確認済）

## Global Constraints

- 仕様書: `docs/superpowers/specs/2026-07-21-p77-seminar-slides-design.md`（本計画の上位文書）
- テンプレ入力（読み取りのみ・変更禁止）: `/Users/kazuhiromiyamura/Desktop/学会/PSE ASIAN2026/PSEAsia2026_slide_v2.pptx`（16:9 = 12192000×6858000 EMU、レイアウトは index 0 `DEFAULT` の1種のみ、フォントは IBM Plex ファミリ）
- 最終出力: `/Users/kazuhiromiyamura/Desktop/ゼミ/論文紹介_P77_LLM_Aspen_かずひろ.pptx`（リポジトリ外・コミットしない）
- スライドは **8枚・日本語**。**スライド上の全テキスト（図中・表・出典含む）は明示指定の 20pt 以上**（ノート欄の台本は表示されないため対象外）
- **略称はそれが登場する各スライド内に正式名称を併記**（MCP=Model Context Protocol、LLM=大規模言語モデル、COM=Component Object Model、CSTR=Continuous Stirred-Tank Reactor、PSE=Process Systems Engineering）
- **難解用語はその場で1行説明**（Aspen Plus・反応セット・backup ファイル）。以下の語は**使用禁止**: フローシート, RAG, TEA, LCA, LNS, MINLP, power-law, トークン
- **格上げ表現の禁止**: 実証, 大幅, 劇的, 画期的, 世界初（成果は抄録の "confirms correct functionality" に合わせ「動作確認」と表現）
- 図は自作3点のみ（MCP 概念図・P-77 構成図・×○回避ルート図）。P-77 の一次図は未公開（仕様書に調査記録あり）
- 台本: 各スライド 250〜500字、合計 2300〜4000字（約12分）
- 色は テンプレ実測パレットから: NAVY `#16304F` / TEXT `#222222` / BLUE `#2E86C1` / GREEN `#2E8851` / GRAY `#8C8C8C` / LIGHT `#EAF3FB` / LIGHT2 `#DCEAF6`、×印のみ RED `#C0392B`
- フォント: 欧文は本文 `IBM Plex Sans`・見出し `IBM Plex Sans SemiBold`（テンプレ準拠）。IBM Plex に和文グリフは無いため、**和文は全 run に `Hiragino Kaku Gothic ProN` を East Asian フォントとして明示指定**（macOS 標準・ゴシック体）
- コミットは各タスク末尾で実施（post-commit フックが自動 push）。メッセージ末尾に `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

### Task 1: 本文・台本データモジュールと内容検証テスト

**Files:**
- Create: `scripts/seminar_p77/content_p77.py`
- Test: `tests/test_p77_content.py`

**Interfaces:**
- Produces: `SLIDES: list[dict]`（8要素。各 dict は `kind` と `notes` を必ず持つ。kind 別の追加キーは下記コードの通り）、`ABBREV_FULL: dict[str,str]`、`FORBIDDEN_WORDS: list[str]`、`REQUIRED_PAIRS: list[tuple[int,str,str]]`、`slide_all_text(idx) -> str`（そのスライドに載る全文字列を結合して返す）
- Consumes: なし（純データ）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_p77_content.py` を以下の内容で作成:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python3 -m pytest tests/test_p77_content.py -v`
Expected: 全テスト ERROR（`ModuleNotFoundError: No module named 'content_p77'`）

- [ ] **Step 3: データモジュールを実装**

`scripts/seminar_p77/content_p77.py` を以下の内容で作成（**この本文・台本が最終原稿**。改変する場合も Global Constraints を守ること）:

```python
# -*- coding: utf-8 -*-
"""P-77 紹介スライド（研究室ゼミ）の本文・台本・検証ルール。

仕様: docs/superpowers/specs/2026-07-21-p77-seminar-slides-design.md
このモジュールは純データ。描画は build_p77_slides.py が行う。
"""

# 略称 → そのスライド内に必須の正式名称（略称が本文に出る全スライドでチェック）
ABBREV_FULL = {
    "MCP": "Model Context Protocol",
    "LLM": "大規模言語モデル",
    "COM": "Component Object Model",
    "CSTR": "Continuous Stirred-Tank Reactor",
    "PSE": "Process Systems Engineering",
}

# スライド・台本での使用禁止語（難解用語と格上げ表現）
FORBIDDEN_WORDS = [
    "フローシート", "RAG", "TEA", "LCA", "LNS", "MINLP", "power-law", "トークン",
    "実証", "大幅", "劇的", "画期的", "世界初",
]

# (slide_index0始まり, 用語, 同スライドに必須の説明文字列)
REQUIRED_PAIRS = [
    (0, "Aspen Plus", "化学プロセスシミュレータ"),
    (3, "Aspen Plus", "化学プロセスシミュレータ"),
    (3, "COM", "Windows のソフト同士を繋ぐ仕組み"),
    (3, "CSTR", "連続槽型反応器"),
    (5, "反応セット", "反応式や速度定数の登録情報"),
    (5, "backup ファイル", "テキスト形式の保存ファイル"),
]

SLIDES = [
    # ---- 1. タイトル -------------------------------------------------
    {
        "kind": "title",
        "headline": "LLM（大規模言語モデル）に Aspen Plus を操作させる",
        "sub": "PSE ASIA 2026 で見つけた MCP（Model Context Protocol）活用研究の紹介",
        "body": [
            "紹介する研究（ポスター P-77）",
            "Large Language Model Assisted Automation for Reactor Process "
            "Simulation, Synthesis, and Optimization",
            "Ting-Yen Chiang, Yu-Sheng Chen, Hao-Yeh Lee, Yan-Ling Yang"
            "（国立台湾科技大学）",
            "PSE ASIA 2026（12th Asian Symposium on Process Systems Engineering、"
            "ソウル・延世大学）ポスター発表（2026年7月7日）",
        ],
        "explain": "※ Aspen Plus：化学プロセスシミュレータ"
        "（プラントの物質・熱の流れを計算する定番の商用ソフト）",
        "footer": "紹介者：丁 一洋 ／ 2026年7月 研究室ゼミ",
        "notes": (
            "今日は、7月に参加した PSE ASIA 2026 から、面白かった研究を1つ共有します。"
            "台湾科技大のグループのポスター発表で、ひとことで言うと「ChatGPT のような"
            "大規模言語モデルに、化学プロセスシミュレータの Aspen Plus を直接操作させた」"
            "という研究です。Aspen Plus はプラントの物質収支や熱収支を計算する定番の"
            "商用ソフトで、手作業での操作がかなり面倒なことでも知られています。"
            "今日は深い理論の話ではなく、「言語モデルに道具を持たせる」という工作寄りの"
            "話なので、気楽に聞いてください。キーワードは MCP、Model Context Protocol "
            "です。聞き慣れない人が多いと思うので、そこから順に説明します。"
        ),
    },
    # ---- 2. 潮流マップ -----------------------------------------------
    {
        "kind": "table",
        "headline": "プロセスシステム工学（PSE, Process Systems Engineering）でも"
        " LLM（大規模言語モデル）研究が主要トピックに",
        "lead": "PSE ASIA 2026 の抄録集（全375ページ）から LLM 関連の発表を数えると 14 件",
        "header": ["分類", "件数", "例"],
        "rows": [
            ["プロセス設計・シミュレーションの自動化", "5",
             "今日の P-77、蒸留の自動化（同じ台湾科技大）など"],
            ["物理モデル構築の自動化", "3",
             "加藤さんの基調講演（AutoPMoB）、バイオ培養、燃料電池"],
            ["数理最適化の補助", "1", "LLM が探索のやり方を設計（熱交換の組合せ問題）"],
            ["文献からの知識・データ抽出", "2", "経済性・環境影響の評価に使うデータ集め"],
            ["機械学習モデリングの補助", "2", "坂下さんの LLM 特徴量選択など"],
            ["複数 LLM の議論による設計評価", "1", "経済性担当と環境担当の AI が討論して設計を絞る"],
        ],
        "note": "14件のうち2件はうちの研究室（基調講演と特徴量選択）",
        "source": "出典：PSE ASIA 2026 抄録集より発表者が分類・集計",
        "notes": (
            "まず学会全体の空気感からです。抄録集を通して数えると、大規模言語モデルを"
            "使った発表が14件ありました。分類は私が付けたものなので目安ですが、"
            "傾向ははっきりしていて、どれも「LLM に人間の作業を代行させる」研究です。"
            "シミュレータの操作、モデル構築、最適化の設計、文献からのデータ集めまで、"
            "対象が違うだけで発想は共通です。口頭セッションの1つは丸ごとこの話題でした。"
            "ちなみに14件のうち2件はうちの研究室です。加藤さんの基調講演と、坂下さんの"
            "特徴量選択ですね。つまりうちも既にこの潮流の中にいるわけですが、今日は"
            "その中から、道具の作り方が一番具体的だった台湾の1件を紹介します。"
        ),
    },
    # ---- 3. MCP とは（図あり） ---------------------------------------
    {
        "kind": "mcp_fig",
        "headline": "MCP（Model Context Protocol）＝ LLM（大規模言語モデル）に"
        "「手」を付ける共通規格",
        "bullets": [
            "LLM を賢くする技術ではない。外の道具と繋ぐ「差し込み口」の共通規格（2024年公開）",
            "道具側が「できる操作の一覧」を公開し、LLM がそこから選んで呼び出す",
            "対応する AI アプリなら同じ道具がそのまま使える（USB-C のイメージ）",
        ],
        "fig": {
            "llm": "LLM\n（Claude・ChatGPT など）",
            "mcp": "MCP\n共通の差し込み口",
            "tools_title": "道具側",
            "tools": "ファイルの読み書き\nデータベース検索\nソフトの操作",
        },
        "notes": (
            "本題の前に MCP の説明です。Model Context Protocol、2024年に公開された"
            "共通規格です。よくある誤解ですが、これは言語モデルを賢くする技術では"
            "ありません。モデルの外にある道具と繋ぐための「差し込み口」の規格です。"
            "道具側は「自分にはこういう操作ができます」という一覧を公開し、LLM は"
            "会話の流れに応じてそこから操作を選んで呼び出します。規格が共通なので、"
            "道具を1回 MCP 対応にすれば、対応するどの AI アプリからも使えます。"
            "USB-C を思い浮かべてもらうのが早いです。実は私たちが使っている "
            "Claude Code の裏でも毎日この仕組みが動いています。今日の研究は、"
            "この差し込み口に Aspen Plus を繋いだ、という話です。"
        ),
    },
    # ---- 4. P-77 全体像（図あり） ------------------------------------
    {
        "kind": "arch_fig",
        "headline": "P-77 の全体像：LLM（大規模言語モデル）＋ MCP（Model Context "
        "Protocol）で Aspen Plus の操作を自動化",
        "boxes": ["指示文\n（自然言語）", "LLM", "MCP ツール群\n（60個超）",
                  "COM", "Aspen Plus"],
        "arrow_back": "計算結果・エラーを LLM に戻して次の操作へ",
        "bullets": [
            "Aspen Plus＝化学プロセスシミュレータ。COM（Windows のソフト同士を繋ぐ仕組み、"
            "Component Object Model）経由で外部から操作する",
            "組み立て→設定→計算→確認→修正を、6 段階の決まった手順で進める"
            "（各段階の詳しい名称は抄録に記載なし）",
            "動作確認の範囲：混合・加熱・気液分離・蒸留・反応器"
            "（CSTR、連続槽型反応器：Continuous Stirred-Tank Reactor）",
        ],
        "notes": (
            "P-77 の全体像です。左から、人間が日本語や英語で指示を書く、LLM が受け取る、"
            "そして MCP のツール群を通じて Aspen Plus を操作する、という流れです。"
            "Aspen Plus 側の窓口は COM という Windows の古い仕組みで、外部プログラムから"
            "設定値を読み書きできます。ポイントは、計算結果やエラーが LLM に戻り、"
            "それを見て次の操作を決める、というループになっていることです。"
            "作業は6段階の決まった手順で進みます。段階の細かい名前は抄録に書かれて"
            "いないので、ここでは構成だけ紹介します。動作確認は、混合・加熱・気液分離・"
            "蒸留・反応器と、単位操作を一通りカバーしています。数値の成功率のような"
            "評価は抄録にはなく、あくまで「一通り動いた」という段階です。"
        ),
    },
    # ---- 5. 設計教訓①（2カラム対比） ---------------------------------
    {
        "kind": "two_col",
        "headline": "設計の教訓①：万能ツール1個より、細かく分けた検証済みツール 60 個超",
        "left_title": "ありがちな作り方",
        "left_items": [
            "「Aspen を操作するコードを書いて」と LLM（大規模言語モデル）に丸投げ",
            "どこを触るかが予測できない",
            "失敗してもどこで間違えたか追いにくい",
        ],
        "right_title": "P-77 の作り方",
        "right_items": [
            "1 機能＝1 ツールに分割（60 個超）",
            "操作先（設定項目の場所）を事前に検証済み",
            "パラメータに範囲チェック（おかしな値を弾く）",
            "危険な操作を止める安全ガード＋プロセス工学の知識ベース",
        ],
        "bottom": "細かく割るほど、LLM の誤操作を「型」で防げる",
        "notes": (
            "ここからが今日いちばん持ち帰ってほしい設計の話です。LLM に道具を持たせる"
            "とき、ありがちなのは「操作コードを書いて」と丸投げする作り方です。"
            "これは自由度が高い反面、どこを触るか予測できず、失敗の原因も追いにくい。"
            "P-77 は逆で、1つの機能を1つのツールに割って、60個以上並べています。"
            "しかも各ツールは、操作先の設定項目の場所を事前に検証してあり、"
            "パラメータには範囲チェックが付き、危険な操作は安全ガードで止まります。"
            "つまり LLM の自由を意図的に狭めて、間違えようがない形にしてから任せる。"
            "地味ですが、これが「一通り動く」を実現している本体だと思います。"
            "私たちが Claude Code 用の道具を作るときも、そのまま使える考え方です。"
        ),
    },
    # ---- 6. 設計教訓②（×○ルート図） ---------------------------------
    {
        "kind": "route_fig",
        "headline": "設計の教訓②：正攻法の窓口がなければ、保存ファイルを直接書き換える",
        "lane1": {"from": "LLM", "via": "COM 経由で反応セットを登録",
                  "to": "COM にはその窓口がない", "ok": False},
        "lane2": {"from": "LLM", "via": "backup ファイルを直接編集",
                  "to": "Aspen Plus に読み込ませる", "ok": True},
        "note_lines": [
            "反応セット＝反応式や速度定数の登録情報",
            "backup ファイル＝Aspen Plus のテキスト形式の保存ファイル（単位換算もツール側で処理）",
            "LLM＝大規模言語モデル、COM＝Component Object Model（Windows のソフト同士を繋ぐ仕組み）",
        ],
        "bottom": "古いソフトの自動化ではよくある「現実解」（公式の使い方ではない点は注意）",
        "notes": (
            "個人的にいちばん好きなのがこのスライドです。反応器を扱うには反応式や"
            "速度定数、いわゆる反応セットを登録する必要がありますが、なんと COM には"
            "それを作る窓口がありません。正攻法が存在しない。そこで彼らがどうしたかと"
            "いうと、Aspen Plus の backup ファイル、これはテキスト形式の保存ファイル"
            "なんですが、それを LLM 側のツールで直接書き換えて、読み込ませました。"
            "単位の換算までツール側で面倒を見ています。行儀が良いかというと微妙ですが、"
            "古いソフトの自動化では、こういうファイル直接編集が最後の手段として"
            "よく登場します。公式の使い方ではないのでリスクは自己責任ですが、"
            "「窓口がないから諦める」ではなく穴を塞ぎに行く姿勢は、実務的で好きです。"
        ),
    },
    # ---- 7. うちへの示唆 ---------------------------------------------
    {
        "kind": "bullets_box",
        "headline": "うちへの示唆：研究室の道具も MCP（Model Context Protocol）で"
        " LLM（大規模言語モデル）に繋げられる",
        "bullets": [
            ("候補①：", "物理モデル構築の自動化（AutoPMoB）— 文献の取得から式の抽出までを指示文で"),
            ("候補②：", "自作の数値計算・シミュレーションコード — 条件を変えて実行し、結果の要約まで任せる"),
            ("候補③：", "実験・計算データの整理スクリプト — 「この図を作って」で済むように"),
        ],
        "box": "議論：どれを MCP 化したら、手間に対して一番効果がありそう？",
        "notes": (
            "最後に、うちの研究室に引きつけて考えてみます。MCP は共通規格なので、"
            "P-77 と同じことは私たちの道具でもできます。候補を3つ挙げました。"
            "1つ目は AutoPMoB のパイプラインです。文献の取得から式の抽出までを"
            "指示文で動かせる形にしておくと、実験の回し方が変わるかもしれません。"
            "2つ目は各自の数値計算コードです。条件を変えて実行して、結果の要約まで"
            "任せるところまで含めて道具化するイメージです。3つ目は日々のデータ整理で、"
            "これが一番手軽です。ここは私の思いつきなので、みなさんの道具で"
            "「これを繋いだら嬉しい」というものがあるか、後で意見を聞かせてください。"
        ),
    },
    # ---- 8. まとめ ----------------------------------------------------
    {
        "kind": "summary",
        "headline": "まとめ：持ち帰り 3 点",
        "items": [
            "プロセスシステム工学（PSE, Process Systems Engineering）でも "
            "LLM（大規模言語モデル）エージェント研究が主要トピック（今大会で14件）",
            "道具側の設計が本体：細かく割って検証・範囲チェック・安全ガード。"
            "窓口がなければ保存ファイルの直接編集という現実解",
            "MCP（Model Context Protocol）は共通規格。うちの道具にも今日から応用できる",
        ],
        "note": "※ 本紹介はポスター抄録のみに基づく（性能の数値評価は抄録に記載なし）",
        "notes": (
            "まとめます。1つ目、プロセスシステム工学の分野でも LLM に作業を代行させる"
            "研究が主要トピックになっていて、今大会だけで14件ありました。2つ目、"
            "その中で P-77 が具体的だったのは道具側の設計です。細かく割って、検証して、"
            "範囲チェックと安全ガードを付ける。窓口がなければファイルを直接書き換える。"
            "3つ目、MCP は共通規格なので、これは台湾の話ではなくて、うちの道具でも"
            "今日から試せる話です。なお、今日の内容はポスターの抄録だけを根拠にして"
            "いるので、性能がどれくらいかという数値の評価はできていません。そこは"
            "割り引いて聞いてください。以上です。質問やコメント、お願いします。"
        ),
    },
]


def slide_all_text(idx: int) -> str:
    """slide idx (0始まり) に載る全テキストを結合して返す（検証用）。"""
    s = SLIDES[idx]
    parts = [s.get("headline", ""), s.get("sub", ""), s.get("lead", ""),
             s.get("explain", ""), s.get("footer", ""), s.get("note", ""),
             s.get("source", ""), s.get("bottom", ""),
             s.get("left_title", ""), s.get("right_title", ""), s.get("box", "")]
    parts += s.get("body", []) + s.get("note_lines", []) + s.get("items", [])
    parts += s.get("boxes", []) + [s.get("arrow_back", "")]
    for b in s.get("bullets", []):
        parts.append("".join(b) if isinstance(b, tuple) else b)
    parts += s.get("left_items", []) + s.get("right_items", [])
    parts += s.get("header", [])
    for row in s.get("rows", []):
        parts += row
    fig = s.get("fig", {})
    parts += list(fig.values())
    for lane in (s.get("lane1"), s.get("lane2")):
        if lane:
            parts += [lane["from"], lane["via"], lane["to"]]
    return "\n".join(p for p in parts if p)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python3 -m pytest tests/test_p77_content.py -v`
Expected: 9 passed

（失敗したら本文を直す。よくある失敗＝台本の字数超過・略称の併記漏れ。**テスト側を緩めるのではなく本文側を直す**）

- [ ] **Step 5: コミット**

```bash
git add scripts/seminar_p77/content_p77.py tests/test_p77_content.py
git commit -m "feat(seminar): P-77スライドの本文・台本データと内容検証テストを追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: ビルダー（テキスト8枚＋ノート台本）と生成物テスト

**Files:**
- Create: `scripts/seminar_p77/build_p77_slides.py`
- Test: `tests/test_p77_slides.py`

**Interfaces:**
- Consumes: `content_p77.SLIDES` ほか Task 1 の全公開名
- Produces: `build(out_path: str) -> None`（テンプレ複製→全削除→8枚描画→保存）、描画ヘルパ `add_text`, `add_bullets`, `add_box`, `add_arrow`, `set_notes`（Task 3 の図描画もこのヘルパを使う）。色定数 `NAVY, TEXT, BLUE, GREEN, GRAY, LIGHT, LIGHT2, RED`、フォント定数 `BODY_FONT, HEAD_FONT`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_p77_slides.py` を以下の内容で作成:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python3 -m pytest tests/test_p77_slides.py -v`
Expected: collection ERROR（`ModuleNotFoundError: No module named 'build_p77_slides'`）

- [ ] **Step 3: ビルダーを実装（テキスト部分。図は Task 3）**

`scripts/seminar_p77/build_p77_slides.py` を以下の内容で作成:

```python
# -*- coding: utf-8 -*-
"""P-77 紹介スライドのビルダー。

テンプレ（PSE Asia 2026 発表スライド）を複製し、既存スライドを全削除して
content_p77.SLIDES の 8 枚を描画する。
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
DEFAULT_OUT = "/Users/kazuhiromiyamura/Desktop/ゼミ/論文紹介_P77_LLM_Aspen_かずひろ.pptx"

NAVY = RGBColor(0x16, 0x30, 0x4F)
TEXT = RGBColor(0x22, 0x22, 0x22)
BLUE = RGBColor(0x2E, 0x86, 0xC1)
GREEN = RGBColor(0x2E, 0x88, 0x51)
GRAY = RGBColor(0x8C, 0x8C, 0x8C)
LIGHT = RGBColor(0xEA, 0xF3, 0xFB)
LIGHT2 = RGBColor(0xDC, 0xEA, 0xF6)
RED = RGBColor(0xC0, 0x39, 0x2B)
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
    add_text(slide, 0.8, 2.25, 11.7, 0.7, c["sub"], size=24, color=BLUE,
             bold=True, font=HEAD_FONT)
    add_text(slide, 0.8, 3.2, 11.7, 2.3, c["body"], size=22, color=TEXT, spacing=1.3)
    add_text(slide, 0.8, 5.6, 11.7, 0.6, c["explain"], size=20, color=GRAY)
    add_text(slide, 0.8, 6.5, 11.7, 0.55, c["footer"], size=22, color=TEXT)


def render_table(slide, c):
    add_headline(slide, c["headline"])
    add_text(slide, 0.55, 1.35, 12.25, 0.55, c["lead"], size=22, color=TEXT, bold=True)
    n_rows = len(c["rows"]) + 1
    table_shape = slide.shapes.add_table(
        n_rows, 3, Inches(0.55), Inches(2.0), Inches(12.25), Inches(4.15))
    table = table_shape.table
    table.columns[0].width = Inches(4.9)
    table.columns[1].width = Inches(1.1)
    table.columns[2].width = Inches(6.25)
    for j, head in enumerate(c["header"]):
        _fill_cell(table.cell(0, j), head, bold=True, color=WHITE, fill=NAVY)
    for i, row in enumerate(c["rows"], start=1):
        fill = LIGHT if i % 2 == 1 else WHITE  # 既定の表スタイルを上書きして縞にする
        for j, val in enumerate(row):
            _fill_cell(table.cell(i, j), val, fill=fill,
                       align=PP_ALIGN.CENTER if j == 1 else PP_ALIGN.LEFT)
    add_text(slide, 0.55, 6.30, 12.25, 0.5, c["note"], size=20, color=BLUE, bold=True)
    add_text(slide, 0.55, 6.85, 12.25, 0.45, c["source"], size=20, color=GRAY)


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
    # 図は Task 3 で追加


def render_arch_fig(slide, c):
    add_headline(slide, c["headline"])
    add_bullets(slide, 0.55, 4.55, 12.25, 2.5, c["bullets"], size=20)
    # 図は Task 3 で追加


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


def render_route_fig(slide, c):
    add_headline(slide, c["headline"])
    add_text(slide, 0.55, 5.30, 12.25, 1.1,
             ["・" + line for line in c["note_lines"]], size=20, color=TEXT)
    add_box(slide, 0.55, 6.45, 12.25, 0.6, c["bottom"], NAVY, WHITE,
            size=20, bold=True, font=HEAD_FONT)
    # レーン図は Task 3 で追加


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
    "route_fig": render_route_fig,
    "bullets_box": render_bullets_box,
    "summary": render_summary,
}


def build(out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(TEMPLATE, out)          # マスター・テーマを丸ごと引き継ぐ
    prs = Presentation(str(out))
    sld_list = prs.slides._sldIdLst      # 既存14枚を全削除
    for sld in list(sld_list):
        sld_list.remove(sld)
    for c in C.SLIDES:
        slide = prs.slides.add_slide(prs.slide_layouts[0])  # 'DEFAULT'
        RENDERERS[c["kind"]](slide, c)
        set_notes(slide, c["notes"])
    prs.save(str(out))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    build(target)
    print(f"wrote {target}")
```

- [ ] **Step 4: テストが通ることを確認**

Run: `python3 -m pytest tests/test_p77_slides.py tests/test_p77_content.py -v`
Expected: 16 passed（生成物テスト7＋内容テスト9）

よくある失敗と対処:
- `size is None`: run を作らず `tf.text = ...` で書いた箇所がある → 必ず `add_run()`＋`_style()` を通す
- 表セルで失敗: `_fill_cell` を通さず `cell.text = ...` した箇所がある

- [ ] **Step 5: コミット**

```bash
git add scripts/seminar_p77/build_p77_slides.py tests/test_p77_slides.py
git commit -m "feat(seminar): P-77スライドビルダー（テキスト8枚+ノート台本）を追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 自作図3点（MCP 概念図・P-77 構成図・×○回避ルート図）

**Files:**
- Modify: `scripts/seminar_p77/build_p77_slides.py`（`render_mcp_fig` / `render_arch_fig` / `render_route_fig` の3関数を差し替え）
- Test: `tests/test_p77_slides.py`（図の存在テストを追記）

**Interfaces:**
- Consumes: Task 2 のヘルパ `add_box`, `add_arrow`, `add_text`、色定数、`content_p77` の `fig` / `boxes` / `lane1` / `lane2` キー
- Produces: 変更なし（`build()` の署名は不変）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_p77_slides.py` の末尾に追記:

```python
from pptx.enum.shapes import MSO_SHAPE_TYPE


def count_autoshapes(slide):
    return sum(1 for sh in slide.shapes
               if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE)


def test_figures_present(prs):
    slides = list(prs.slides)
    # S3: 箱3+矢印2 ≥ 5 / S4: 箱5+矢印4+戻り矢印1 ≥ 9（他要素は加算されるだけ）
    # S6: 箱・矢印・結果箱×2レーン+下部バー ≥ 7
    assert count_autoshapes(slides[2]) >= 5, "S3 の MCP 概念図がない"
    assert count_autoshapes(slides[3]) >= 9, "S4 の構成図がない"
    assert count_autoshapes(slides[5]) >= 7, "S6 の回避ルート図がない"


def test_figure_labels_rendered(prs):
    slides = list(prs.slides)
    assert "共通の差し込み口" in slide_text(slides[2])
    assert "Aspen Plus" in slide_text(slides[3])
    assert "窓口がない" in slide_text(slides[5])
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `python3 -m pytest tests/test_p77_slides.py -v -k figure`
Expected: 2 failed（`test_figures_present` / `test_figure_labels_rendered`）

- [ ] **Step 3: 3つのレンダラを図つきの完全版に差し替え**

`build_p77_slides.py` の `render_mcp_fig` / `render_arch_fig` / `render_route_fig` を以下で**置き換える**:

```python
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
    xs = [(0.55, 2.1), (3.0, 1.75), (5.1, 2.55), (8.0, 1.65), (10.0, 2.75)]
    fills = [LIGHT2, NAVY, BLUE, LIGHT2, GREEN]
    colors = [TEXT, WHITE, WHITE, TEXT, WHITE]
    for (x, w), fill, col, label in zip(xs, fills, colors, c["boxes"]):
        add_box(slide, x, 1.85, w, 1.35, label, fill, col, size=20, bold=True)
    for gap_x in (2.68, 4.78, 7.68, 9.68):
        add_arrow(slide, gap_x, 2.32, 0.3, 0.42)
    add_arrow(slide, 3.0, 3.5, 8.5, 0.5, left=True)
    add_text(slide, 3.0, 4.02, 8.5, 0.45, c["arrow_back"], size=20, color=GRAY,
             align=PP_ALIGN.CENTER)
    add_bullets(slide, 0.55, 4.55, 12.25, 2.5, c["bullets"], size=20)


def render_route_fig(slide, c):
    add_headline(slide, c["headline"])
    for lane, y in ((c["lane1"], 1.6), (c["lane2"], 3.35)):
        ok = lane["ok"]
        add_box(slide, 0.7, y, 1.7, 1.15, lane["from"], NAVY, WHITE, size=22,
                bold=True, font=HEAD_FONT)
        add_arrow(slide, 2.55, y + 0.35, 3.9, 0.45, label=lane["via"],
                  fill=GREEN if ok else GRAY)
        mark = "○ " if ok else "✗ "
        add_box(slide, 6.7, y, 5.9, 1.15, mark + lane["to"],
                LIGHT if ok else WHITE, GREEN if ok else RED, size=22,
                bold=True, outline=GREEN if ok else RED)
    add_text(slide, 0.55, 5.30, 12.25, 1.1,
             ["・" + line for line in c["note_lines"]], size=20, color=TEXT)
    add_box(slide, 0.55, 6.45, 12.25, 0.6, c["bottom"], NAVY, WHITE,
            size=20, bold=True, font=HEAD_FONT)
```

- [ ] **Step 4: 全テストが通ることを確認**

Run: `python3 -m pytest tests/test_p77_content.py tests/test_p77_slides.py -v`
Expected: 18 passed（矢印ラベル・図中文字も 20pt テストの対象に自動的に含まれる）

- [ ] **Step 5: コミット**

```bash
git add scripts/seminar_p77/build_p77_slides.py tests/test_p77_slides.py
git commit -m "feat(seminar): 自作図3点（MCP概念図・構成図・回避ルート図）を追加

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 最終生成・目視検証・開発記録への追記

**Files:**
- Modify: `docs/development_log.tex`（更新履歴 itemize の末尾、`\end{itemize}` の直前に1項目追記）
- 出力（コミットしない）: `/Users/kazuhiromiyamura/Desktop/ゼミ/論文紹介_P77_LLM_Aspen_かずひろ.pptx`

**Interfaces:**
- Consumes: `build_p77_slides.py` の `__main__`（既定出力パスで実行）

- [ ] **Step 1: 全テストを最終確認**

Run: `python3 -m pytest tests/test_p77_content.py tests/test_p77_slides.py -v`
Expected: 18 passed

- [ ] **Step 2: 本番パスへ生成して開く**

```bash
python3 scripts/seminar_p77/build_p77_slides.py
open "/Users/kazuhiromiyamura/Desktop/ゼミ/論文紹介_P77_LLM_Aspen_かずひろ.pptx"
```

Expected: `wrote /Users/kazuhiromiyamura/Desktop/ゼミ/論文紹介_P77_LLM_Aspen_かずひろ.pptx` と表示され、PowerPoint（または Keynote）が開く。

- [ ] **Step 3: 目視チェック（仕様書の受け入れ基準のうち機械検証できない項目）**

- 8枚すべてで文字のはみ出し・図の重なりがないこと（特に S2 の表と S4 の構成図）
- テンプレ由来の配色（紺・青系）に馴染んでいること
- ノート表示モードで台本が読めること

崩れがあれば `build_p77_slides.py` の座標を修正 → `python3 -m pytest tests/ -k p77` が通ることを再確認 → 再生成。

- [ ] **Step 4: フォント埋め込み（手動・PowerPoint がある場合のみ）**

PowerPoint for Mac: メニュー「PowerPoint → 環境設定 → 保存 → ファイルにフォントを埋め込む」にチェック → 開いている deck を上書き保存。（発表を自分の Mac で行うなら省略可。IBM Plex はインストール済み）

- [ ] **Step 5: 開発記録に追記**

`docs/development_log.tex` の更新履歴 itemize 末尾（最後の `\item` の後、`\end{itemize}` の前）に以下を追記:

```latex
  \item 2026-07-21（P-77 紹介スライド）：PSE ASIA 2026 のポスター P-77（台湾科技大、LLM＋MCP による Aspen Plus 操作自動化）を研究室ゼミで紹介するスライド 8 枚を作成。仕様 \texttt{docs/superpowers/specs/2026-07-21-p77-seminar-slides-design.md}・計画 \texttt{docs/superpowers/plans/2026-07-21-p77-seminar-slides.md} に基づき、本文・台本を \texttt{scripts/seminar\_p77/content\_p77.py}、描画を \texttt{scripts/seminar\_p77/build\_p77\_slides.py} に実装（PSE Asia 発表テンプレのマスターを流用、自作図 3 点、ノート欄に台本約 12 分）。pytest（\texttt{tests/test\_p77\_content.py}・\texttt{tests/test\_p77\_slides.py}）で 8 枚構成・全テキスト 20pt 以上・略称の正式名称併記・禁止語（格上げ表現含む）・台本分量を機械検証。出力 \texttt{\textasciitilde/Desktop/ゼミ/論文紹介\_P77\_LLM\_Aspen\_かずひろ.pptx} はリポジトリ外のためコミット対象外。
```

- [ ] **Step 6: コミット**

```bash
git add docs/development_log.tex
git commit -m "docs(devlog): P-77紹介スライド作成を開発記録に追記

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 仕様書の受け入れ基準との対応表

| 仕様書チェック項目 | 検証手段 |
|---|---|
| 8枚構成・各スライドにリード文 | `test_slide_count` + `test_slide_texts_match_content_module` |
| 略称のスライド内正式名称併記 | `test_abbreviations_have_full_names_on_each_slide` |
| 難解用語の言い換え/1行説明 | `test_no_forbidden_words` + `test_required_explanation_pairs` |
| 全テキスト 20pt 以上 | `test_all_runs_at_least_20pt_and_explicit` |
| 格上げ表現なし | `test_no_forbidden_words`（禁止語リストに含む） |
| 研究室2件への言及（S2） | S2 の `note` 行（`test_slide_texts_match_content_module` で描画確認） |
| 台本 約12分 | `test_notes_length_per_slide` + `test_notes_total_length` |
| テンプレ配色・フォント維持 | `test_canvas_is_16_9` + `test_fonts_are_ibm_plex`（＋Task 4 目視） |
| フォント埋め込み | Task 4 Step 4（手動） |
| 開いて崩れがない | Task 4 Step 3（目視） |
| 抄録ベースの明記 | `test_summary_slide_declares_abstract_only_basis` |
