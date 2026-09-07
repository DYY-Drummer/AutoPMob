# W2 第 1 章 Introduction の本文化＋図 1 設計書

作成日: 2026-09-07 ／ 対象: `docs/論文化ギャップ分析_2026-08-30.md` の W2 ／ ステータス: 自律実行（本人不在。設計判断は本書で確定し、報告時に確認を仰ぐ）

## 1. 目的と成功基準

**目的**: 骨子 P1〜P8 のままの第 1 章を段落化し、AutoPMoB の全体像と本研究の位置づけを示す図 1 を新規作成して P2 の直後に置く。

**成功基準**:

1. 第 1 章が箇条書きを含まない本文（貢献 3 点の列挙のみ番号付き）になり、各段落の先頭文だけを拾うと章の要約になる（修論設計書の品質基準 2）。
2. 図 1 が英語ラベル・ベクタ（PDF）で生成され、(a) AutoPMoB の 4 段階の中で本研究の段階が強調され、(b) 本研究の入出力（説明文＋入出力変数 → 複数文献由来の数式 DB → 変数を共有し閉じた数式集合）が CSTR の 3 式で具体化されている。数式・共有変数の辺は厳密に対応する（加藤ルール B1・B3）。
3. 数値はすべて `experiments/*.json` から転記し、段落ごとに `% source:` コメントを付ける。A2 の結果（BM25・E5）を P6 に 1 句反映する。
4. 作文規則: 略語は初出で正式名称（CSTR・ODE・LLM・MLP・DoF）、"play an important role" などの常套句を具体に置換、「有意」は検定後のみ、能動態・冗語削除（English 11 原則）。
5. `latexmk -gg` exit 0、未定義参照なし、Overfull は既存の 1 件のみ。
6. devlog・ギャップ分析（W2 ✅）・memory を更新する。commit はしない。

## 2. 設計判断

| 項目 | 決定 | 理由 |
|---|---|---|
| 図 1 の構成 | 上下 2 段。(a) AutoPMoB pipeline: Documents → Extraction → Equivalence judgment → **Equation-set retrieval (this study)** → Physical model。(b) This study: Query（description・inputs・outputs）→ Equation database（11,146 equations, 361 documents; 無関係な式も混在）→ Retrieved closed set（CSTR の 3 式を文献別に色分け、共有変数 $r_A$・$k$ の辺、"3 equations, 3 unknowns: DoF = 0"） | 位置づけ（どの段階か）と課題の中身（集合として閉じること）を 1 図で自己完結させる。既存の PSE Asia 図は旧 GNN 手法用で流用不可 |
| CSTR の 3 式 | $\mathrm{d}C_A/\mathrm{d}t = (F/V)(C_{A0} - C_A) - r_A$、$r_A = k C_A$、$k = A\exp(-E/RT)$。Inputs $F, V, C_{A0}, A, E, R, T$、Output $C_A$、未知数 $C_A, r_A, k$ | 第 4 章 §4.3 の例と同じ 3 式。反応速度式とアレニウスの式を分ける（作文規則 E4）。$\tau$ を持ち込まない（孤立式回避 B1）。辺 = 共有変数: ODE–rate law は $r_A$（$C_A$ も共有）、rate law–Arrhenius は $k$ |
| 描画方法 | matplotlib（`generate_figures_thesis.py` に `overview()` を追加、`main()` にも登録）。mathtext で数式、DejaVu Sans、色は既存図と同じ系統（提案＝青 `#1a73e8`、灰 `#9aa0a6`、文献別に 3 色） | 既存図と同じ生成経路・体裁。TikZ は dvipdfmx 環境での検証コストが高い |
| 図の配置 | P2 の段落直後に `\begin{figure}[htb]`、幅 `0.98\linewidth`。キャプションは自己完結（4 段階と本研究の段階、(b) の読み方、DoF = 0 の意味） | 設計書「図 1 を 1 ページ目に」。圧縮設定解除（W5）後も章頭に近い位置に来る |
| 段落構成 | P1 問題（物理モデルの役割と構築の負担）／P2 AutoPMoB と要素技術（図 1）／P3 本研究の課題と「集合」でなければならない理由（CSTR）／P4 手法（2 段階・集合特徴・逐次選択・自己停止）／P5 データと分割／P6 結果（古典 IR・BM25/E5・LLM・機構・介入 2 回・自己停止）／P7 貢献 3 点（enumerate）／P8 章構成 | 骨子どおり。P6 に A1（第 1 段再設計 0.780）と A2（BM25/E5）を追加 |
| 第 1 章での数値 | 0.521→0.743（A）、0.466→0.689（B）、p = 0.00195、BM25/E5 最良混合 ≤ 0.602/0.555、LLM 0.23–0.30、分野一致 p = 0.98、学習版逐次 X≥8 で +0.056、閉包率 0.88/0.94 vs 0.78/0.87、第 1 段再設計 0.780 | 第 6 章と同一の一次データ。序論では丸めた値を使い、厳密値は第 6 章に委ねる |
| Abstract・第 7 章 | 触らない | W3 の範囲 |
| テスト | `tests/test_fig_overview.py`: `overview()` が PDF/PNG を生成し、図中の文字列に必須ラベル（"this study"、3 式、"DoF"）が含まれる。第 1 章は `latexmk` と、骨子マーカー（`\item \textbf{P`）が残っていないことの grep で確認 | 文章そのものは機械検証できないので、生成物の存在と自己完結に必要なラベルだけを検証する |

## 3. 図 1 の仕様（`overview()`）

- サイズ 7.2 × 4.6 inch、2 行（上 (a) 1.3、下 (b) 2.6 の高さ比）。
- (a) 5 つの角丸ボックスを矢印で連結。4 番目のみ塗り `#1a73e8`・白字・太枠、ラベル "Equation-set retrieval\n(this study)"。他は薄灰。ボックス下に 1 行の補足（例: Extraction: "equations, variables, data"; Equivalence judgment: "variables, equations"; Physical model: "closed system"）。
- (b) 左: Query カード（"Description: concentration dynamics of a CSTR / with a first-order reaction", "Inputs: $F, V, C_{A0}, A, E, R, T$", "Outputs: $C_A$"）。中: Equation database（重ねたカード 3 枚に "11,146 equations / 361 documents"、例として無関係式 1 つ "$Q = U A \Delta T$" を灰色で）。右: 3 式カード（Doc 1 青系・Doc 2 橙系・Doc 3 緑系の縁）を縦に並べ、ODE–rate law を "$r_A$" ラベル付き辺、rate law–Arrhenius を "$k$" ラベル付き辺で結ぶ。下に "3 equations, 3 unknowns ($C_A, r_A, k$): degrees of freedom 0 $\Rightarrow$ solvable"。
- 矢印: Query → database（"retrieve"）、database → set（"rank and select as a set"）。
- 図中に引用番号・スライド番号は書かない（B4/B5）。

## 4. スコープ外

- Abstract の書き直し、第 7 章の本文化（W3）。
- 手法パイプライン図・DAE 生成手順図（設計書 §6 の他の新規図）。
- 圧縮設定の復元（W5）。
