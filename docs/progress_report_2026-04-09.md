# 進捗報告：データ拡大による数式Retrieval性能の改善

**報告日**: 2026-04-09
**プロジェクト**: AutoPMob — 科学文献からの物理モデル自動構築

---

## 1. 背景と目的

科学文献PDFから数式を抽出し、クエリに対して最適な物理モデル（数式）を自動選択するRetrieval システムを開発中。従来のTF-IDF+Jaccard ベースラインに対し、学習型リランカー（MLP）およびグラフ構造特徴量の有効性を検証している。

前回報告時点（3/30）では、グラフ特徴量が性能を**悪化**させるという問題があった。本報告では、その原因分析と対策（データ拡大）の結果を報告する。

---

## 2. データ拡大の内容

| | 拡大前 | Round 1 | Round 2（最新） |
|--|--------|---------|----------------|
| 数式数 | 975 | 1,613 | **3,244** |
| ソース数 | 22 | 52 | **71** |
| 変数数 | 1,099 | 2,098 | **3,704** |
| ドメイン数 | 157 | — | **31ハンドブックドメイン + PDF 40ソース** |

**拡大手法：**
- **ハンドブック式の生成**（LLM知識から31ドメイン、650式）：流体力学、伝熱、物質移動、熱力学、反応工学、電気化学、バイオプロセス、環境工学、原子力、半導体、音響学、光学、航空力学 等
- **新論文の取得と抽出**（Semantic Scholar API → 17本のopen-access PDF、308式）
- **大型テキストブック3冊の分割抽出**（pypdfで80ページずつ分割、39チャンク、1,311式）

---

## 3. 技術的手法の詳細

### 3.1 数式抽出パイプライン

**使用API**: Google Gemini 2.5 Pro（Structured Output モード）

PDFファイルをGemini APIにアップロードし、以下のプロンプトで数式を構造化JSONとして抽出する：

```
Extract all numbered equations and equations found in the appendices from
this textbook in LaTeX format. For each equation, identify the
"variable definitions" and the "physical meaning (context)" from the
surrounding text. Output the result in the JSON format specified below.

Constraints:
- Extract the content EXACTLY as it appears in the literature.
  Do not make unauthorized corrections or modifications.
- Preserve every variable symbol and subscript exactly as printed:
  e.g. if the source writes c_w, do NOT change it to c_p.

Output a single JSON array. Each element must have these fields:
- source_id, eq_id, equation (LaTeX), variables (記号→定義の辞書),
  context_text (物理的意味), domain (プロセス・ドメイン名)
```

APIの出力は `response_json_schema` パラメータでPydanticスキーマを指定し、型安全なJSON出力を強制している。レート制限は5 RPM（12秒間隔）。

### 3.2 ハンドブック式生成（LLM知識ベース）

論文PDFがない基礎的な工学方程式を、LLMの知識から直接生成した。31ドメイン（Round 1: 15ドメイン、Round 2: 16ドメイン）について、各ドメインのトピックリストを指定してプロンプトを構築：

```
You are an expert in {domain_label} and mathematical modeling.
Generate exactly {n} fundamental equations used in {domain_label},
in LaTeX format.

Cover these topics:
{topics}  # 例: "Navier-Stokes equations, Bernoulli equation, ..."

For each equation provide:
- source_id, eq_id, equation (LaTeX), variables (全記号の定義),
  context_text (物理的意味1-3文), domain

IMPORTANT:
- Each equation MUST be distinct (no duplicates).
- Include both the differential form and algebraic form where applicable.
- Variables dict must include ALL symbols that appear in the equation.
```

**対象ドメイン例（全31ドメイン）**：流体力学、伝熱、物質移動、熱力学、反応工学、プロセス制御、電気化学、バイオプロセス、環境工学、振動学、電気工学、高分子工学、燃焼工学、分離プロセス、輸送現象、原子力工学、半導体物理、音響学、光学、地盤工学、航空力学、HVAC・冷凍、食品工学、製薬工学、構造力学 等

各ドメインあたり20〜25式を生成（合計約650式）。

### 3.3 論文PDFの自動取得と抽出

**Semantic Scholar API** を使用し、15カテゴリのクエリで関連論文を検索：
- 例: `"fluid dynamics pipe flow mathematical model equations"`, `"electrochemical cell battery model equations"` 等
- Open-access PDFのみをダウンロード（各クエリ20〜30件を取得し、PDFリンクがあるもののみ）
- ダウンロードしたPDF（17本）を3.1の抽出パイプラインで処理 → 308式

### 3.4 大型テキストブックの分割抽出

Gemini APIのページ上限（約1,000ページ）を超えるテキストブック（3冊、最大1,400ページ）に対し、**pypdf** で80ページ単位のチャンクに分割：

- 3冊 → 39チャンクに分割
- 各チャンクを個別にGemini APIにアップロード・抽出
- `processed_chunks.json` で処理済みチャンクを追跡（中断時に再開可能）
- 合計1,311式を抽出

### 3.5 訓練ケース生成

#### コアケース生成（LLM）

統一数式カタログ（3,244式）をドメイン別にバッチ分割し（各バッチ≤300式）、Gemini 2.5 Proで物理的に意味のある訓練ケースを生成：

```
You are helping to create training data for a graph neural network that
selects relevant equations for physical process modeling.

CATALOG ({N} equations, domains: {domain_list}):
{catalog_json}   # model_id, domain, variable_symbols のリスト

TASK:
Using ONLY the models listed above, create approximately {n_cases}
realistic and physically meaningful core training cases.

For each core case:
- Pick one or more model_ids that together represent a coherent physical model.
- Define a context string describing the physical scenario in natural language.
- Choose input_variables and output_variables from variable_symbols.
- correct_model_ids: list of model_ids forming a correct mathematical model.

CONDITIONS FOR correct_model_ids:
(1) Given input_variables values and the equations, it MUST be possible
    to solve for all output_variables.
(2) The equation set MUST be sufficient and minimal.
```

11バッチ中9バッチが成功（2バッチはGemini 503エラーで失敗）、**123コアケース**を生成。

#### データ拡張（Python テンプレート）

1つのコアケースから9つのバリアントを自動生成：

| バリアント | 手法 | 目的 |
|-----------|------|------|
| `original` (×1) | そのまま | ベースケース |
| `context_paraphrased` (×3) | テンプレートで文脈を言い換え | 語彙の多様性 |
| `swap_io` (×1) | 入力↔出力を完全スワップ | 逆問題への対応 |
| `random_io_from_models` (×4) | 式の全変数からランダムにI/O割当 | 変数選択の多様性 |

言い換えテンプレート例：
- `"This case describes the following physical model: {context}"`
- `"In this scenario, the underlying process can be summarized as: {context}"`
- `"From the perspective of process modeling, this case focuses on: {context}"`

結果：123コア × 9バリアント = **1,107訓練ケース**

---

## 4. 評価手法

- **3方式の比較**（2段階 retrieval: TF-IDF候補抽出 → MLPリランキング）：
  - `baseline`：TF-IDF cosine × 0.7 + 変数Jaccard × 0.3（固定重み）
  - `reranker-7`：7次元特徴量の学習MLP（グラフ無し）
  - `reranker-10`：上記 + クエリ条件付きグラフ特徴量3次元（2-hop到達率、近傍重複、ブリッジ比率）
- **評価指標**：MRR（Mean Reciprocal Rank）、5 seed平均±標準偏差
- **データ分割**：source_id（論文）単位でtrain/val/test分割（データリーク防止）

---

## 5. 主要結果

### 全体MRR（5 seed平均±標準偏差）

| 式数 | baseline | reranker-7 | reranker-10 (graph) |
|------|----------|-----------|-------------------|
| 975 | 0.875 ± 0.088 | 0.885 ± 0.119 | 0.879 ± 0.123 |
| 1,613 | 0.900 ± 0.069 | 0.942 ± 0.052 | 0.948 ± 0.046 |
| **3,244** | **0.924 ± 0.040** | **0.937 ± 0.018** | **0.949 ± 0.031** |

### Original ケースのMRR（LLM合成でないケースのみ）

| 式数 | baseline | reranker-7 | reranker-10 (graph) |
|------|----------|-----------|-------------------|
| 975 | 0.839 | 0.860 | 0.842 |
| 1,613 | 0.885 | 0.936 | 0.935 |
| **3,244** | **0.921** | **0.942** | **0.952** |

### グラフ特徴量の効果（reranker-10 − reranker-7）

| 式数 | 差分 | 解釈 |
|------|------|------|
| 975 | **−0.006** | グラフが害になる |
| 1,613 | +0.006 | ほぼ同等 |
| 3,244 | **+0.012** | グラフが有効 |

---

## 6. 考察

1. **データ規模がグラフ構造の価値を決定する**：小規模（975式）ではグラフ特徴量がノイズとなり性能を悪化させたが、3,244式ではグラフが有効な追加信号を提供し、全手法中最良の性能を達成した。

2. **学習型リランカーの有効性**：固定重みベースラインから学習MLPへの変更だけで、975式で+1%、3,244式で+1.3〜2.5%の改善。統計的にも分散が大幅に縮小（0.088→0.040）。

3. **ベースラインの頑健性**：TF-IDF+Jaccardベースラインも式数増加で単調改善（0.875→0.924）。語彙的手法が強いが、グラフ構造との組み合わせで更に改善可能。

---

## 7. 今後の計画

| 優先度 | 項目 | 目的 |
|--------|------|------|
| 高 | 論文執筆 | データ規模×グラフ有効性のスケーリングを主テーマに |
| 中 | 訓練ケース追加（123→200+コア） | Gemini API不安定で一部生成できなかった分の補完 |
| 低 | GNN学習特徴量の再テスト | GCN-refined SVDを3,244式で再評価 |
