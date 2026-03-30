# AutoPMob — Claude Code 向けプロジェクト概要

このファイルは **Claude Code など別環境で開発を続ける**ための引き継ぎ用サマリです。詳細・数式・図表は `docs/development_log.tex`（LaTeX）が一次情報です。

---

## 1. 研究目的・方針

- **最終ゴール**: GNN を含む機械学習で **物理モデル（数式セット）の自動構築・選択** を行う研究の下流データと実験基盤を整えること。
- **当面のコアタスク**: **Equation Retrieval** — 与えられたケース（文脈 `context`、入出力変数 `input_variables` / `output_variables`）に対し、式ライブラリ全体から **正解となる式 ID（`correct_model_ids`）を上位にランキング**する。正解は **複数本あり得る**（multi-positive）。
- **評価指標**: **MRR**（各ケースで正解集合のうち最も上位に来た式の順位の逆数の平均）と **Recall@K**。単一ラベル accuracy は不採用。
- **分割の原則**: **source_id（論文・ハンドブック単位）で train/val/test を分ける**。同一 PDF 由来の式・文脈が train と test に混ざるとリークで過大評価になる。実装は `run_retrieval_experiments.py` 等で case の正解 ID から source を推定して分割。

---

## 2. 現時点の実験的結論（重要）

- **学習付き GNN/MLP 単体より、シンプルなハイブリッドが強い**: `run_retrieval_experiments.py` の比較では、**TF-IDF（文脈）+ 変数集合の Jaccard（I/O 一致）を混ぜた `baseline_mix`** が現状ベストだった（詳細は `experiments/retrieval_experiments.md`）。
- **素朴な「式＝ノード、変数共有＝エッジ」グラフ**はエッジが密になり、GNN の集約が retrieval に効きにくい — 開発記録に **全文分析**あり（`docs/development_log.tex` の「なぜ現状のグラフ構築ではGNNの強みを活かせないか」）。
- **対応済みの設計変更**: **式–変数二部グラフ**（式ノード + 変数ノード、エッジは「式がその変数を含む」のみ）。`equation_graph.pt` を再生成するには `python build_graph_data.py` が必要（古い `.pt` は形式が異なる可能性あり）。
- **残ターゲット**: hard negative、クエリ条件付きエッジ／サブグラフ、エッジ重み、正解共起エッジ、複数 seed 評価など — `development_log.tex` の「試したいこと・実験アイデア」参照。

---

## 3. データ構造（必ず読む）

### `unified_equations.json`（式ライブラリ）

- 1 要素 = 1 式。**グローバル式 ID** は必ず **`{source_id}__{eq_id}`**（例: `handbook_chemeng__eq_1`）。`eq_id` だけは文書ローカルなので単独利用禁止。
- 主要キー: `source_id`, `eq_id`, `equation`, `variables`（記号 → 説明）, `context_text`, `domain` など。

### `training_cases.json`（検索タスク）

- 主要キー: `case_id`, `variant_type`, `context`, `input_variables`, `output_variables`, `correct_model_ids`（上記グローバル式 ID のリスト）。
- `variant_type` 例: `original`, `swap_io`, `random_io_from_models`, `context_paraphrased`。

### グラフ `equation_graph.pt`（PyTorch Geometric `Data`）

- **二部グラフ版**: `x` は式ノード + 変数ノードの特徴（768 次元 CodeBERT 系）。`edge_index` は二部エッジ。`num_equations` で式ノード数。`edge_index_eq_eq` は式同士のみの隣接（後方互換・一部スクリプト用）。
- **式の埋め込みとして使うのは先頭 `num_equations` 行**。

---

## 4. 式表現の二系統

| 系統 | 内容 | 主な利用箇所 |
|------|------|----------------|
| **A: CodeBERT** | `equation_graph.pt` の `x`（式+文脈の [CLS]） | `train_gnn_retriever.py`, `q_mlp_to_x` 系 |
| **B: TF-IDF + SVD(256)** | 式テキストとクエリテキストを同じベクトル空間に | `svd_cos`, `baseline_mix`, `train_gnn_refine_svd.py` |

- **パターン1** (`train_gnn_retriever.py`): 式側を GCN/GAT で更新（二部グラフは全ノードに通し、出力の先頭 `num_equations` 行を式埋め込み）。クエリは SVD→MLP→768。**異空間のため学習が難しくベースラインを下回りやすい**という記述あり。
- **パターン2** (`train_gnn_refine_svd.py`): SVD 初期埋め込みに **残差 GCN**（`E = E0 + α·GCN(E0)`、`α` 学習可能・初期 0）でグラフ情報を足す。`run_retrieval_experiments.py` の `gnn_refine_svd_residual`。

---

## 5. 主要スクリプト一覧

| スクリプト | 役割 |
|------------|------|
| `extract_equations.py` | 単一 PDF から数式抽出（Gemini API） |
| `build_equation_graph.py` | PDF 一括・Handbook 生成・`unified_equations.json` 統合（`processed_pdfs.json` でスキップ管理） |
| `normalize_notation.py` | `config/notation_map.json` に基づく表記正規化 |
| `build_graph_data.py` | `unified_equations.json` → `equation_graph.pt`（式–変数二部グラフ） |
| `generate_training_cases.py` | LLM + 拡張で `training_cases.json` |
| `validate_dataset.py` | ID・変数整合・重複などの健全性チェック |
| `analyze_dataset.py` | 統計・図・`dataset_report.txt`（二部グラフ対応） |
| `add_context_paraphrases.py` | context 言い換えバリアント追加 |
| `run_retrieval_experiments.py` | 複数方式の同一分割比較・簡易グリッド探索 → `experiments/` |
| `train_gnn_retriever.py` | CodeBERT 空間 GNN retriever（`--use-gnn`, `--gnn-type gcn\|gat`） |
| `train_gnn_refine_svd.py` | SVD 空間残差 GCN |
| `fetch_papers.py` | Semantic Scholar メタデータ |
| `baseline_retrieval_eval.py` | ベースライン評価（実験ランナーからも利用） |

---

## 6. よく使うコマンド

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...   # または GOOGLE_API_KEY

# データ健全性
python validate_dataset.py

# グラフ再構築（JSON 更新後は必ず）
python build_graph_data.py

# データセットレポート・図
python analyze_dataset.py

# 方式比較（結果は experiments/retrieval_experiments.md）
python run_retrieval_experiments.py
```

README の PDF 一括・Handbook は `python build_equation_graph.py --mode all` 等（`README.md` 参照）。

---

## 7. 技術スタック

- Python 3、**PyTorch**、**PyTorch Geometric**、**transformers**（CodeBERT）、**scikit-learn**（TF-IDF/SVD）、**google-genai**（Gemini）、Pydantic、NetworkX 等（`requirements.txt`）。

---

## 8. リポジトリ運用（このプロジェクト固有）

- **変更のたび**に `docs/development_log.tex` の **更新履歴**へ 1 行追記（理由つき）。バグ対応時は「遭遇した問題と解決策」に `\subsection` 追加。
- **コミット後 GitHub へ push**（`.cursor/rules/development-log.mdc`）。`./scripts/install-git-hooks.sh` で post-commit push を入れられる（README 記載）。
- **API キーをコミットしない**（`.gitignore` で `.env` 等）。

---

## 9. ユーザーからの方針メモ

- 実装・実験について **「要求に縛られず大胆に試してよい」** という意向あり（新アイデアは開発記録の「試したいこと」と整合させつつ進めてよい）。

---

## 10. 次にやると良いこと（優先度は状況依存）

1. `build_graph_data.py` 実行後、`validate_dataset.py` / `analyze_dataset.py` / `run_retrieval_experiments.py` で **二部グラフ版の数値が一通り出る**ことを確認する。
2. **GAT**（`train_gnn_retriever.py --gnn-type gat`）や **hard negative** など、開発記録に列挙した実験を 1 本ずつ切る。
3. `research_context.md` は英日メモ用に空だった — 必要なら本ファイルへのリンクか短い研究メモを追記。

**詳細の一次ソース**: `docs/development_log.tex`（特に「データセット定義と評価」「GNNの仕組みと埋め込み」「試したいこと」）。
