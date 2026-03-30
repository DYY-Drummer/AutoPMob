# AutoPMob — 科学文献からの数式抽出パイプライン

GNN（グラフニューラルネットワーク）を用いた物理モデル自動構築の研究のため、科学文献PDFから数式と変数定義を抽出し JSON で保存するパイプラインです。

## セットアップ

```bash
pip install -r requirements.txt
export GEMINI_API_KEY='your-api-key'   # または GOOGLE_API_KEY
```

## 使い方

```bash
# PDF から数式抽出（出力: extracted_equations.json）
python extract_equations.py papers/038.pdf

# 抽出と同時に変数表記を正規化（訓練用 *_normalized.json を生成）
python extract_equations.py papers/038.pdf --normalize

# 既存 JSON のみ正規化
python normalize_notation.py extracted_equations.json -o extracted_equations_normalized.json

# 数式グラフ用データセット統合（PDF一括 + Handbook 生成 → unified_equations.json）
python build_equation_graph.py --mode all    # handbook 生成 → input_pdfs/ の PDF 一括
python build_equation_graph.py --mode pdf    # PDF のみ（input_pdfs/ 内の .pdf）
python build_equation_graph.py --mode handbook  # 化学工学基礎式 20 個を LLM から生成のみ

# 論文メタデータ取得（Semantic Scholar → paper_dataset_links.json）
python fetch_papers.py
```

## 開発記録（ゼミ・論文用）

- **CLAUDE.md** … Claude Code 等で続きから開発するときの引き継ぎ（目的・データ・実験結論・スクリプト一覧）。
- **docs/development_log.tex** … 遭遇した問題・解決策・技術メモを LaTeX で記録。
- PDF 生成: `cd docs && pdflatex development_log.tex`

## 自動化

- **開発記録の追記**: 問題・解決策が出たときに、AI（Cursor）が `docs/development_log.tex` に追記するルールを `.cursor/rules/` に登録済みです。
- **GitHub 同期**: `git commit` のたびに自動で `git push` するフックを用意しています。**初回またはクローン後**に一度だけ実行してください:
  ```bash
  ./scripts/install-git-hooks.sh
  ```
  リモート（\texttt{origin}）を追加したうえでコミットすると、以降はコミット時に自動で push されます。

## GitHub との同期

リモート `origin` は **DYY-Drummer/AutoPMob** に設定済みです。

1. **GitHub でリポジトリを作成**（未作成の場合）  
   https://github.com/new で Repository name を `AutoPMob` にし、Create repository（README 等は追加しない）。
2. **初回プッシュ**（ターミナルで認証して実行）:
   ```bash
   git push -u origin main
   ```
3. 今後の変更はすべて GitHub へ同期する場合:
   ```bash
   git add .
   git commit -m "説明メッセージ"
   git push
   ```
   （API キーは `.env` に置き、`.gitignore` で除外済みです。コミットしないでください。）

## 構成

- `extract_equations.py` … 単一 PDF から数式抽出
- `build_equation_graph.py` … 統合パイプライン（`input_pdfs/` 一括、Handbook 生成、`unified_equations.json`）
- `normalize_notation.py` … 変数表記の正規化
- `config/notation_map.json` … 正規化ルール
- `input_pdfs/` … 一括処理する PDF を置くフォルダ
- `processed_pdfs.json` … 抽出済み PDF のファイル名一覧（ここに無い PDF だけ次回処理される）
- `docs/development_log.tex` … 開発記録（LaTeX）
