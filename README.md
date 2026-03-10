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
```

## 開発記録（ゼミ・論文用）

- **docs/development_log.tex** … 遭遇した問題・解決策・技術メモを LaTeX で記録。
- PDF 生成: `cd docs && pdflatex development_log.tex`

## GitHub との同期

1. リポジトリを初期化済みの場合、リモートを追加してプッシュ:
   ```bash
   git remote add origin https://github.com/<your-username>/AutoPMob.git
   git branch -M main
   git push -u origin main
   ```
2. 今後の変更はすべて GitHub へ同期する場合:
   ```bash
   git add .
   git commit -m "説明メッセージ"
   git push
   ```
   （API キーは `.env` に置き、`.gitignore` で除外済みです。コミットしないでください。）

## 構成

- `extract_equations.py` … 抽出メイン
- `normalize_notation.py` … 変数表記の正規化
- `config/notation_map.json` … 正規化ルール
- `docs/development_log.tex` … 開発記録（LaTeX）
