# docs

- **development_log.tex** … 開発記録（ゼミ・論文用）。問題と解決策、技術メモ、今後の予定を記載。

## ビルド

```bash
cd docs
pdflatex development_log.tex
# 必要に応じて 2 回実行（目次など）
pdflatex development_log.tex
```

生成物: `development_log.pdf`
