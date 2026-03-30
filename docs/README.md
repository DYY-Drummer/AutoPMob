# docs

- **development_log.tex** … 開発記録（ゼミ・論文用）。問題と解決策、技術メモ、今後の予定を記載。

## TeX 環境のインストール（macOS + Homebrew）

管理者パスワードが使える場合は軽量な **BasicTeX** でもよい:

```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
```

パスワードなしで入れる場合は **Homebrew の texlive 公式**（依存が多く容量大）:

```bash
brew install texlive
```

いずれも PATH に `/opt/homebrew/bin`（Apple Silicon）が通っていれば、`pdflatex` / `uplatex` / `dvipdfmx` が使えます。

## ビルド（日本語 jsarticle）

このプロジェクトの `development_log.tex` は **日本語**のため、`pdflatex` だけではそのままでは通りません。**upLaTeX + dvipdfmx** を使います。

```bash
cd docs
uplatex -interaction=nonstopmode development_log.tex
uplatex -interaction=nonstopmode development_log.tex
dvipdfmx development_log.dvi
```

生成物: `development_log.pdf`（中間ファイル `development_log.dvi`, `.aux`, `.log`, `.toc`, `.out` は `.gitignore` 推奨）。
