#!/bin/sh
# Git フックを .git/hooks にインストールする。
# クローン後や初回セットアップ時に実行: ./scripts/install-git-hooks.sh
set -e
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
cp scripts/post-commit .git/hooks/post-commit
chmod +x .git/hooks/post-commit
echo "Installed .git/hooks/post-commit (auto push on commit)."
