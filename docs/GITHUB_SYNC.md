# GitHub との同期手順

すべてのコード変動を GitHub に反映するための手順です。

## 1. リポジトリはすでに初期化済みです

- `git init` 済み
- 初回コミット済み（`main` ブランチ）

## 2. リモート設定（済）

- リモート \texttt{origin} は **https://github.com/DYY-Drummer/AutoPMob.git** に設定済みです。

## 3. GitHub にリポジトリを作成する（未作成の場合）

1. [GitHub](https://github.com) にログイン（ユーザ名: DYY-Drummer）
2. 右上の **+** → **New repository**
3. Repository name: `AutoPMob`
4. Public / Private を選択
5. **Create repository** をクリック（README や .gitignore は追加しない）

## 4. 初回プッシュ

ターミナルでプロジェクトのルート（AutoPMob）に移動し、認証できる状態で実行:

```bash
git push -u origin main
```

（HTTPS の場合は GitHub のユーザ名・パスワードまたは Personal Access Token が求められます。SSH を使う場合は `git remote set-url origin git@github.com:DYY-Drummer/AutoPMob.git` で URL を変更してから \texttt{git push}。）

## 5. 今後の変更をすべて GitHub に反映する

コードを変更したら、つど以下を実行するとすべて GitHub に同期されます。

```bash
git add .
git status          # 追加されるファイルを確認（.env や extracted_equations.json は .gitignore で除外）
git commit -m "変更内容の短い説明"
git push
```

- **API キー**は `.env` に保存し、`.gitignore` で除外しているためコミットされません。GitHub には絶対に上げないでください。

## 6. コミット時の自動 push

リポジトリには `scripts/post-commit` が含まれています。クローン後や初回に以下を実行すると、`git commit` のたびに自動で `git push` されます。

```bash
./scripts/install-git-hooks.sh
```

（`origin` を追加したうえでコミットしてください。）

## 7. ほかのマシンでクローンする場合

```bash
git clone https://github.com/DYY-Drummer/AutoPMob.git
cd AutoPMob
./scripts/install-git-hooks.sh   # 自動 push を有効にする
pip install -r requirements.txt
# GEMINI_API_KEY をその環境で設定
```
