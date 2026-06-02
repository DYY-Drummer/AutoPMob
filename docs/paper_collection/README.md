# 論文収集パッケージ

**目的**: AutoPMob プロジェクトのデータベース $\text{DB}_\text{eq}$ を **32 分野 × 10 報 ≈ 320 報** で拡張する。

---

## 📂 ファイル一覧

| ファイル | 用途 |
|---------|------|
| `論文選定マニュアル.md` | 選定基準・Tier 判定・専門家観点を体系化 |
| `論文検索キーワード_32分野.csv` | Google Scholar 検索用キーワード集 |
| `論文評価チェックシート.xlsx` | 論文を 1 件ずつ評価・記録 |
| `README.md` | 本ファイル |

---

## 🚀 使い方

### Step 1：マニュアル熟読
**`論文選定マニュアル.md`** を読み、評価軸 4 つと必須/推奨/排除項目を理解する。

### Step 2：検索キーワード取得
**`論文検索キーワード_32分野.csv`** から、自分の担当分野（または興味分野）の検索クエリをコピーし、Google Scholar に貼り付け。

### Step 3：論文評価
ヒットした論文 PDF を取得し、**`論文評価チェックシート.xlsx`** の「評価フォーム」シートで各項目を ✓ / ✗ で記入。
- 必須 6 項目すべて ✓ → 採用候補
- 排除 1 項目でも ✓ → 即除外
- 推奨項目の充足数で Tier 判定

### Step 4：採用論文の登録
「採用論文一覧」シートに採用論文を行として追加。ファイル名・分野・Tier・予想式数を記録。

### Step 5：PDF を所定の場所に保存
```
/Users/kazuhiromiyamura/Desktop/AutoPMob/pdfs/<domain>/<filename>.pdf
```

### Step 6：抽出パイプライン投入
```bash
cd /Users/kazuhiromiyamura/Desktop/AutoPMob
python extract_equations.py pdfs/<domain>/<filename>.pdf
python build_equation_graph.py --add pdfs/<domain>/<filename>.pdf
python validate_dataset.py  # 重複・健全性チェック
python build_graph_data.py  # 二部グラフ再構築
```

### Step 7：進捗確認
「進捗ダッシュボード」シートで分野ごとの収集状況を確認。目標 10 報/分野に到達するまで Step 2-6 を繰り返す。

---

## 📊 数値目標

| 項目 | 現状 | 目標 |
|------|---:|---:|
| 論文総数 | 39 報 | **320 報** |
| 数式総数（PDF 由来）| 2,574 式 | **8,000 〜 10,000 式** |
| 連立系を持つ論文比率 | 未測定 | 50% 以上 |
| 分野均等性 | 偏り大 | 各分野 8-13 報 |

---

## 💡 Tips

### 効率的な収集順序

優先度順：
1. **教科書**（"textbook" "fundamentals" "principles" を検索に追加）
2. **Review 論文**（"review" "survey" "tutorial"）
3. **応用論文**（具体的なプロセス名・装置名を含めて）

### 著名教科書の例（Tier 1 候補）

| 分野 | 教科書 |
|------|--------|
| 反応工学 | Fogler *Elements of Chemical Reaction Engineering* |
| プロセス制御 | Seborg *Process Dynamics and Control* |
| 輸送現象 | Bird-Stewart-Lightfoot *Transport Phenomena* |
| 熱力学 | Smith Van Ness Abbott |
| 伝熱 | Incropera *Heat and Mass Transfer* |
| 流体力学 | White *Fluid Mechanics* |
| 分離 | Seader *Separation Process Principles* |
| 電気化学 | Bard Faulkner *Electrochemical Methods* |
| バイオ | Shuler *Bioprocess Engineering* |
| 燃焼 | Turns *Introduction to Combustion* |

### 採用判断に迷ったとき

論文 PDF を Claude（私）に渡して評価依頼してください：
> 「[ファイル名] を選定基準で評価してください」

→ 必須/推奨/排除のチェックリストで自動評価し、Tier 判定と採用可否を返します。

### 仮説 B との関連

仮説 B（関連性の低い式の効用）の検証のため、**ターゲット分野以外の論文も意図的に含める**ことが重要。
- ただし必須項目はすべて満たすこと
- 「無関係でも数式品質は高い」論文を歓迎

---

## 📞 問い合わせ

論文選定で迷ったときは、Claude に評価を依頼してください。
