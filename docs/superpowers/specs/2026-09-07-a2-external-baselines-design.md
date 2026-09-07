# A2 外部ベースライン（BM25・密ベクトル検索）設計書

作成日: 2026-09-07 ／ 対象: `docs/論文化ギャップ分析_2026-08-30.md` の A2 ／ ステータス: **実施済み（2026-09-07）**。結果は `docs/論文化ギャップ分析_2026-08-30.md` A2 と `docs/development_log.tex` を参照。設計判断は本人不在のため本書で確定し、報告時に確認を仰ぐ。実施中の変更: 混合重みは 3 点でなく 12 点掃引（min–max 正規化で「同じ重み」の意味が変わるため最良混合で比較）

## 1. 目的と成功基準

**目的**: 修論の外部比較は「古典 IR（TF-IDF 混合）」と「LLM 直接生成」のみで、査読者の「ベースラインが弱すぎないか（BM25 は？ 埋め込み検索は？）」に答えられない。BM25 と密ベクトル検索（E5）を**現行ベースラインと同じ評価枠**で測り、表 3 に行を足す。

**成功基準**:

1. 新スクリプトが、文章スコアを TF-IDF・正規化なし・重み (0.7, 0.3) にしたとき、`experiments/strat_A.json` / `strat_B.json` の baseline の per-seed 値（Recall@K_correct / MAP / Recall@20）を**完全に再現**する（評価枠が同一であることの回帰確認）。
2. BM25・E5 の単独（文章のみ）と混合（＋変数一致度）の Recall@K_correct / MAP / Recall@20 を、設定 A・B × 層化分割 10 seed で JSON に保存する（mean ± SD と per-seed）。
3. 各外部ベースラインと「古典 IR ベースライン」「提案手法（reranker-10S）」の対応あり検定（Wilcoxon 符号順位・対応あり t・Cohen's d_z）を出す。
4. 第 1 段の候補窓としての被覆率（k = 10〜400）を BM25・E5 で測り、A1 の最良 w30-70（k=50 で 0.967）と比べる。
5. 修論（§5.2 比較手法・§6.1 表 3・付録・文献）を更新し `latexmk` が exit 0 で通る。数値はすべて JSON から転記する。
6. `docs/development_log.tex` とギャップ分析（A2 の行）を更新する。

## 2. 設計判断（本人不在のため本書で確定。報告時に変更可）

| 項目 | 決定 | 理由 |
|---|---|---|
| 密ベクトルモデル | `intfloat/e5-base-v2`（110M パラメータ、Hugging Face から取得、safetensors 438 MB） | ギャップ分析が「E5 系」を例示。英語文書向けの標準的な文埋め込みモデル。sentence-transformers は未導入なので `transformers` で直接推論（平均プーリング＋L2 正規化、接頭辞 `query: ` / `passage: `） |
| BM25 実装 | 自前（scipy sparse・ベクトル化）。Okapi BM25、k1 = 1.5、b = 0.75、IDF は Lucene 形式 ln(1 + (N − n + 0.5)/(n + 0.5))、単語ユニグラム、トークナイズは TF-IDF と同じ規則（小文字化・`\b\w\w+\b`） | `rank_bm25` は未導入で pip 追加を避ける。既定値は `rank_bm25` と同じ k1・b。テストで純 Python 参照実装と一致を確認 |
| 混合の正規化 | BM25・E5 の文章スコアはクエリごとに min–max で [0, 1] に正規化してから混合。TF-IDF は現行どおり生のコサイン（回帰再現のため） | BM25 は非有界、E5 のコサインは [0.7, 0.95] 付近に圧縮されるため、生値のまま 0.7/0.3 で混ぜると変数項の実効重みが手法ごとに変わる。min–max は混合検索の標準手順 |
| 混合重み | 3 通り: 文章のみ (1, 0)、表 3 と同じ (0.7, 0.3)、A1 最良 (0.3, 0.7) | 「同じ土俵」(0.7, 0.3) を主結果に、A1 で古典 IR 自体が 0.521→0.600 に上がることが分かっているので (0.3, 0.7) も付け「最良重みでも提案手法に届くか」を見る |
| E5 のクエリ文 | 2 通り: (a) context のみ（現行ベースラインの文章項と同じ）、(b) context ＋ 入出力変数列（`case_text(c, io=True)`） | 密ベクトルなら変数記号も文章として与えられるという反論に先回りする |
| 評価対象 | 全 DB（11,146 式）の順位付け。学習なし | 現行ベースラインと同じ土俵 |
| 分割・seed | `set_aware_reranker.stratified_src_split` を同じ特徴量で呼ぶ。seed は正典 10 個。設定 A = original, multisource_*, dae_*（1,823 件）／設定 B = dae_*（1,000 件） | strat_A/B と同じ test ケース集合になり、per-seed の対応あり検定ができる |
| 指標 | `evaluate_multi_eq.compute_all_ranks / case_metrics / aggregate_metrics` を流用 | 修論の定義と厳密に同一 |
| 第 2 段（reranker-10S）を BM25/E5 の候補窓の上に載せる実験 | **今回はやらない**。被覆率だけ測り、E5 混合の被覆率が w30-70（0.967）を上回った場合のみ次の介入候補として報告する | ギャップ分析の最小構成（表 3 に行を足す）に合わせる。載せるには `set_aware_reranker.py` の第 1 段を差し替える改修が要り半日〜1 日を超える |
| 図 `fig_method_comparison` | 変更しない | 表 3 が担う。図を変えると本文・キャプションの連鎖修正が要る |
| commit | しない（作業ツリーに残す） | devlog 運用メモ「commit/push は本人が求めたときのみ」に従う |

## 3. 構成

```
external_baselines.py            # 新規: スコアラー・混合・評価・被覆率（CLI）
analyze_external_baselines.py    # 新規: strat_A/B との対応あり検定 + LaTeX 表本体の出力
run_external_baselines.sh        # 新規: 設定 A/B × 5 スコアラー × 3 重み を一括実行
tests/test_external_baselines.py # 新規: BM25 参照実装一致・正規化・混合・被覆率・回帰再現
experiments/external_baselines_A.json / _B.json      # 結果（mean±SD, per_seed）
experiments/external_baselines_coverage.json         # 被覆率（設定 A・B, k=10..400）
experiments/external_baselines_stats.json            # 検定
experiments/embeddings/                              # E5 埋め込みキャッシュ（.gitignore に追加）
```

### 3.1 `external_baselines.py`

- **データ**: `two_stage_query_conditioned.load_equations / load_cases / eq_key / eq_text / eq_vars / case_text / io_vars / jaccard / in_vars / out_vars / case_src` を流用。variant フィルタは `set_aware_reranker.main` と同じ規則（接尾辞 `_` は前方一致）。
- **スコアラー**（`--scorers`、既定 `tfidf:none,tfidf:minmax,bm25:minmax,e5:minmax,e5io:minmax`）:
  - `tfidf`: `TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1,2), min_df=1)` を全式で fit、クエリ = `case_text(c)`、コサイン類似度。現行と同一。
  - `bm25`: 上記の自前 BM25。索引 = `eq_text(e)`、クエリ = `case_text(c)`。
  - `e5` / `e5io`: `intfloat/e5-base-v2`。passage = `eq_text(e)`、query = `case_text(c)` / `case_text(c, io=True)`。コサイン。埋め込みは `experiments/embeddings/<model>__<kind>__<sha1(texts)[:12]>.npy` にキャッシュ。device は MPS → CPU の順に自動選択し JSON に記録。
  - 正規化 `none` / `minmax`（クエリごと、全 DB にわたり (s − min)/(max − min)、max = min なら 0）。
- **混合**: `score = w_text · S̃ + w_var · Jaccard(io_vars(c), eq_vars(e))`。`--weights 1:0,0.7:0.3,0.3:0.7`。
- **評価**: seed ごとに層化分割 → test ケース（正解式が DB に 1 件以上）→ 全 DB を降順に並べ `compute_all_ranks`（欠落なし）→ `case_metrics` → `aggregate_metrics`。結果ラベルは `"{scorer}__{norm}__w{100·w_text:02d}-{100·w_var:02d}"`（例 `bm25__minmax__w70-30`, `e5__minmax__w100-00`）。JSON は strat_A.json と同じ形（`results[label]` に各指標の `{mean, std}` と `per_seed`）。`--save-per-case` で per-case も保存（既定 off）。
- **被覆率**（`--coverage`）: 分割なしで設定内の全ケースについて `coverage(order, corr, k) = |corr ∩ top-k| / |corr|` を k ∈ {10, 25, 50, 100, 200, 400} で平均（`analyze_stage1_coverage.py` と同じ定義、mean と SEM）。
- **性能**: 文章スコアは全ケース × 全 DB の行列（ケース数 ≤ 1,823 × 11,146 ≈ 2,000 万 float32 = 81 MB）を seed ループの外で 1 回だけ計算し、seed ループでは行を引くだけにする。Jaccard も同様に 1 回だけ（sparse 二値行列の積で計算）。

### 3.2 `analyze_external_baselines.py`

- 入力: `experiments/strat_{A,B}.json`（baseline・reranker-10S の per_seed）と `experiments/external_baselines_{A,B}.json`。
- 各ラベルについて `analyze_significance.paired_stats` で **vs baseline** と **vs reranker-10S** を Recall@K_correct / MAP / Recall@20 で計算し `external_baselines_stats.json` に保存。
- 表 3 用に、`mean ± SD`（Recall@K_correct）、MAP、Recall@20 の行を LaTeX 形式で標準出力に印字する（転記ミス防止）。

### 3.3 テスト（pytest）

1. BM25: 3 文書のおもちゃコーパスで純 Python 参照実装（同じ式）とスコアが一致。
2. min–max: 定数行は全 0、一般行は max = 1・min = 0。
3. 混合: 重み (1, 0) は文章スコアそのもの、(0, 1) は Jaccard そのもの。
4. 被覆率: 正解 3 件中 2 件が上位 k に入れば 2/3。
5. 回帰再現: 設定 B・seed 42・`tfidf:none`・(0.7, 0.3) の Recall@K_correct / MAP / Recall@20 が `strat_B.json` の seed 42 と一致（許容 1e-6）。実データを読むので数秒〜十数秒。

### 3.4 修論の更新箇所

- `Experiment.tex` §5.2: 外部比較に「BM25\cite{BM25}」「密ベクトル検索（E5\cite{E5}）」を追加し、単独と混合（同じ重み・min–max）の定義、E5 のクエリ文、学習なしを 1 段落で述べる。
- `ResultsAndDiscussion.tex` §6.1: 表 3 に設定ごとに 2 行（BM25 混合・E5 混合、(0.7, 0.3)）を追加。文章のみ・(0.3, 0.7) の値と検定結果は本文の段落で述べる。`% source:` コメントを付ける。
- `Appendix.tex`: ハイパーパラメータ表に BM25（k1, b, IDF）と E5（モデル ID・プーリング・最大長・接頭辞・正規化）を追加。
- `main.tex`: `\bibitem{E5}`（Wang ら, arXiv:2212.03533 — 公開ページで書誌を確認してから記入）。初出順の整列を維持。
- 被覆率が w30-70 を上回った場合のみ §6.8 に 1〜2 文追加。
- 作文規則: 略語の初出正式名称（BM25 は固有名として扱い展開しない、E5 はモデル名）、主観語なし、「有意」は検定後のみ、能動態・冗語削除。

## 4. 判断規則（結果の読み方）

- BM25・E5 が提案手法を下回れば、表 3 に追加して「学習した第 2 段は、より強い文章スコアでも置き換えられない」と書ける。
- (0.3, 0.7) の混合で古典 IR（0.600）と同程度なら「文章スコアの強さより変数一致度の重みが効く」と §6.8 と整合させて書く。
- E5 単独が TF-IDF 単独を大きく上回るなら、密ベクトルを第 1 段に使う改修を次の介入候補として明記する（今回は実施しない）。

## 5. スコープ外

- BM25/E5 を第 1 段にした reranker-10S の再訓練。
- E5 以外の埋め込みモデルの比較（`--model` で差し替え可能な実装にはする）。
- 図 `fig_method_comparison` の更新。
