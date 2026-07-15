# 設計書：PFIによるケース別依存プロファイル

- 日付: 2026-07-15
- 状態: 承認済（実装計画へ）
- 対象: 設定A（1,823件）reranker-10S
- 関連: `analyze_pfi.py`（集約PFI・既存）, `analyze_feature_x_difficulty.py`（fingerprint図・既存）, 修論 §6.x 機構分析

## 1. 背景と目的

**問い**: 「特徴量はなぜ効くのか」。平均的な重要度（集約PFI）ではなく、**その特徴を壊すと正解→不正解に転じるのはどんなケースか**を同定・特徴づけることで機構を説明する。

**既存の到達点**: `analyze_pfi.py` は訓練済み reranker-10S に対し、各特徴（および群）を置換したときの Recall@K_correct の**平均低下**を 10 seed × 20置換で測る。群PFI・共有置換・冗長性指標まで実装済み。ただし出力は集約値（`experiments/pfi_results.json`）のみで、**per-case では何も残していない**。

**本設計のギャップ**: PFIを per-case に降ろし、(1) 置換で反転するケースを同定、(2) そのケース群を属性で特徴づける。分析深度は**記述プロファイル（dependent vs robust の属性対比）**に限定（発見モデル・用量反応の完全連結は非スコープ）。

## 2. 産出する主張の形

set-aware 3特徴について、置換で反転するケース群の属性プロファイルを示す。期待される絵姿（実測で検証する仮説であり、結果は正直に報告する）:

- **gComp / gCoh**（変数重なり群）: 分離度 `sep`（正解式で高く不正解式で低い度合い）が大きいケースで反転が集中。
- **gDom**（話題群）: 反転ほぼゼロ。§6.6 の「domainは効かない」というnullを、ablationとは独立なPFI側から裏取りする。

## 3. per-case 信号の定義

各 `(seed, case_id, 特徴 f)` について、キャッシュ済みテストケースを置換前後で採点し以下を計算する。

- `base_R` = 置換前の Recall@K_correct（ケース単位、`[0,1]`、複数正解式では分数値）
- `perm_R_mean` = N_PERM=20 回の置換後 Recall の平均
- **`drop = base_R − perm_R_mean`** … 期待低下（連続・**主信号**、全ケース対象）
- **`flip`** = 1 なら「正解→不正解」反転。定義: `base_R == 1.0` かつ 過半（≥ N_PERM/2）の置換で `perm_R < 1.0`。それ以外は 0
- `flip_rate` = 置換のうち `perm_R < 1.0` となった割合（`base_R==1.0` のケースのみ意味を持つ）

**依存集合 / 頑健集合**（二値対比の母集団）:
- **依存** D(f) = { baseline で解けていた（`base_R==1.0`）うち `flip==1` }
- **頑健** R(f) = { `base_R==1.0` かつ `flip==0` }
- baseline解済みへ条件付けることで難易度交絡を除き、f の寄与を分離する。`base_R<1.0` のケースは二値対比から除外するが、連続 `drop` のプロファイルには全ケースを用いる。

**置換範囲**: per-case は **scope="case"（ケース内置換）のみ**で収集する。global（全体置換）は集約表用にのみ残す（ケース横断で混ざり per-case 解釈に不適）。

**冗長性の注意（既存 `analyze_pfi.py` の設計注意と同じ）**: 補完性/一貫性等は変数重なり情報を共有し冗長。単独置換は相棒が信号を復元するため、per-case の `flip` も**依存の下界**（過小評価）になる。対策として `GROUP_var`（変数重なり群まるごと置換）の per-case 版も併走し、「群として変数重なりに依存するケース」も同定する。

## 4. 特徴づけの軸と統計（深度A）

依存 D(f) vs 頑健 R(f) を以下の軸で対比する。

**属性軸**:
- 難易度4軸: `n_correct`(=|正解式|), `n_input`, `n_output`, `n_sources`
- `variant`（original / multisource / dae）
- **対象特徴の自ケース値** `sep_i(f) = mean_f(正解式の候補行) − mean_f(不正解式の候補行)`。キャッシュ済み `feats` と `corr` から直接算出（再結合不要）。単独特徴のみ定義（群は非対象）。

**統計**:
- 各属性軸で依存 vs 頑健の 平均±SEM、Mann–Whitney U 検定 p、効果量（rank-biserial r）
- `drop` 〜 `sep` の Spearman 順位相関（`base_R==1.0` の全記録上）
- 集計単位 = `(seed, case_id)` 記録。10 seed をプールし、各記録を1観測とする
- 有意性の主張は検定後のみ。図はエラーバー（SEM）必須。軸ラベルは定義済み語のみ（加藤研作文ルール／江口コメント準拠）

## 5. 実装（2ファイル・関心分離）

### 5.1 `analyze_pfi.py` の拡張（集約出力は不変）

- `score_to_RK(model, cache, feat_override=None, return_per_case=False)`:
  - `return_per_case=True` のとき `(aggregate, per_case_list)` を返す。`per_case_list[i]` = cache順に並んだケース単位の Recall@K_correct（内部の `case_metrics` から METRIC を抽出するだけ）。既定 `False` で現行と同一挙動。
- `run()`:
  - seed ごとに base per-case を1回計算: `base_agg, base_pc = score_to_RK(..., return_per_case=True)`。
  - seed ごとに `sep` を単独10特徴について1回計算（cache の `feats`・`corr` から）。正解行/不正解行が空なら NaN。
  - 既存の置換ループ（`scope="case"`, 各 label, N_PERM）で `_, perm_pc = score_to_RK(..., feat_override=pf, return_per_case=True)` を取得し、per-case に `perm_R` を累積、`flip` をカウント。
  - per-case 収集対象 = 単独10特徴 ＋ `GROUP_var`（`scope="case"` のみ）。
  - cache に `n_correct = len(rec["corr"])` を追加（他属性 `n_input/n_output/n_sources/variant/case_id` は既存）。
  - `experiments/pfi_per_case.json` に出力（下記スキーマ）。既存 `experiments/pfi_results.json` は現状のまま維持。

### 5.2 新規 `analyze_pfi_profile.py`

- `experiments/pfi_per_case.json` を読み込み、§4 のプロファイル＋検定を実施。
- 出力: `experiments/pfi_profile_stats.json` ＆ 図 `docs/figures/fig_pfi_dependence_profile.png` / `.pdf`。
- 和文フォント処理は `analyze_feature_x_difficulty.py` に準拠（japanize_matplotlib フォールバック、`pdf.fonttype=42`、`axes.unicode_minus=False`）。

**図の構成（2パネル）**:
- (a) 分離度プロファイル: x = `sep` の分位ビン、y = 平均 `drop` ± SEM、set-aware 3特徴の線。gComp/gCoh は右上がり・gDom は0付近平坦、を1枚で示す。
- (b) 属性対比: gComp・gCoh について、依存−頑健の標準化平均差（(mean_dep − mean_rob)/pooled_sd）を属性軸 {n_correct, n_input, n_output, n_sources, sep} で横棒。どの属性が依存ケースを特徴づけるかを示す。

**再実行性**: 重い訓練は `analyze_pfi.py` の1回のみ。プロファイル/作図は per-case dump から軽量に再実行できる。

## 6. データスキーマ

`experiments/pfi_per_case.json`:
```json
{
  "config": {"setting": "A", "set_mask": ["Comp","Coh","Dom"], "top_k": 50, "epochs": 15},
  "metric": "Recall@K_correct",
  "n_seeds": 10, "n_perm": 20, "scope": "case",
  "features": ["text_sim","io_jaccard","svd_sim","input_cov","output_cov",
               "specificity","domain","gComp","gCoh","gDom","GROUP_var"],
  "records": [
    {"seed": 42, "case_id": "core_032_v1", "feature": "gComp",
     "base_R": 1.0, "perm_R_mean": 0.55, "drop": 0.45,
     "flip": 1, "flip_rate": 0.8, "sep": 0.42,
     "n_correct": 2, "n_input": 9, "n_output": 1, "n_sources": 1,
     "variant": "original"}
  ]
}
```
（`sep` は群 `GROUP_var` では null。記録数の目安 ≈ 10 seed × 約365テストケース × 11特徴 ≈ 4万件、数MB。）

`experiments/pfi_profile_stats.json`:
```json
{
  "metric": "Recall@K_correct", "n_seeds": 10,
  "features": {
    "gComp": {
      "n_dependent": 0, "n_robust": 0, "flip_fraction_of_solved": 0.0,
      "spearman_drop_sep": {"rho": 0.0, "p": 0.0},
      "attrs": {
        "n_input": {"mean_dep": 0.0, "sem_dep": 0.0, "mean_rob": 0.0, "sem_rob": 0.0,
                     "mannwhitney_p": 0.0, "rank_biserial_r": 0.0}
      }
    }
  }
}
```

## 7. スコープ

**含む**: 設定A（1,823件）、10特徴すべて置換（per-case 図は set-aware 3種、`GROUP_var` は `pfi_profile_stats.json` に併記）、記述プロファイル＋検定、図1枚。

**含めない（初期スコープ外・spec に拡張点として明記）**:
- 設定B/DAE への拡張（`analyze_pfi.py` の設定切替は将来対応）
- 発見モデル（浅い決定木 / L1ロジスティックによるプロファイル自動抽出）
- 対象特徴の自ケース値による用量反応の完全連結（fingerprint図と一部重複するため別途）

## 8. 成功基準

1. `analyze_pfi.py` 実行後、`experiments/pfi_results.json` が従来と同一（集約PFI不変）で、かつ `experiments/pfi_per_case.json` が新規生成される。
2. `analyze_pfi_profile.py` が図＋ `pfi_profile_stats.json` を生成する。
3. 実測の健全性チェック: gDom の依存件数（flip）がほぼ0で、集約PFIの gDom≈0 と整合。gComp/gCoh は依存件数>0 かつ `drop`〜`sep` の Spearman ρ>0（符号の向きを確認、値は正直に報告）。
4. 図が作文ルール準拠: エラーバー明示・軸ラベルは定義済み語・主観語なし・自己完結キャプション。

## 9. リスクと対応

- **冗長性による過小評価**: 単独 `flip` は依存の下界。`GROUP_var` 併走で群依存を捕捉（§3）。
- **複数正解式での分数 Recall**: 二値 `flip` は `base_R==1.0` に限定、連続 `drop` は全ケース。両者併記で情報損失を回避（§3）。
- **seed間のケース重複**: 同一 case_id が複数 seed のテストに出現。プールして `(seed, case_id)` を1観測として扱い、過剰な独立性仮定を避ける（必要なら seed 内対応の安定性を補助的に確認）。
- **記述に留める**: 深度Aは相関・対比の記述であり因果主張はしない。因果は既存の greedy 介入実験が担う。
