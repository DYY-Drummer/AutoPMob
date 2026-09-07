# B1 LLM 等価判定の検証（第 2 判定器 Claude Fable 5.1）設計書

作成日: 2026-09-07 ／ 対象: `docs/論文化ギャップ分析_2026-08-30.md` の B1 ／ ステータス: **実施済み（2026-09-07）**。結果はギャップ分析 B1 と devlog を参照。判断規則の帰結: κ = 0.72 ≥ 0.6 だが Recall@K の差 > 0.03 → 両判定器の値を併記

## 1. 目的と成功基準

**目的**: LLM 直接生成の採点は判定器 claude-opus-4-8 の等価判定に依存しており、判定の妥当性が未検証（§6.9 第 4 項）。同じ 608 ペア（設定 A 183・B 279・全体 146）を第 2 判定器 claude-fable-5-1 に**同じプロンプト**で判定させ、判定間一致（一致率・Cohen's κ）と第 2 判定器のもとでの Recall@K / coverage を報告する。

**成功基準**:
1. 第 2 判定器の結果が `experiments/llm_<label>_equiv_results_fable51.json` に第 1 判定器と同じ形式（`per_case[].match_index`）で保存され、対象ケースと式の順序が第 1 判定器と厳密に一致する。
2. `experiments/judge_agreement_stats.json` に、設定ごとと全体プールで、(a) 「等価な生成式あり」ラベルの一致率と κ、(b) 「上位 K 件内に等価式あり」（Recall@K の判定単位）ラベルの一致率と κ、(c) 番号まで同じ割合、(d) 両判定器の Recall@K・coverage を出す。
3. 修論 §5.2（判定器の検証 1〜2 文）・§6.9 第 4 項（人手検証はなしのまま、第 2 判定器との一致を記載）・必要なら §6.1 の LLM 数値に幅を付ける。ビルド exit 0。
4. devlog・ギャップ分析（B1 ✅）・memory 更新。commit はしない。

## 2. 設計判断

| 項目 | 決定 | 理由 |
|---|---|---|
| 第 2 判定器 | `claude-fable-5-1`（本人指定）。adaptive thinking（既定・省略不可）、effort 既定（high）、`max_tokens` 16000 | 第 1 判定器 Opus 4.8 と同じ Anthropic 系列だが別世代・別モデル。ギャップ分析は「別ファミリ（例: Gemini）」を例示していたが本人が Fable を選択。同系列であることは修論に明記する |
| プロンプト | 第 1 判定器と**同一**（`evaluate_llm_equiv.py` の `JUDGE`）。`build_prompt()` に切り出して両者で共有（第 1 判定器の動作は不変） | 判定間一致はプロンプト差を含めてはならない |
| フォールバック | **使わない**（`fallbacks` 未指定）。`stop_reason == "refusal"` は記録して 1 回再試行、解消しなければ欠測として κ から除外し件数を報告 | フォールバック先が Opus 4.8（第 1 判定器）になると「第 2 判定器」でなくなる。化学工学の式の等価判定で拒否はまず起きない見込み |
| 対象ケース | 第 1 判定器の `per_case` の case_id と順序に従う（predictions ではなく結果ファイルを正）。K（正解式数）が一致することを assert | 第 1 判定器が判定できなかった 3 ケース（A 2 件・全体 1 件）を揃えて除外するため |
| 並列 | 3 ラベル（set_A / set_B / set_full）を別プロセスで同時実行、各プロセス内は逐次。`caffeinate -i` | 呼び出し 147 回×数十秒を 3 本に分ける。RateLimitError は既存どおり待って再試行 |
| 一致度の指標 | 二値ラベル 2 種の一致率と Cohen's κ（sklearn の `cohen_kappa_score` と照合するテスト付き）、番号一致率、両判定器の Recall@K / coverage とその差 | κ は偶然一致を補正した標準指標（ギャップ分析の指定） |
| 判断規則 | κ ≥ 0.6（substantial）かつ Recall@K の差 ≤ 0.03 なら §5.2 に 1〜2 文で「判定は判定器に依存しない」と書く。それ未満なら §6.1 の LLM 数値を「0.xx–0.yy（2 判定器）」の幅で示し、§6.9 に判定器依存を明記する | ギャップ分析 B1 の判断規則 |
| 費用 | 147 呼び出し、入力 ~800 トークン・出力（思考込み）数千トークン → 概算 15〜30 USD。実測トークン数を JSON に記録 | 本人の API キー。報告で実測を示す |

## 3. 構成

```
evaluate_llm_equiv.py            # 変更: build_prompt() を切り出し（judge() の動作は不変）
judge_second_opinion.py          # 新規: 第 2 判定器で同一ペアを再判定（--label, --judge-model, --tag, --limit）
analyze_judge_agreement.py       # 新規: 判定間一致（一致率・κ・番号一致・R@K/coverage 差）→ judge_agreement_stats.json
run_second_judge.sh              # 新規: 3 ラベル並列実行 + 分析
tests/test_judge_agreement.py    # 新規: κ の toy 検証（sklearn 照合）・整列・集計
experiments/llm_set_{A,B,full}_equiv_results_fable51.json
experiments/judge_agreement_stats.json
```

## 4. スコープ外
- 人手検証（本人の判定が要るため、必要なら次回）。
- 生成器の再実行（predictions は再利用）。
- 別ファミリ（Gemini 等）の第 3 判定器。
