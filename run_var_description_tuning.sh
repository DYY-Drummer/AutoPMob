#!/bin/bash
# 変数の説明による照合の追加実験: 検証用ケースで τ と第 1 段の重みを選ぶ（仕様 docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §7）
#   設定 A・B を、seed の前半（42,123,456,789,1024）と後半（2024,3141,5926,7777,9999）に分けて、4 プロセスで並列に実行する。
#   部分ファイルとログは experiments/var_desc_logs/ に書く（追跡しない）。4 つとも成功したら、設定ごとに結合して
#   experiments/var_desc_tuned_A.json / var_desc_tuned_B.json にする。
#   どれかが失敗したら、ログの場所を標準エラーに出して終了コード 1 で止まる。ALL DONE は全部成功したときだけ。
set -u
cd "$(dirname "$0")"
LOG=experiments/var_desc_logs
mkdir -p "$LOG"
HALF1=42,123,456,789,1024
HALF2=2024,3141,5926,7777,9999
pids=()
for S in A B; do
  for PART in 1 2; do
    if [ "$PART" -eq 1 ]; then SEEDS=$HALF1; else SEEDS=$HALF2; fi
    python3 tune_var_description.py run --setting "$S" --seed-list "$SEEDS" --save-per-case \
      --output "$LOG/tune_${S}_${PART}.json" > "$LOG/tune_${S}_${PART}.log" 2>&1 &
    pids+=($!)
  done
done
fail=0
for p in "${pids[@]}"; do  # 素の wait は終了状態を捨てるので、PID ごとに待つ（どれかが失敗しても、残りは終わるまで待つ）
  wait "$p" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "TUNING RUN FAILED (see $LOG/tune_*.log)" >&2
  exit 1
fi
for S in A B; do
  python3 tune_var_description.py merge --setting "$S" \
    --inputs "$LOG/tune_${S}_1.json" "$LOG/tune_${S}_2.json" \
    --output "experiments/var_desc_tuned_${S}.json" > "$LOG/tune_merge_${S}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "TUNING MERGE FAILED (setting $S; see $LOG/tune_merge_${S}.log)" >&2
    exit 1
  fi
done
echo "ALL DONE"
