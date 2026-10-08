#!/bin/bash
# 変数の説明による照合の本実験（仕様 docs/superpowers/specs/2026-10-01-variable-description-matching-design.md）
#   設定 A・B を並列に、条件 S0,S1(+S1-r50/r100),D1,D0 × 10 seed。
#   続けて τ±0.02（設定 A・seed 42・D1）と、第 1 段の重みの掃引による被覆率（設定 A）。
#   どれかの run が失敗したら、失敗した run とログの場所を標準エラーに出して終了コード 1 で止まる。ALL DONE は全部成功したときだけ。
set -u
cd "$(dirname "$0")"
LOG=experiments/var_desc_logs
mkdir -p "$LOG"
pids=()
for S in A B; do
  python3 run_var_description.py --setting "$S" --conds S0,S1,D1,D0 --renames 0.5,1.0 \
    --save-per-case --output "experiments/var_desc_${S}.json" > "$LOG/main_${S}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do  # 素の wait は終了状態を捨てるので、PID ごとに待つ（片方が失敗しても、もう片方は終わるまで待つ）
  wait "$p" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "MAIN RUN FAILED (see $LOG/main_*.log)" >&2
  exit 1
fi
for D in -0.02 +0.02; do
  python3 run_var_description.py --setting A --conds D1 --seed-list 42 --tau-delta "$D" \
    --output "experiments/var_desc_tau${D}.json" > "$LOG/tau${D}.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "TAU RUN FAILED (--tau-delta $D; see $LOG/tau${D}.log)" >&2
    exit 1
  fi
done
python3 run_var_description.py --setting A --conds S0,S1,D1,D0 --coverage \
  --output experiments/var_desc_coverage.json > "$LOG/coverage.log" 2>&1
if [ $? -ne 0 ]; then
  echo "COVERAGE RUN FAILED (see $LOG/coverage.log)" >&2
  exit 1
fi
echo "ALL DONE"
