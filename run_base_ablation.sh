#!/bin/bash
# 基本特徴量の個別追加（2特徴量に1つずつ）：baseline→7特徴量の +0.092 の内訳分解
# 設定A・層化分割・top-k 50・10シード（xd の他モードと同一条件）
set -u
cd "$(dirname "$0")"
mkdir -p experiments/xd

MODES="reranker-2+Svd reranker-2+InCov reranker-2+OutCov reranker-2+Spec reranker-2+Dom"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local mode="$1" seed="$2"
  local safe; safe=$(echo "$mode" | tr "+" "_")
  local out="experiments/xd/${safe}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $mode seed=$seed (exists)"; return 0; fi
  if python3 set_aware_reranker.py --modes "$mode" --seed-list "$seed" \
       --split stratified --variants "original,multisource_,dae_" \
       --save-per-case --output "$out" > "experiments/xd/${safe}__${seed}.log" 2>&1; then
    echo "done $mode seed=$seed"
  else
    echo "FAIL $mode seed=$seed"
  fi
}
export -f run_one

JOBS=$(mktemp)
for m in $MODES; do for s in $SEEDS; do echo "$m $s"; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
