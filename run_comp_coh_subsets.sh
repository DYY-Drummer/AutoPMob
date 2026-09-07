#!/bin/bash
# 補完性/一貫性の寄与が偶然でないかの確認：
#   性質の違う2つの訓練ケース集合（DAEのみ / 複数文献のみ）で reranker-7/+Comp/+Coh を再訓練。
#   層化分割・top-k 50・10シード（既存の設定A実験と同条件、対象ケースのみ変更）。
set -u
cd "$(dirname "$0")"
mkdir -p experiments/xd_cc

MODES="reranker-7 reranker-7+Comp reranker-7+Coh"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local subset="$1" mode="$2" seed="$3"
  local variants
  case "$subset" in
    dae) variants="dae_" ;;
    ms)  variants="multisource_" ;;
  esac
  local safe; safe=$(echo "$mode" | tr "+" "_")
  local out="experiments/xd_cc/${subset}_${safe}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $subset $mode $seed"; return 0; fi
  if python3 set_aware_reranker.py --modes "$mode" --seed-list "$seed" \
       --split stratified --variants "$variants" \
       --save-per-case --output "$out" > "experiments/xd_cc/${subset}_${safe}__${seed}.log" 2>&1; then
    echo "done $subset $mode $seed"
  else
    echo "FAIL $subset $mode $seed"
  fi
}
export -f run_one

JOBS=$(mktemp)
for sub in dae ms; do for m in $MODES; do for s in $SEEDS; do echo "$sub $m $s"; done; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
