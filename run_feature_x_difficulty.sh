#!/bin/bash
# 特徴量×難易度クロス：(mode × seed) を単位に並列実行（設定A・層化分割・10 seed）
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xd
export VARIANTS="original,multisource_,dae_"
mkdir -p "$OUTDIR"

MODES="reranker-7 reranker-7+Comp reranker-7+Coh reranker-7+Dom reranker-10S"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local mode="$1" seed="$2"
  local safe; safe=$(echo "$mode" | tr "+" "_")
  local out="$OUTDIR/${safe}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $mode seed=$seed (exists)"; return 0; fi
  if python3 set_aware_reranker.py --modes "$mode" --seed-list "$seed" \
       --split stratified --variants "$VARIANTS" \
       --save-per-case --output "$out" > "$OUTDIR/${safe}__${seed}.log" 2>&1; then
    echo "done $mode seed=$seed"
  else
    echo "FAIL $mode seed=$seed (see $OUTDIR/${safe}__${seed}.log)"
  fi
}
export -f run_one

# ジョブ一覧（mode,seed）→ 6並列
JOBS=$(mktemp)
for m in $MODES; do for s in $SEEDS; do echo "$m $s"; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"

xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
