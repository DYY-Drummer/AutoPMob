#!/bin/bash
# A1 フェーズ2: 最強構成（学習版 greedy・top_k=200）で第1段の変数寄り重みが効くか
#   条件: reranker-10S --train-greedy, w30-70（k=50 スイープの最良重み）, 10 seed
#   参照: 現行重み w70-30 の学習版 greedy は experiments/xg_A/train__{seed}.json
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xs1_k200
export VARIANTS="original,multisource_,dae_"
mkdir -p "$OUTDIR"

WT=0.3; WV=0.7; WL="w30-70"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local seed="$1"
  local out="$OUTDIR/train_${WL}__${seed}.json"
  if [ -s "$out" ]; then echo "skip seed=$seed (exists)"; return 0; fi
  if python3 set_aware_reranker.py --modes reranker-10S --seed-list "$seed" \
       --split stratified --variants "$VARIANTS" --top-k 200 --train-greedy \
       --stage1-w-text "$WT" --stage1-w-var "$WV" \
       --save-per-case --output "$out" > "$OUTDIR/train_${WL}__${seed}.log" 2>&1; then
    echo "done seed=$seed"
  else
    echo "FAIL seed=$seed (see $OUTDIR/train_${WL}__${seed}.log)"
  fi
}
export -f run_one
export WT WV WL OUTDIR VARIANTS

JOBS=$(mktemp)
for s in $SEEDS; do echo "$s"; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 1 bash -c 'run_one "$0"' < "$JOBS"
rm -f "$JOBS"
touch "$OUTDIR/sweep_k200.done"
echo "ALL JOBS FINISHED"
