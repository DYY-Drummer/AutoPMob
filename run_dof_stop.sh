#!/bin/bash
# DoF=0 停止規則の評価：学習版greedy + --stop-dof
#   reranker-10S・層化分割・top_k=200・save-per-case・10 seed
#   2設定: dae(DAEのみ) と A(原型+複数文献+DAE=1823)
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xd_dof
export TOPK=200
mkdir -p "$OUTDIR"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local setting="$1" seed="$2"
  local out="$OUTDIR/${setting}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $setting seed=$seed (exists)"; return 0; fi
  local variants=""
  case "$setting" in
    dae) variants="dae_" ;;
    A)   variants="original,multisource_,dae_" ;;
  esac
  if python3 set_aware_reranker.py --modes reranker-10S --train-greedy --stop-dof \
       --seed-list "$seed" --split stratified --variants "$variants" --top-k "$TOPK" \
       --save-per-case --output "$out" > "$OUTDIR/${setting}__${seed}.log" 2>&1; then
    echo "done $setting seed=$seed"
  else
    echo "FAIL $setting seed=$seed (see $OUTDIR/${setting}__${seed}.log)"
  fi
}
export -f run_one

JOBS=$(mktemp)
for st in dae A; do for s in $SEEDS; do echo "$st $s"; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
