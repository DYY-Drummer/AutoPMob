#!/bin/bash
# 3者比較：静的参照 / 推論のみgreedy / 学習版greedy
#   すべて reranker-10S・DAEのみ(設定B相当,最難)・層化分割・top_k=200・save-per-case
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xg
export VARIANTS="dae_"
export TOPK=200
mkdir -p "$OUTDIR"

SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

# config: static(なし) / infer(--greedy) / train(--train-greedy)
run_one() {
  local config="$1" seed="$2"
  local out="$OUTDIR/${config}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $config seed=$seed (exists)"; return 0; fi
  local flag=""
  case "$config" in
    static) flag="" ;;
    infer)  flag="--greedy" ;;
    train)  flag="--train-greedy" ;;
  esac
  if python3 set_aware_reranker.py --modes reranker-10S $flag \
       --seed-list "$seed" --split stratified --variants "$VARIANTS" --top-k "$TOPK" \
       --save-per-case --output "$out" > "$OUTDIR/${config}__${seed}.log" 2>&1; then
    echo "done $config seed=$seed"
  else
    echo "FAIL $config seed=$seed (see $OUTDIR/${config}__${seed}.log)"
  fi
}
export -f run_one

JOBS=$(mktemp)
for c in static infer train; do for s in $SEEDS; do echo "$c $s"; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 2 bash -c 'run_one "$0" "$1"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
