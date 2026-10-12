#!/bin/bash
# A1 の補完: 設定 B（合成のみ）で第 1 段の重み 0.3/0.7 の本手法（reranker-10S）を 10 seed 実行する。
#   加藤先生の 8/31 報告への注釈（重みは複数の組で比べ、最良を本文に載せる）に応えるため、
#   設定 A だけだった w30-70 の本手法を設定 B でもそろえる。baseline の w30-70 は
#   experiments/external_baselines_B.json（tfidf__none__w30-70）に既にある。
#   既定重み w70-30 の参照は strat_B.json。条件（層化分割・top_k=50・15 epoch 等）は strat_B と同じ。
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xs1_B
mkdir -p "$OUTDIR"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local seed="$1"
  local out="$OUTDIR/reranker-10S_w30-70__${seed}.json"
  if [ -s "$out" ]; then echo "skip seed=$seed (exists)"; return 0; fi
  if python3 set_aware_reranker.py --modes reranker-10S --seed-list "$seed" \
       --split stratified --variants "dae_" --top-k 50 \
       --stage1-w-text 0.3 --stage1-w-var 0.7 \
       --save-per-case --output "$out" > "$OUTDIR/reranker-10S_w30-70__${seed}.log" 2>&1; then
    echo "done seed=$seed"
  else
    echo "FAIL seed=$seed (see $OUTDIR/reranker-10S_w30-70__${seed}.log)"
  fi
}
export -f run_one

echo $SEEDS | tr ' ' '\n' | xargs -P 5 -n 1 bash -c 'run_one "$0"'
touch "$OUTDIR/sweep_B_k50.done"
