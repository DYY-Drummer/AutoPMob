#!/bin/bash
# baseline→reranker-7 の分解用中間条件：同じ2特徴量（text_sim, io_jaccard）をMLPで学習
# 設定A・層化分割・top-k 50・10シード（xd の他モードと同一条件）
set -u
cd "$(dirname "$0")"
mkdir -p experiments/xd

run_one() {
  local s="$1"
  local out="experiments/xd/reranker-2__${s}.json"
  if [ -s "$out" ]; then echo "skip $s (exists)"; return 0; fi
  if python3 set_aware_reranker.py --modes reranker-2 --seed-list "$s" \
       --split stratified --variants "original,multisource_,dae_" \
       --save-per-case --output "$out" > "experiments/xd/reranker-2__${s}.log" 2>&1; then
    echo "done $s"
  else
    echo "FAIL $s (see experiments/xd/reranker-2__${s}.log)"
  fi
}
export -f run_one

printf '%s\n' 42 123 456 789 1024 2024 3141 5926 7777 9999 | \
  xargs -P 6 -n 1 bash -c 'run_one "$0"'
echo "ALL JOBS FINISHED: $(ls experiments/xd/reranker-2__*.json 2>/dev/null | wc -l | tr -d ' ')/10"
