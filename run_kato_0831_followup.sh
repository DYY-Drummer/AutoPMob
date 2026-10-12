#!/bin/bash
# 8/31 報告への加藤先生の注釈に応える追加実行（2026-10-12）。
#   (1) 最良重みの baseline（文章 0.4／変数 0.6）を per-case 付きで設定 A・B に実行する
#       → 正解式数・入力変数数ごとの差を、最良重みの baseline に対して数え直すため。
#       集計値は experiments/external_baselines_{A,B}.json の tfidf__none__w40-60 と同じになるはず。
#   (2) 逐次選択の 3 通りの比較を候補数 k=50 でも行う（注釈「なぜ k=50 で行わないのか」）。
#       一括（固定参照）の k=50 は既存の strat_{A,B}.json を使う。ここでは推論のみ逐次（infer）と
#       学習も逐次（train）を設定 B → 設定 A の順に実行する。条件は run_greedy_3way.sh と同じで k だけ 50。
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xk50
mkdir -p "$OUTDIR"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local job="$1" setting="$2" seed="$3"
  local variants
  case "$setting" in
    A) variants="original,multisource_,dae_" ;;
    B) variants="dae_" ;;
  esac
  local out="$OUTDIR/${job}_${setting}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $job $setting seed=$seed (exists)"; return 0; fi
  local args
  case "$job" in
    baseline40) args="--modes baseline --stage1-w-text 0.4 --stage1-w-var 0.6" ;;
    infer)      args="--modes reranker-10S --greedy" ;;
    train)      args="--modes reranker-10S --train-greedy" ;;
  esac
  if python3 set_aware_reranker.py $args --seed-list "$seed" --split stratified \
       --variants "$variants" --top-k 50 --save-per-case --output "$out" \
       > "$OUTDIR/${job}_${setting}__${seed}.log" 2>&1; then
    echo "done $job $setting seed=$seed"
  else
    echo "FAIL $job $setting seed=$seed (see $OUTDIR/${job}_${setting}__${seed}.log)"
  fi
}
export -f run_one

JOBS=$(mktemp)
for st in A B; do for s in $SEEDS; do echo "baseline40 $st $s"; done; done > "$JOBS"
for st in B A; do for j in infer train; do for s in $SEEDS; do echo "$j $st $s"; done; done; done >> "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"' < "$JOBS"
rm -f "$JOBS"
echo "ALL JOBS FINISHED"
