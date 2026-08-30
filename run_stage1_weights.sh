#!/bin/bash
# A1: 第1段の混合重みスイープ（設定A・層化分割・top_k=50・10 seed・save-per-case）
#   モード: reranker-10S（静的再ランク＝要訓練）と baseline（全DB順位付け＝訓練なし）
#   重み: w50-50 / w30-70 / w00-100
#   現行 w70-30 は既存の experiments/xd/reranker-10S__*.json と strat_A.json を再利用する
#   （本セッションで既定重みの再実行が seed=42 の正典値を再現することを確認済み）。
set -u
cd "$(dirname "$0")"
export OUTDIR=experiments/xs1
export VARIANTS="original,multisource_,dae_"
mkdir -p "$OUTDIR"

MODES="reranker-10S baseline"
WLABELS="w50-50 w30-70 w00-100"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local mode="$1" wl="$2" seed="$3"
  local wt wv
  case "$wl" in
    w70-30)  wt=0.7; wv=0.3 ;;
    w50-50)  wt=0.5; wv=0.5 ;;
    w30-70)  wt=0.3; wv=0.7 ;;
    w00-100) wt=0.0; wv=1.0 ;;
    *) echo "unknown weight label: $wl"; return 1 ;;
  esac
  local safe; safe=$(echo "$mode" | tr "+" "_")
  local out="$OUTDIR/${safe}_${wl}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $mode $wl seed=$seed (exists)"; return 0; fi
  if python3 set_aware_reranker.py --modes "$mode" --seed-list "$seed" \
       --split stratified --variants "$VARIANTS" --top-k 50 \
       --stage1-w-text "$wt" --stage1-w-var "$wv" \
       --save-per-case --output "$out" > "$OUTDIR/${safe}_${wl}__${seed}.log" 2>&1; then
    echo "done $mode $wl seed=$seed"
  else
    echo "FAIL $mode $wl seed=$seed (see $OUTDIR/${safe}_${wl}__${seed}.log)"
  fi
}
export -f run_one

JOBS=$(mktemp)
for m in $MODES; do for w in $WLABELS; do for s in $SEEDS; do echo "$m $w $s"; done; done; done > "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (6並列)"
xargs -P 6 -n 3 bash -c 'run_one "$0" "$1" "$2"' < "$JOBS"
rm -f "$JOBS"
touch "$OUTDIR/sweep_k50.done"
echo "ALL JOBS FINISHED"
