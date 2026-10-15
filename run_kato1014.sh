#!/bin/bash
# 2026-10-14 加藤先生の指示 3・4・5 による再実験（第 1 段の重み 0.3/0.7、候補数 N1 を見直した設定）。
#   C（主結果）  : 一括の並べ直し（static）、逐次選択（推論のみ infer・訓練も逐次 train）× 設定 A・B
#   D（§5・§8） : 特徴量の組ごとに訓練し直すアブレーション（設定 A）
#   条件は run_stage1_weights.sh と同じ（層化分割・15 エポック・不正解 8 件など）で、
#   --stage1-w-text 0.3 --stage1-w-var 0.7 --top-k $TOPK --fast を付ける。
#   --fast は既定と同じ値を返す（tests/test_stage2_fast.py、seed 42 で正典の per-case と一致を確認）。
# 使い方: TOPK=2000 OUTDIR=experiments/kato1014/raw/n2000 [EXTRA="--hard-neg"] [JOBSET=static|C|D|all] [P=4] bash run_kato1014.sh
set -u
cd "$(dirname "$0")"
: "${TOPK:?TOPK を指定する（例: 2000、第 1 段なしは 11146）}"
: "${OUTDIR:?OUTDIR を指定する}"
export TOPK OUTDIR
export EXTRA="${EXTRA:-}"
JOBSET="${JOBSET:-all}"
P="${P:-4}"
mkdir -p "$OUTDIR"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_one() {
  local name="$1" setting="$2" seed="$3"; shift 3
  local variants
  case "$setting" in
    A) variants="original,multisource_,dae_" ;;
    B) variants="dae_" ;;
  esac
  local out="$OUTDIR/${name}_${setting}__${seed}.json"
  if [ -s "$out" ]; then echo "skip $name $setting seed=$seed (exists)"; return 0; fi
  # shellcheck disable=SC2086
  if python3 set_aware_reranker.py "$@" --seed-list "$seed" --split stratified \
       --variants "$variants" --top-k "$TOPK" --stage1-w-text 0.3 --stage1-w-var 0.7 \
       --fast $EXTRA --save-per-case --output "$out" > "${out%.json}.log" 2>&1; then
    echo "done $name $setting seed=$seed"
  else
    echo "FAIL $name $setting seed=$seed (see ${out%.json}.log)"
  fi
}
export -f run_one

JOBS=$(mktemp)
if [ "$JOBSET" = "static" ]; then
  for st in A B; do for s in $SEEDS; do echo "static $st $s --modes reranker-10S"; done; done >> "$JOBS"
fi
if [ "$JOBSET" = "C" ] || [ "$JOBSET" = "all" ]; then
  for st in A B; do for s in $SEEDS; do
    echo "static $st $s --modes reranker-10S"
    echo "infer $st $s --modes reranker-10S --greedy"
    echo "train $st $s --modes reranker-10S --train-greedy"
  done; done >> "$JOBS"
fi
if [ "$JOBSET" = "D" ] || [ "$JOBSET" = "all" ]; then
  for m in reranker-2 reranker-7 reranker-2+Spec reranker-2+Svd reranker-2+Dom \
           reranker-7+Comp reranker-7+Coh reranker-7+Dom; do
    safe=$(echo "$m" | tr "+" "_")
    for s in $SEEDS; do echo "$safe A $s --modes $m"; done
  done >> "$JOBS"
fi
echo "total jobs: $(wc -l < "$JOBS")  (${P} 並列, TOPK=$TOPK, EXTRA='$EXTRA')"
xargs -P "$P" -L 1 bash -c 'run_one "$@"' _ < "$JOBS"
rm -f "$JOBS"
touch "$OUTDIR/run_${JOBSET}.done"
echo "ALL JOBS FINISHED"
