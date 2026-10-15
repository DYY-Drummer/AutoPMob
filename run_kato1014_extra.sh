#!/bin/bash
# 2026-10-14 の指示 5（§5・§8・§9.1 のやり直し）のうち、run_kato1014.sh 以外の実行。設定 A。
#   分野ラベルの合併（kato1014_domain_merge.py：orig・L1・K100・K40 × 10 分割。x7 は群で数える。旧 domain_merge/ は
#   群の名前で部分文字列を比べた誤った版で、使わない）
#   PFI（kato1014_pfi.py：10 分割、シャッフル 20 回）
#   誤りの位置（kato1014_case_outcomes.py：乱数の種 42）
#   話題の特徴量の AUC と候補内の相関（kato1014_topic_diag.py：訓練なし）
# 使い方: TOPK=2000 OUTDIR=experiments/kato1014/raw/n2000_hard TAG=n2000_hard [EXTRA="--hard-neg"] [P=4] bash run_kato1014_extra.sh
set -u
cd "$(dirname "$0")"
: "${TOPK:?}"; : "${OUTDIR:?}"; : "${TAG:?}"
export TOPK OUTDIR TAG
export EXTRA="${EXTRA:-}"
P="${P:-4}"
mkdir -p "$OUTDIR/domain_merge_v2" "$OUTDIR/pfi"
SEEDS="42 123 456 789 1024 2024 3141 5926 7777 9999"

run_job() {
  local kind="$1" a1="$2" a2="${3:-}"
  local out log
  case "$kind" in
    merge)
      out="$OUTDIR/domain_merge_v2/${a1}__${a2}.json"; log="${out%.json}.log"
      [ -s "$out" ] && { echo "skip $kind $a1 $a2"; return 0; }
      # shellcheck disable=SC2086
      python3 kato1014_domain_merge.py --merge "$a1" --seed "$a2" --top-k "$TOPK" $EXTRA --output "$out" > "$log" 2>&1 ;;
    pfi)
      out="$OUTDIR/pfi/pfi_A__${a1}.json"; log="${out%.json}.log"
      [ -s "$out" ] && { echo "skip $kind $a1"; return 0; }
      # shellcheck disable=SC2086
      python3 kato1014_pfi.py --seed "$a1" --top-k "$TOPK" $EXTRA --output "$out" > "$log" 2>&1 ;;
    outcomes)
      out="experiments/kato1014/case_outcomes_${TAG}.json"; log="$OUTDIR/case_outcomes.log"
      [ -s "$out" ] && { echo "skip $kind"; return 0; }
      # shellcheck disable=SC2086
      python3 kato1014_case_outcomes.py --top-k "$TOPK" --tag "$TAG" $EXTRA > "$log" 2>&1 ;;
    topic)
      out="experiments/kato1014/topic_diag_${TAG}.json"; log="$OUTDIR/topic_diag.log"
      [ -s "$out" ] && { echo "skip $kind"; return 0; }
      python3 kato1014_topic_diag.py --top-k "$TOPK" --tag "$TAG" > "$log" 2>&1 ;;
  esac && echo "done $kind $a1 $a2" || echo "FAIL $kind $a1 $a2 (see $log)"
}
export -f run_job

JOBS=$(mktemp)
echo "topic" >> "$JOBS"
echo "outcomes" >> "$JOBS"
for s in $SEEDS; do echo "pfi $s"; done >> "$JOBS"
# orig は群で数える x7 が元の x7 と同じになる確認（reranker-2_Dom_A と per-case が一致するはず）
for m in orig L1 K100 K40; do for s in $SEEDS; do echo "merge $m $s"; done; done >> "$JOBS"
echo "total jobs: $(wc -l < "$JOBS")  (${P} 並列, TOPK=$TOPK, EXTRA='$EXTRA')"
xargs -P "$P" -L 1 bash -c 'run_job "$@"' _ < "$JOBS"
rm -f "$JOBS"
touch "$OUTDIR/run_extra.done"
echo "ALL JOBS FINISHED"
