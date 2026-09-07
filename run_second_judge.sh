#!/bin/bash
# B1: 第 2 判定器（既定 claude-fable-5-1）で LLM 直接生成の等価判定を 3 ラベル並列にやり直し、
#     第 1 判定器（claude-opus-4-8）との一致度を分析する。
#   python3 judge_second_opinion.py --label set_A|set_B|set_full --tag $TAG
#   python3 analyze_judge_agreement.py --tag $TAG → experiments/judge_agreement_stats.json
# 長時間の API ジョブなので caffeinate -i bash run_second_judge.sh で実行する。
set -u
cd "$(dirname "$0")"
TAG=${TAG:-fable51}
MODEL=${MODEL:-claude-fable-5-1}
for L in set_A set_B set_full; do
  python3 judge_second_opinion.py --label "$L" --judge-model "$MODEL" --tag "$TAG" \
    > "experiments/judge_${TAG}_${L}.log" 2>&1 &
done
wait
python3 analyze_judge_agreement.py --tag "$TAG" 2>&1 | tee "experiments/judge_agreement_${TAG}.log"
echo "ALL DONE $(date)"
