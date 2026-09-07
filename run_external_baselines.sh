#!/bin/bash
# A2: 外部ベースライン（BM25・E5 密ベクトル）を設定 A・B で一括実行し、検定まで行う。
#   - 5 スコアラー (tfidf:none, tfidf:minmax, bm25:minmax, e5:minmax, e5io:minmax)
#   - 3 重み (文章のみ / 0.7:0.3 = 表 3 と同じ / 0.3:0.7 = A1 最良)
#   - 層化分割 10 seed・全 DB 順位付け・学習なし・被覆率 (k=10..400) 付き
#   E5 の埋め込みは experiments/embeddings/ にキャッシュされる（初回のみ数分）。
#   長時間 CPU/GPU ジョブなので caffeinate -i bash run_external_baselines.sh で実行する。
set -eu
cd "$(dirname "$0")"
mkdir -p experiments
for S in A B; do
  echo "=== Setting $S: $(date) ==="
  python3 external_baselines.py --setting "$S" --coverage \
    --output "experiments/external_baselines_${S}.json" 2>&1 | tee "experiments/external_baselines_${S}.log"
done
echo "=== analysis: $(date) ==="
python3 analyze_external_baselines.py 2>&1 | tee experiments/external_baselines_stats.log
echo "ALL DONE $(date)"
