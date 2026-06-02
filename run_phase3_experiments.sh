#!/bin/bash
# Phase 3: 3 つの制約付き実験を順次実行
set -e
cd /Users/kazuhiromiyamura/Desktop/AutoPMob
SCRIPT="python3 set_aware_reranker.py --seeds 10 --modes baseline,reranker-10S"
LOG_DIR=/tmp

echo "========================================"
echo "Phase 3 実験開始: $(date)"
echo "========================================"

# 実験 A: original + multisource_v3 + dae_*
echo ""
echo "=== 実験 A: original + multisource_v3 + dae_* ==="
$SCRIPT --variants "original,multisource_v3,dae_" \
        --output experiments/phase3_A_original_multi_dae.json \
        > $LOG_DIR/phase3_A.log 2>&1
echo "  完了: $(date)"

# 実験 B: dae_* のみ
echo ""
echo "=== 実験 B: dae_* のみ ==="
$SCRIPT --variants "dae_" \
        --output experiments/phase3_B_dae_only.json \
        > $LOG_DIR/phase3_B.log 2>&1
echo "  完了: $(date)"

echo ""
echo "========================================"
echo "Phase 3 全実験完了: $(date)"
echo "========================================"
