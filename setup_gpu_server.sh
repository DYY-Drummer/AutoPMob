#!/usr/bin/env bash
# Phase 5: GPU サーバ初期セットアップ + Phase 5.A sweep キックオフ補助.
# 前提:
#   - ~/.ssh/config に Host autopmob-a100 のエントリがある（HostName/User/IdentityFile 設定済み）
#   - PC1 (A100 x4) を使う想定。PC3 (H100) を使う場合は HOST_ALIAS を変更
#
# 使い方:
#   ./setup_gpu_server.sh           # サーバ側にリポを clone + コンテナ作成 + 環境確認
#   ./setup_gpu_server.sh sync      # ローカルの最新コード差分を rsync で送る（commit 不要）
#   ./setup_gpu_server.sh smoke     # サーバ上で 1 config だけ smoke 実行（5-10 分）
#   ./setup_gpu_server.sh sweep-A   # サーバ上で Phase 5.A の 36 config を 4 GPU 並列実行
#   ./setup_gpu_server.sh fetch     # サーバ上の experiments/phase5_sweep/ をローカルに rsync で取得
#
# 設定ここから -------------------------------------------------------
HOST_ALIAS="${HOST_ALIAS:-autopmob-a100}"     # ~/.ssh/config のホスト名
SERVER_PROJECT_DIR="${SERVER_PROJECT_DIR:-~/project/AutoPMob}"
CONTAINER_NAME="${CONTAINER_NAME:-kato_autopmob}"
DOCKER_IMAGE="${DOCKER_IMAGE:-research_env_cuda12:latest}"
GIT_REMOTE="${GIT_REMOTE:-}"                   # 例: git@github.com:Username/AutoPMob.git
# 設定ここまで -------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")"

ssh_exec() {
    # SSH 接続を 1 回張って中で複数コマンドを実行（パスフレーズ入力を最小化）
    ssh -o "ServerAliveInterval=30" "$HOST_ALIAS" "$@"
}

docker_exec() {
    ssh_exec "docker exec $CONTAINER_NAME bash -lc '$*'"
}

cmd_init() {
    echo "=== Step 1: ssh 疎通確認 ==="
    ssh_exec 'hostname; nvidia-smi -L'

    if [[ -z "$GIT_REMOTE" ]]; then
        echo "WARN: GIT_REMOTE が未設定です。最初の clone はスキップして rsync で送ります。"
        echo "      （commit 済みなら GIT_REMOTE を設定し直して clone する方が確実）"
        cmd_sync_full
    else
        echo "=== Step 2: サーバ側にリポを clone ==="
        ssh_exec "mkdir -p \$(dirname $SERVER_PROJECT_DIR) && \
                  if [ ! -d $SERVER_PROJECT_DIR/.git ]; then \
                    git clone $GIT_REMOTE $SERVER_PROJECT_DIR; \
                  else echo 'already cloned, pulling'; cd $SERVER_PROJECT_DIR && git pull; fi"
    fi

    echo "=== Step 3: Docker コンテナの存在確認・作成 ==="
    ssh_exec "if docker ps -a --format '{{.Names}}' | grep -q '^${CONTAINER_NAME}\$'; then \
                echo 'container exists'; docker start $CONTAINER_NAME || true; \
              else \
                docker run -itd --gpus all --name $CONTAINER_NAME \
                  -v $SERVER_PROJECT_DIR:/workspace/project \
                  -v /data:/data \
                  $DOCKER_IMAGE bash; \
              fi"

    echo "=== Step 4: コンテナ内で pip 依存をインストール ==="
    docker_exec "cd /workspace/project && pip3 install -q -r requirements.txt"

    echo "=== Step 5: torch.cuda 確認 ==="
    docker_exec 'python3 -c "import torch; print(\"cuda available:\", torch.cuda.is_available(), \"; n_gpus:\", torch.cuda.device_count())"'

    echo ""
    echo "OK. 次は:  ./setup_gpu_server.sh smoke"
}

cmd_sync_full() {
    echo "=== ローカル→サーバ rsync (PDFs, .git/, .venv/ などを除外) ==="
    ssh_exec "mkdir -p $SERVER_PROJECT_DIR"
    rsync -avzh --delete \
        --exclude '.git/' --exclude '__pycache__/' --exclude '.venv/' \
        --exclude '*.pyc' --exclude '.claude/' --exclude '/extracted/' \
        --exclude '/docs/figures/*.png' \
        --exclude '/input_pdfs/' --exclude '/pdfs/' \
        --exclude '/docs/paper_collection/' --exclude '/docs/*.pdf' \
        --exclude '*.dvi' --exclude '*.aux' --exclude '*.toc' --exclude '*.log' \
        ./ "$HOST_ALIAS:$SERVER_PROJECT_DIR/"
}

cmd_smoke() {
    echo "=== 1 config smoke test (1 seed, 3 epochs, dae_X1 のみ) ==="
    docker_exec "cd /workspace/project && python3 set_aware_reranker.py \
                   --seeds 1 --modes baseline,reranker-10S \
                   --variants dae_X1 \
                   --output /tmp/phase5_smoke.json \
                   --hidden-dim 128 --epochs 3 --batch-size 8 --n-neg-samples 4"
}

cmd_sweep_A() {
    echo "=== Phase 5.A sweep: 36 configs × 4 GPU 並列 ==="
    docker_exec "cd /workspace/project && \
                 nohup python3 -u run_phase5_sweep.py --phase A \
                   --max-parallel 4 --n-gpus 4 \
                   > /tmp/phase5_A.log 2>&1 &
                 echo 'sweep started, see /tmp/phase5_A.log on server'"
}

cmd_sweep_A_tail() {
    ssh_exec "docker logs -f --tail 50 $CONTAINER_NAME 2>/dev/null || tail -F /tmp/phase5_A.log"
}

cmd_fetch() {
    echo "=== サーバ→ローカル rsync (experiments/phase5_sweep/) ==="
    rsync -avzh "$HOST_ALIAS:$SERVER_PROJECT_DIR/experiments/phase5_sweep/" \
                ./experiments/phase5_sweep/
}

case "${1:-help}" in
    init)        cmd_init ;;
    sync)        cmd_sync_full ;;
    smoke)       cmd_smoke ;;
    sweep-A)     cmd_sweep_A ;;
    sweep-A-tail) cmd_sweep_A_tail ;;
    fetch)       cmd_fetch ;;
    help|"")
        echo "Usage: $0 {init|sync|smoke|sweep-A|sweep-A-tail|fetch}"
        echo ""
        echo "推奨実行順序:"
        echo "  1. ssh-keygen + ssh-copy-id でパスワード無し SSH を確立"
        echo "  2. ~/.ssh/config に Host $HOST_ALIAS エントリを追加"
        echo "  3. $0 init      # サーバ初期化（clone or rsync + コンテナ作成 + 環境確認）"
        echo "  4. $0 smoke     # 1 config だけ smoke で動作確認（~5 分）"
        echo "  5. $0 sweep-A   # 本番 sweep をバックグラウンドで起動（~90 分）"
        echo "  6. $0 sweep-A-tail   # 進捗確認（Ctrl-C で抜ける）"
        echo "  7. $0 fetch     # 結果取得"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
