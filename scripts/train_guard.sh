#!/bin/bash
# ============================================================
# 训练守护脚本 — OOM/崩溃自动恢复
# 由 systemd quantllm-train.service 调用
# train.py 内置 checkpoint 恢复逻辑，重启后自动从最新 checkpoint 继续
# ============================================================

set -e

PROJECT_DIR="/opt/quant-llm"
VENV="$PROJECT_DIR/finetune-env/bin/python3"
TRAIN_SCRIPT="$PROJECT_DIR/scripts/train.py"
LOG_DIR="$PROJECT_DIR/output"
OLLAMA_URL="http://localhost:11434"

MAX_RETRIES=5
RETRY_COUNT=0

cd "$PROJECT_DIR"

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练启动 (第 $((RETRY_COUNT+1)) 次)"

    # 释放 ollama 显存
    curl -s "$OLLAMA_URL/api/generate" -d '{"model":"qwen3:14b","keep_alive":0}' >/dev/null 2>&1 || true
    curl -s "$OLLAMA_URL/api/generate" -d '{"model":"deepseek-r1:32b","keep_alive":0}' >/dev/null 2>&1 || true
    sleep 3

    # 检查 GPU 显存
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "${gpu_mem:-0}" -gt 5000 ]; then
        echo "[WARN] GPU 显存占用 ${gpu_mem}MB，等待 30s..."
        sleep 30
        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ "${gpu_mem:-0}" -gt 5000 ]; then
            echo "[ERROR] GPU 显存仍占用 ${gpu_mem}MB，跳过本次尝试"
            RETRY_COUNT=$((RETRY_COUNT + 1))
            sleep 60
            continue
        fi
    fi

    # 运行训练
    $VENV $TRAIN_SCRIPT 2>&1 | tee -a "$LOG_DIR/train_v3.log"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练正常完成！"
        exit 0
    fi

    # 检查是否 OOM
    if dmesg -T 2>/dev/null | tail -20 | grep -q "Out of memory.*python3"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到 OOM，等待 60s 后重试（从 checkpoint 恢复）..."
        sleep 60
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练异常退出 (code=$EXIT_CODE)，等待 30s 后重试..."
        sleep 30
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 已达最大重试次数 ($MAX_RETRIES)，停止"
exit 1
