#!/bin/bash
# 因子数据更新脚本（可由 cron/systemd 定时触发）

set -e

PROJECT_DIR="/opt/quant-llm"
VENV="$PROJECT_DIR/finetune-env/bin/activate"
SCRIPT="$PROJECT_DIR/scripts/build_stock_factors.py"
LOG_DIR="$PROJECT_DIR/logs/factors"

mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/factors_${TS}.log"

cd "$PROJECT_DIR"
source "$VENV"

python3 "$SCRIPT" --max-symbols 1200 --roe-limit 260 >"$LOG_FILE" 2>&1
echo "[FACTORS] updated, log=$LOG_FILE"
