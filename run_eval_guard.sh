#!/bin/bash
# 评估守护脚本 - 在 screen 中运行，挂了自动重启
cd /opt/quant-llm || exit 1
source finetune-env/bin/activate

LOG="/opt/quant-llm/output/eval_log.txt"
CHECKPOINT="/opt/quant-llm/output/eval_checkpoint.jsonl"
RESULT="/opt/quant-llm/output/eval_results.json"

while true; do
    # 如果结果文件已存在，说明评估完成
    if [ -f "$RESULT" ]; then
        echo "[$(date)] 评估已完成，结果文件存在: $RESULT" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] 启动评估..." | tee -a "$LOG"
    python scripts/evaluate.py --max-holdout 30 2>&1 | tee -a "$LOG"
    EXIT_CODE=$?

    # 检查是否完成
    if [ -f "$RESULT" ]; then
        echo "[$(date)] 评估完成！退出码: $EXIT_CODE" | tee -a "$LOG"
        break
    fi

    echo "[$(date)] 评估中断（退出码: $EXIT_CODE），10秒后自动重启..." | tee -a "$LOG"
    sleep 10
done

echo "[$(date)] 守护脚本退出" | tee -a "$LOG"
