#!/bin/bash
# 数据扩充串联脚本：quant_calc 完成后自动接力 reasoning
set -e
cd /opt/quant-llm
source finetune-env/bin/activate

echo "[$(date '+%H:%M:%S')] 等待 quant_calculations 完成..."

# 等待 quant_calc 进程结束
while pgrep -f "generate_quant_calculations.py" > /dev/null 2>&1; do
    sleep 30
done

echo "[$(date '+%H:%M:%S')] quant_calculations 已完成"
echo "[$(date '+%H:%M:%S')] 当前条数: $(wc -l < training-data/quant_calculations.jsonl)"

echo "[$(date '+%H:%M:%S')] 开始 reasoning_enhanced..."
python3 -u scripts/add_reasoning_chains.py 2>&1 | tee output/reasoning_log_v2.txt

echo "[$(date '+%H:%M:%S')] reasoning_enhanced 已完成"
echo "[$(date '+%H:%M:%S')] 输出条数: $(wc -l < training-data/reasoning_enhanced.jsonl 2>/dev/null || echo 0)"
echo "[$(date '+%H:%M:%S')] 全部数据扩充完成！"
