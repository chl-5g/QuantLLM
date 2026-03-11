#!/bin/bash
# QuantLLM 评估进度监控 + 自动重启
# 用法: bash watch_eval.sh          （查看进度）
#       bash watch_eval.sh --guard   （进程中断时自动重启，适合 cron）

LOG_FILE="/opt/quant-llm/output/eval.log"
PID_FILE="/opt/quant-llm/output/eval.pid"
RESULT_DIR="/opt/quant-llm/output"
VENV="/opt/quant-llm/finetune-env/bin/activate"
SCRIPT="/opt/quant-llm/scripts/evaluate.py"

# 检查评估进程
find_eval_pid() {
    # 优先用 PID 文件
    if [ -f "$PID_FILE" ]; then
        local saved_pid
        saved_pid=$(cat "$PID_FILE")
        if kill -0 "$saved_pid" 2>/dev/null; then
            echo "$saved_pid"
            return
        fi
    fi
    # 兜底: pgrep
    pgrep -f "python.*evaluate.py" | head -1
}

# 从日志解析进度
parse_progress() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "0 0"
        return
    fi
    # 匹配 [finetuned] 31/171... 或 [baseline] 10/171...
    local last_line
    last_line=$(grep -oP '\[(finetuned|baseline)\]\s+\K\d+/\d+' "$LOG_FILE" | tail -1)
    if [ -n "$last_line" ]; then
        local current total
        current=$(echo "$last_line" | cut -d/ -f1)
        total=$(echo "$last_line" | cut -d/ -f2)
        echo "$current $total"
    else
        echo "0 0"
    fi
}

# 检查是否已有结果文件
latest_result() {
    ls -t "${RESULT_DIR}"/eval_results*.json 2>/dev/null | head -1
}

# 启动评估
start_eval() {
    cd /opt/quant-llm || exit 1
    source "$VENV"
    nohup python "$SCRIPT" >> "$LOG_FILE" 2>&1 &
    echo "$!" > "$PID_FILE"
    echo "$!"
}

# === 主逻辑 ===

eval_pid=$(find_eval_pid)
read -r current total <<< "$(parse_progress)"
result_file=$(latest_result)

echo "========== QuantLLM 评估进度 =========="

# 已有完成的结果
if [ -n "$result_file" ]; then
    result_time=$(stat -c '%Y' "$result_file")
    now=$(date +%s)
    age_min=$(( (now - result_time) / 60 ))
    echo "  结果文件: $(basename "$result_file") (${age_min}分钟前)"

    # 从结果文件提取摘要
    python3 -c "
import json, sys
with open('$result_file') as f:
    data = json.load(f)
ft = data.get('finetuned', [])
if not ft:
    print('  (结果为空)')
    sys.exit()
print(f'  测试条数: {len(ft)}')
# ROUGE-L
rl = [r['rouge_l'] for r in ft if 'rouge_l' in r]
if rl:
    print(f'  ROUGE-L: {sum(rl)/len(rl):.3f} (avg, {len(rl)} holdout)')
# 结构化
sc = [r['structured_count'] for r in ft]
print(f'  结构化得分: {sum(sc)/len(sc):.1f} (avg)')
# 数值正确性
num = [r for r in ft if 'numeric_accuracy' in r and r['numeric_accuracy']]
if num:
    correct = sum(1 for r in num for name, ok in r['numeric_accuracy'] if ok)
    total_n = sum(len(r['numeric_accuracy']) for r in num)
    print(f'  数值正确: {correct}/{total_n}')
# 分类统计
cats = {}
for r in ft:
    c = r.get('category', 'unknown')
    cats[c] = cats.get(c, 0) + 1
print(f'  分类: {cats}')
" 2>/dev/null
fi

# 进度信息
if [ -n "$eval_pid" ]; then
    if [ "$total" -gt 0 ]; then
        pct=$(python3 -c "print(f'{${current}/${total}*100:.1f}')")
        echo "  进度: ${current}/${total} (${pct}%)"
    else
        echo "  进度: 模型加载中..."
    fi

    # 日志最后活动时间
    if [ -f "$LOG_FILE" ]; then
        log_time=$(stat -c '%Y' "$LOG_FILE")
        now=$(date +%s)
        log_age=$(( (now - log_time) / 60 ))
        echo "  日志更新: ${log_age}分钟前"
    fi

    echo "  评估进程: PID ${eval_pid} (运行中)"
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$gpu_mem" ] && echo "  GPU 显存: ${gpu_mem} MB"
else
    # 进程不在运行
    if [ -n "$result_file" ]; then
        echo "  评估进程: 未运行（评估已完成）"
    elif [ -f "$LOG_FILE" ] && [ "$current" -gt 0 ] && [ "$current" -lt "$total" ]; then
        echo "  进度: ${current}/${total} (中断)"
        echo "  评估进程: 未运行（异常中断！）"
        if [ "${1}" = "--guard" ]; then
            echo "  >>> 自动重启评估..."
            new_pid=$(start_eval)
            echo "  >>> 已重启: PID ${new_pid}, 日志: ${LOG_FILE}"
        else
            echo "  提示: 用 bash watch_eval.sh --guard 可自动重启"
        fi
    elif [ -f "$LOG_FILE" ]; then
        # 日志存在但没解析到进度，可能还在加载模型时中断
        echo "  评估进程: 未运行（可能在加载阶段中断）"
        if [ "${1}" = "--guard" ]; then
            echo "  >>> 自动重启评估..."
            new_pid=$(start_eval)
            echo "  >>> 已重启: PID ${new_pid}, 日志: ${LOG_FILE}"
        else
            echo "  提示: 用 bash watch_eval.sh --guard 可自动重启"
        fi
    else
        echo "  评估进程: 未运行（未启动过）"
    fi
fi

# 最后几行日志
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "--- 最近日志 ---"
    tail -5 "$LOG_FILE" | sed 's/^/  /'
fi

echo "======================================="
