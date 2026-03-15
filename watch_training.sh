#!/bin/bash
# QuantLLM 训练进度监控脚本
# 用法: bash /opt/quant-llm/watch_training.sh
# 守护模式: bash /opt/quant-llm/watch_training.sh --guard（自动重启）

# 自动检测当前活跃的训练日志和输出目录
if [ -f "/opt/quant-llm/output/train_v3.log" ] && pgrep -f "train.py" >/dev/null 2>&1; then
    LOG_FILE="/opt/quant-llm/output/train_v3.log"
    OUTPUT_DIR="/opt/quant-llm/output/quant-qwen2.5-14b-v3"
elif [ -f "/opt/quant-llm/output/train_r32_clean.log" ]; then
    LOG_FILE="/opt/quant-llm/output/train_r32_clean.log"
    OUTPUT_DIR="/opt/quant-llm/output/quant-qwen2.5-14b-lora-r32"
else
    LOG_FILE="/opt/quant-llm/output/training_log.txt"
    OUTPUT_DIR="/opt/quant-llm/output/quant-qwen2.5-14b-lora"
fi
PID_FILE="/opt/quant-llm/output/train.pid"

# 检查训练进程
train_pid=""
if [ -f "$PID_FILE" ]; then
    saved_pid=$(cat "$PID_FILE")
    if kill -0 "$saved_pid" 2>/dev/null; then
        train_pid="$saved_pid"
    fi
fi
if [ -z "$train_pid" ]; then
    train_pid=$(pgrep -f "python.*/opt/quant-llm/scripts/train.py" | head -1)
fi

# 从日志的 tqdm 进度条提取实时进度
# 格式: "  7%|▋         | 1307/19692 [2:06:59<27:20:42,  5.35s/it]"
parse_log_progress() {
    if [ ! -f "$LOG_FILE" ]; then
        return 1
    fi
    # 取最后一条主训练进度行（排除 eval 进度条，eval 的总数通常较小）
    # 主训练进度的 total 通常 > 5000
    local line
    line=$(grep -oP '\d+%\|[^|]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:]+,\s*[\d.]+s/it\]' "$LOG_FILE" | \
           while IFS= read -r l; do
               t=$(echo "$l" | grep -oP '\d+/\K\d+')
               [ "$t" -gt 5000 ] 2>/dev/null && echo "$l"
           done | tail -1)

    if [ -z "$line" ]; then
        # 兜底：取任何最后一条进度行
        line=$(grep -oP '\d+%\|[^|]*\|\s*\d+/\d+\s*\[[\d:]+<[\d:]+,\s*[\d.]+s/it\]' "$LOG_FILE" | tail -1)
    fi

    [ -z "$line" ] && return 1

    # 解析各字段
    current=$(echo "$line" | grep -oP '\|\s*\K\d+(?=/)')
    total=$(echo "$line" | grep -oP '\d+/\K\d+')
    elapsed=$(echo "$line" | grep -oP '\[\K[\d:]+(?=<)')
    remaining=$(echo "$line" | grep -oP '<\K[\d:]+')
    speed=$(echo "$line" | grep -oP '[\d.]+(?=s/it)')
    pct=$(echo "$line" | grep -oP '^\d+(?=%)')

    echo "$current $total $elapsed $remaining $speed $pct"
}

# 从 checkpoint 的 trainer_state.json 提取 loss/lr（如果存在）
parse_checkpoint() {
    local latest_ckpt
    latest_ckpt=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -z "$latest_ckpt" ] && return 1

    local state_file="${latest_ckpt}/trainer_state.json"
    [ ! -f "$state_file" ] && return 1

    python3 -c "
import json
d = json.load(open('$state_file'))
logs = [l for l in d['log_history'] if 'loss' in l]
loss = f\"{logs[-1]['loss']:.4f}\" if logs else 'N/A'
lr_logs = [l for l in d['log_history'] if 'learning_rate' in l]
lr = f\"{lr_logs[-1]['learning_rate']:.2e}\" if lr_logs else 'N/A'
print(f'{loss} {lr}')
" 2>/dev/null
}

echo "========== QuantLLM 训练进度 =========="

progress=$(parse_log_progress)

if [ -n "$progress" ]; then
    read -r current total elapsed remaining speed pct <<< "$progress"
    echo "  进度: ${current}/${total} (${pct}%)"
    echo "  已运行: ${elapsed}"
    echo "  预计剩余: ${remaining}"
    echo "  速度: ${speed}s/step"

    # 尝试从 checkpoint 获取 loss/lr
    ckpt_info=$(parse_checkpoint)
    if [ -n "$ckpt_info" ]; then
        read -r loss lr <<< "$ckpt_info"
        echo "  Loss: ${loss}"
        echo "  LR: ${lr}"
    fi
else
    # 回退：从 checkpoint 读取（旧模式）
    latest_ckpt=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$latest_ckpt" ]; then
        state_file="${latest_ckpt}/trainer_state.json"
        if [ -f "$state_file" ]; then
            info=$(python3 -c "
import json
d = json.load(open('$state_file'))
step = d['global_step']
mx = d['max_steps']
ep = d['epoch']
logs = [l for l in d['log_history'] if 'loss' in l]
loss = logs[-1]['loss'] if logs else 0
lr_logs = [l for l in d['log_history'] if 'learning_rate' in l]
lr = lr_logs[-1]['learning_rate'] if lr_logs else 0
print(f'{step} {mx} {ep:.2f} {loss:.4f} {lr:.2e}')
" 2>/dev/null)
            if [ -n "$info" ]; then
                read -r step mx ep loss lr <<< "$info"
                pct=$(python3 -c "print(f'{${step}/${mx}*100:.1f}')")
                echo "  进度: ${step}/${mx} (${pct}%)"
                echo "  Epoch: ${ep}"
                echo "  Loss: ${loss}"
                echo "  LR: ${lr}"
                echo "  (来源: checkpoint, 日志无 tqdm 输出)"
            fi
        fi
    else
        echo "  未找到训练进度信息"
    fi
fi

# 训练进程状态
if [ -n "$train_pid" ]; then
    # 计算进程运行时间
    start_time=$(ps -o lstart= -p "$train_pid" 2>/dev/null)
    echo "  训练进程: PID ${train_pid} (运行中)"
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$gpu_mem" ] && echo "  GPU 显存: ${gpu_mem} MB"
else
    if [ -f "$LOG_FILE" ] && grep -q "训练完成" "$LOG_FILE" 2>/dev/null; then
        echo "  训练进程: 未运行（训练已完成）"
    elif [ -n "$progress" ] && [ "$current" -lt "$total" ] 2>/dev/null; then
        echo "  训练进程: 未运行（异常中断！）"
        if [ "${1}" = "--guard" ]; then
            echo "  >>> 自动重启训练..."
            cd /opt/quant-llm
            source /opt/quant-llm/finetune-env/bin/activate
            export http_proxy=http://192.168.0.10:6152 https_proxy=http://192.168.0.10:6152 no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
            nohup bash run.sh train >> "$LOG_FILE" 2>&1 &
            echo "$!" > "$PID_FILE"
            echo "  >>> 已重启: PID $!"
        else
            echo "  提示: 用 --guard 可自动重启"
        fi
    else
        echo "  训练进程: 未运行"
    fi
fi
echo "======================================="
