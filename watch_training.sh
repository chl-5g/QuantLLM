#!/bin/bash
# QuantLLM 训练进度监控脚本
# 用法: bash ~/watch_training.sh
# 守护模式: bash ~/watch_training.sh --guard（cron 调用）

OUTPUT_DIR="/opt/quant-llm/output/quant-qwen2.5-14b-lora"
PID_FILE="/tmp/quant-llm/train.pid"

# 找最新的 checkpoint
latest_ckpt=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)

if [ -z "$latest_ckpt" ]; then
    echo "未找到 checkpoint"
    exit 1
fi

# 从 trainer_state.json 提取信息
state_file="${latest_ckpt}/trainer_state.json"
if [ ! -f "$state_file" ]; then
    echo "未找到 trainer_state.json"
    exit 1
fi

current_step=$(python3 -c "import json; d=json.load(open('$state_file')); print(d['global_step'])")
max_steps=$(python3 -c "import json; d=json.load(open('$state_file')); print(d['max_steps'])")
epoch=$(python3 -c "import json; d=json.load(open('$state_file')); print(f\"{d['epoch']:.2f}\")")
last_loss=$(python3 -c "import json; d=json.load(open('$state_file')); logs=[l for l in d['log_history'] if 'loss' in l]; print(f\"{logs[-1]['loss']:.4f}\" if logs else 'N/A')")
last_lr=$(python3 -c "import json; d=json.load(open('$state_file')); logs=[l for l in d['log_history'] if 'learning_rate' in l]; print(f\"{logs[-1]['learning_rate']:.2e}\" if logs else 'N/A')")
ckpt_time=$(stat -c %Y "$state_file")
now=$(date +%s)
ckpt_age=$(( (now - ckpt_time) / 60 ))
pct=$(python3 -c "print(f'{${current_step}/${max_steps}*100:.1f}')")

# 检查训练进程：优先用 PID 文件，精确匹配主进程
train_pid=""
if [ -f "$PID_FILE" ]; then
    saved_pid=$(cat "$PID_FILE")
    if kill -0 "$saved_pid" 2>/dev/null; then
        train_pid="$saved_pid"
    fi
fi
# 兜底：pgrep 只匹配主进程（排除 compile_worker 等子进程）
if [ -z "$train_pid" ]; then
    train_pid=$(pgrep -f "python.*/opt/quant-llm/scripts/train.py" | head -1)
fi

echo "========== QuantLLM 训练进度 =========="
echo "  Checkpoint: $(basename $latest_ckpt) (${ckpt_age}分钟前)"
echo "  进度: ${current_step}/${max_steps} (${pct}%)"
echo "  Epoch: ${epoch}"
echo "  Loss: ${last_loss}"
echo "  LR: ${last_lr}"
if [ -n "$train_pid" ]; then
    echo "  训练进程: PID ${train_pid} (运行中)"
    gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$gpu_mem" ] && echo "  GPU 显存: ${gpu_mem} MB"
else
    if [ -f "${OUTPUT_DIR}/training_log.txt" ]; then
        echo "  训练进程: 未运行（训练已完成）"
    elif [ "$current_step" -lt "$max_steps" ]; then
        echo "  训练进程: 未运行（异常中断！）"
        if [ "${1}" = "--guard" ]; then
            echo "  >>> 自动重启训练..."
            cd /tmp/quant-llm
            source /opt/quant-llm/finetune-env/bin/activate
            export http_proxy=http://192.168.0.10:6152 https_proxy=http://192.168.0.10:6152 no_proxy=localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16
            nohup python /opt/quant-llm/scripts/train.py >> /tmp/quant-llm/train.log 2>&1 &
            echo "$!" > "$PID_FILE"
            echo "  >>> 已重启: PID $!, 日志: /tmp/quant-llm/train.log"
        else
            echo "  提示: 用 bash ~/watch_training.sh --guard 可自动重启"
        fi
    else
        echo "  训练进程: 未运行"
    fi
fi
echo "======================================="
