#!/bin/bash
# QuantLLM 训练进度监控脚本
# 用法: bash ~/watch_training.sh

OUTPUT_DIR="/tmp/quant-llm/output/quant-qwen2.5-14b-lora"

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

# 检查训练进程
train_pid=$(pgrep -f "python.*train.py" | head -1)

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
            nohup python /opt/quant-llm/scripts/train.py > /tmp/quant-llm/train.log 2>&1 &
            echo "  >>> 已重启: PID $!, 日志: /tmp/quant-llm/train.log"
        else
            echo "  提示: 用 bash ~/watch_training.sh --guard 可自动重启"
        fi
    else
        echo "  训练进程: 未运行"
    fi
fi
echo "======================================="
