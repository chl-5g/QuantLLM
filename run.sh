#!/bin/bash
# 量化交易垂类大模型 — 自动化运行脚本
# 用法：
#   ./run.sh all        # 完整流程
#   ./run.sh crawl      # 仅爬取A股数据
#   ./run.sh convert    # 仅转换A股数据
#   ./run.sh generate   # 仅生成GitHub训练对
#   ./run.sh merge      # 仅合并数据集
#   ./run.sh train      # 仅训练

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/root/finetune-env"

# 激活虚拟环境
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
else
    echo "错误：虚拟环境 $VENV 不存在"
    exit 1
fi

step_crawl() {
    echo "========== 爬取A股数据 =========="
    python "$SCRIPT_DIR/scripts/crawl_ashare.py"
}

step_convert() {
    echo "========== 转换A股数据 =========="
    python "$SCRIPT_DIR/scripts/convert_ashare_to_training.py"
}

step_generate() {
    echo "========== 生成GitHub训练对 =========="
    python "$SCRIPT_DIR/scripts/generate_training_data.py"
}

step_merge() {
    echo "========== 合并数据集 =========="
    python "$SCRIPT_DIR/scripts/prepare_dataset.py"
}

step_train() {
    echo "========== 开始训练 =========="
    python "$SCRIPT_DIR/scripts/train.py"
}

case "${1:-help}" in
    crawl)    step_crawl ;;
    convert)  step_convert ;;
    generate) step_generate ;;
    merge)    step_merge ;;
    train)    step_train ;;
    all)
        step_crawl
        step_convert
        step_generate
        step_merge
        step_train
        ;;
    *)
        echo "用法: $0 {crawl|convert|generate|merge|train|all}"
        exit 1
        ;;
esac
