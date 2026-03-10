#!/bin/bash
# ============================================================
# QuantLLM 一键执行脚本
# 用法: bash /opt/quant-llm/run.sh [step]
#
# 不带参数：执行全部流程
# 带参数：执行指定步骤
#   crawl      — 仅数据采集（A股+多市场）
#   convert    — 仅数据转换
#   generate   — 数据增强（FinGPT+量化计算+推理链）
#   merge      — 仅合并训练集
#   train      — 仅训练
#   export     — 导出 GGUF
#   eval       — 模型评估
#   all        — 全部流程
# ============================================================

set -e

PROJECT_DIR="/opt/quant-llm"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
DATA_DIR="$PROJECT_DIR/training-data"
VENV="$PROJECT_DIR/finetune-env/bin/activate"
OLLAMA_URL="http://localhost:11434"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $*"; exit 1; }

cd "$PROJECT_DIR"

# ============================================================
# 环境检查
# ============================================================
check_env() {
    log "环境检查..."

    if [ ! -f "$VENV" ]; then
        err "微调虚拟环境不存在: $VENV"
    fi
    source "$VENV"

    python3 -c "import akshare" 2>/dev/null || err "akshare 未安装，运行: pip install akshare"
    python3 -c "import pandas" 2>/dev/null  || err "pandas 未安装"
    python3 -c "import numpy" 2>/dev/null   || err "numpy 未安装"

    log "环境检查通过"
}

# ============================================================
# Step 1: 数据采集
# ============================================================
step_crawl() {
    log "========== Step 1: 数据采集 =========="

    log "[1/2] A股全量历史行情..."
    python3 "$SCRIPTS_DIR/crawl_ashare.py"
    log "A股采集完成"

    log "[2/2] 期货 + ETF + 可转债..."
    python3 "$SCRIPTS_DIR/crawl_multi_market.py"
    log "多市场采集完成"
}

# ============================================================
# Step 2: 数据转换
# ============================================================
step_convert() {
    log "========== Step 2: 数据转换 =========="

    # 检查是否有行情数据
    ashare_count=$(ls "$DATA_DIR/ashare/advanced/" 2>/dev/null | wc -l)
    multi_count=$(ls "$DATA_DIR/futures/advanced/" "$DATA_DIR/etf/advanced/" "$DATA_DIR/cbond/advanced/" 2>/dev/null | wc -l)

    if [ "$ashare_count" -eq 0 ] && [ "$multi_count" -eq 0 ]; then
        err "没有行情数据，请先执行: bash run.sh crawl"
    fi

    log "全市场行情 → 训练数据..."
    python3 "$SCRIPTS_DIR/convert_all_to_training.py"
    log "转换完成"
}

# ============================================================
# Step 2.5: 数据增强（FinGPT + 量化计算 + 推理链）
# ============================================================
step_generate() {
    log "========== Step 2.5: 数据增强 =========="

    # FinGPT（仅需网络）
    log "[1/3] FinGPT A股预测数据..."
    python3 "$SCRIPTS_DIR/fetch_fingpt_data.py" || warn "FinGPT 数据下载失败，跳过"

    # 量化计算种子扩展（需 ollama）
    if curl -s "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        log "[2/3] 量化计算种子扩展 (qwen3:14b)..."
        python3 "$SCRIPTS_DIR/generate_quant_calculations.py" || warn "量化计算扩展失败，跳过"

        log "[3/3] 推理链增强 (deepseek-r1:32b)..."
        python3 "$SCRIPTS_DIR/add_reasoning_chains.py" || warn "推理链增强失败，跳过"
    else
        warn "ollama 不可用，跳过量化计算和推理链增强"
    fi

    log "数据增强完成"
}

# ============================================================
# Step 3: 合并训练集
# ============================================================
step_merge() {
    log "========== Step 3: 合并训练集 =========="

    # v1 指令数据
    if [ ! -f "$DATA_DIR/merged_train.jsonl" ]; then
        log "生成 v1 指令数据..."
        python3 "$SCRIPTS_DIR/prepare_dataset.py"
    else
        v1_count=$(wc -l < "$DATA_DIR/merged_train.jsonl")
        log "v1 指令数据已存在 (${v1_count} 条)"
    fi

    # 合并 v1 + 行情数据 → v2
    log "合并最终训练集..."
    python3 "$SCRIPTS_DIR/merge_and_retrain.py"

    v2_count=$(wc -l < "$DATA_DIR/merged_train_v2.jsonl")
    log "最终训练集: merged_train_v2.jsonl (${v2_count} 条)"
}

# ============================================================
# Step 4: 模型训练
# ============================================================
step_train() {
    log "========== Step 4: QLoRA 训练 =========="

    if [ ! -f "$DATA_DIR/merged_train_v2.jsonl" ]; then
        err "训练数据不存在，请先执行: bash run.sh merge"
    fi

    # 检查 GPU
    if ! nvidia-smi &>/dev/null; then
        err "未检测到 GPU"
    fi

    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$gpu_mem" -gt 5000 ]; then
        warn "GPU 显存已占用 ${gpu_mem}MB，尝试释放 ollama 模型..."
        curl -s "$OLLAMA_URL/api/generate" -d '{"model":"qwen3:14b","keep_alive":0}' >/dev/null 2>&1 || true
        curl -s "$OLLAMA_URL/api/generate" -d '{"model":"deepseek-r1:32b","keep_alive":0}' >/dev/null 2>&1 || true
        sleep 5

        gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "$gpu_mem" -gt 5000 ]; then
            err "GPU 显存仍占用 ${gpu_mem}MB，无法启动训练。请手动释放显存。"
        fi
        log "GPU 显存已释放"
    fi

    log "开始训练..."
    python3 "$SCRIPTS_DIR/train.py"

    log "训练完成！模型保存在: output/quant-qwen2.5-14b-lora/"
}

# ============================================================
# Step 5: 导出 GGUF
# ============================================================
step_export() {
    log "========== Step 5: 导出 GGUF =========="
    python3 "$SCRIPTS_DIR/export_gguf.py"
    log "导出完成"
}

# ============================================================
# Step 6: 模型评估
# ============================================================
step_eval() {
    log "========== Step 6: 模型评估 =========="

    if [ ! -d "$PROJECT_DIR/output/quant-qwen2.5-14b-lora" ]; then
        err "模型目录不存在，请先训练"
    fi

    # 释放 ollama 显存
    curl -s "$OLLAMA_URL/api/generate" -d '{"model":"qwen3:14b","keep_alive":0}' >/dev/null 2>&1 || true
    curl -s "$OLLAMA_URL/api/generate" -d '{"model":"deepseek-r1:32b","keep_alive":0}' >/dev/null 2>&1 || true
    sleep 3

    python3 "$SCRIPTS_DIR/evaluate.py"
    log "评估完成"
}

# ============================================================
# 主流程
# ============================================================

STEP="${1:-all}"

echo ""
echo "============================================================"
echo " QuantLLM — 量化交易 AI 助手"
echo " 执行步骤: $STEP"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

check_env

case "$STEP" in
    crawl)
        step_crawl
        ;;
    convert)
        step_convert
        ;;
    generate)
        step_generate
        ;;
    merge)
        step_merge
        ;;
    train)
        step_train
        ;;
    export)
        step_export
        ;;
    eval)
        step_eval
        ;;
    all)
        step_crawl
        step_convert
        step_generate
        step_merge
        step_train
        ;;
    *)
        echo "用法: bash run.sh [crawl|convert|generate|merge|train|export|eval|all]"
        echo ""
        echo "  crawl      数据采集（A股+期货+ETF+可转债）"
        echo "  convert    行情数据 → 训练问答对"
        echo "  generate   数据增强（FinGPT+量化计算+推理链）"
        echo "  merge      合并所有数据源 → 最终训练集"
        echo "  train      QLoRA 微调训练"
        echo "  export     导出 GGUF 格式"
        echo "  eval       模型评估"
        echo "  all        执行全部流程（默认）"
        exit 1
        ;;
esac

echo ""
log "========== 完成 =========="
