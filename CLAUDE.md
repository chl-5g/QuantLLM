# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantLLM — 基于 Qwen2.5-14B 的 QLoRA 微调量化交易助手。目标是本地部署的 A股/期货/ETF/可转债量化分析模型。

- 基座模型：`unsloth/Qwen2.5-14B-bnb-4bit`
- 硬件：RTX A5000 24GB, CUDA 12.2
- 训练框架：Unsloth + TRL SFTTrainer + PEFT
- 项目目录：`/opt/quant-llm/`
- Python 环境：`/opt/quant-llm/finetune-env/`（venv, Python 3.12）

## Commands

所有操作通过 `run.sh` 入口执行，需先激活 venv：

```bash
source /opt/quant-llm/finetune-env/bin/activate

# 完整流水线
bash /opt/quant-llm/run.sh all

# 单步执行
bash /opt/quant-llm/run.sh crawl      # 爬取 A股+多市场数据
bash /opt/quant-llm/run.sh convert    # 市场数据 → ChatML 训练对
bash /opt/quant-llm/run.sh generate   # 数据增强（FinGPT+量化计算+推理链）
bash /opt/quant-llm/run.sh merge      # 合并所有数据源 → merged_train_v2.jsonl
bash /opt/quant-llm/run.sh train      # QLoRA 微调
bash /opt/quant-llm/run.sh eval       # 评估（65题手写+holdout）
bash /opt/quant-llm/run.sh export     # 导出 GGUF

# 直接运行单个脚本
python scripts/evaluate.py --consistency 3   # 带一致性检测的评估
python scripts/compare_evals.py              # 跨版本评估对比
```

## Architecture

### 数据流水线

```
crawl_ashare.py + crawl_multi_market.py    → training-data/{ashare,futures,etf,cbond}/*.jsonl
        ↓
convert_all_to_training.py                  → all_market_train.jsonl (~10k)
        ↓
fetch_fingpt_data.py                        → fingpt_forecaster.jsonl (1.2k)
generate_quant_calculations.py (ollama)     → quant_calculations.jsonl (~500)
add_reasoning_chains.py (ollama)            → reasoning_enhanced.jsonl (~2k)
        ↓
merge_and_retrain.py                        → merged_train_v2.jsonl (~39-43k)
        ↓
train.py                                    → output/quant-qwen2.5-14b-lora/
        ↓
evaluate.py                                 → output/eval_results_v{N}.json
export_gguf.py                              → output/gguf/
```

### 核心模块

- **`scripts/_config.py`** — 所有脚本的配置入口。加载 `config.yaml`，校验必要字段和取值范围，提供 `cfg` 字典、路径常量和 `call_ollama()` 辅助函数。所有脚本统一 `from _config import cfg`。
- **`config.yaml`** — 中心化配置：数据路径、LoRA 参数（r=32, rslora）、训练超参、ollama 连接、风控参数等。修改配置只改此文件。
- **`convert_all_to_training.py`** — 按市场类型生成领域特定问答模板（技术分析、交易信号 JSON、评分等），每个市场有独立的模板集。
- **`train.py`** — QLoRA 训练：4bit 量化加载 → LoRA 挂载（7个目标模块）→ ChatML 格式化 → 分层抽样 train/val split → SFTTrainer + early stopping + cosine annealing。
- **`evaluate.py`** — 三维评估：65 道手写题（含 15 道对抗性）+ holdout 集 + ROUGE-L/数值正确性/一致性指标。结果按版本存档。

### RAG 检索增强（规划中）

```
用户查询 → bge-large-zh-v1.5 编码 → FAISS 检索 top-3 → 注入 system prompt → ollama 推理
```

- **向量库**：FAISS (faiss-cpu)，IndexFlatIP，~200MB 索引
- **Embedding**：BAAI/bge-large-zh-v1.5（1024维，CPU 运行避免抢 GPU）
- **索引内容**：merged_train_v2.jsonl 的 50k Q+A 对（只编码 user question）
- **不索引**：原始行情 OHLCV（推理时以 `[MARKET_DATA]` 结构化注入）
- **检索策略**：top_k=3，score_threshold=0.35，含 `[MARKET_DATA]` 的查询跳过 RAG
- **Prompt 注入**：参考资料放在 system prompt 中的 `[参考资料]...[/参考资料]` 块
- **Token 预算**：系统 50 + 参考资料 460 + 问题 200 = 710，剩余 1338 给生成
- **新增文件**：`scripts/rag_build_index.py`、`scripts/rag_retrieve.py`、`scripts/rag_serve.py`
- **配置**：config.yaml 新增 `rag:` 段（enabled/embedding_model/top_k/score_threshold 等）
- **不引入**：LangChain、ChromaDB、GPU embedding、re-ranking

### 关键设计决策

- **推理链（`<think>` 标签）只用于知识解释和量化计算类问题**，不用于交易决策（基于 StockBench 研究结论）
- **交易信号统一为 JSON 输出格式**：`{action, symbol, reason, confidence, stop_loss}`
- **数据增强依赖本地 ollama**：qwen3:14b（种子扩展）+ deepseek-r1:32b（推理链）
- **训练数据带来源标记**（`source` 字段），支持分层抽样和来源分析

## Config Reference

训练关键参数在 `config.yaml` 中（不要硬编码）：

| 参数 | 值 | 说明 |
|------|-----|------|
| lora.r | 32 | LoRA 秩 |
| lora.use_rslora | true | Rank-Stabilized LoRA |
| training.learning_rate | 2e-4 | 学习率 |
| training.num_epochs | 3 | 训练轮数 |
| training.eval_ratio | 0.02 | 验证集比例（分层抽样） |
| training.early_stopping_patience | 3 | 早停耐心值 |
| model.max_seq_length | 2048 | 最大序列长度 |

## Working Conventions

- 每次启动 Claude Code 后，先探查模型训练情况（检查是否有正在运行的训练进程、最新 checkpoint、训练日志末尾），向用户汇报当前状态
- 每次执行完命令产生改动后，增量提交并推送到 GitHub
- 所有新脚本必须从 `_config.py` 读取配置，禁止硬编码路径或参数
- 训练数据目录 `training-data/` 和模型输出 `output/` 已在 `.gitignore` 中
- 系统提示词定义在 `config.yaml` 的 `model.system_prompt`，所有数据生成脚本共用
- 看训练进度 → 执行 `bash /opt/quant-llm/watch_training.sh`
- 用户说"写入记忆" → 同时更新 MEMORY.md 和本文件（`/opt/quant-llm/CLAUDE.md`）
