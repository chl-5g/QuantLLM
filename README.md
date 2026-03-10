# QuantLLM — 量化交易垂类大模型微调项目

> **本项目完全由 AI 生成。** 包括所有 Python 脚本、Shell 脚本、数据处理流程、训练配置及本文档，均由 Claude Opus 4.6 在人类指导下自主编写完成。

基于 Qwen2.5-14B 的量化交易领域 QLoRA 微调项目。覆盖 A股、商品期货、ETF基金、可转债四大市场。

## 快速开始

```bash
# 一键执行全部流程（数据采集 → 转换 → 合并 → 训练）
bash /tmp/quant-llm/run.sh

# 或分步执行
bash /tmp/quant-llm/run.sh crawl     # 仅数据采集
bash /tmp/quant-llm/run.sh convert   # 仅数据转换
bash /tmp/quant-llm/run.sh generate  # 数据增强（FinGPT+量化计算+推理链）
bash /tmp/quant-llm/run.sh merge     # 仅合并训练集
bash /tmp/quant-llm/run.sh train     # 仅训练
bash /tmp/quant-llm/run.sh export    # 导出 GGUF
bash /tmp/quant-llm/run.sh eval      # 模型评估
```

`run.sh` 会自动完成环境检查、依赖验证、GPU 显存释放、数据路径配置等工作，无需手动干预。

## 硬件要求

- GPU: NVIDIA RTX A5000 24GB（或同等显存以上）
- 内存: 64GB+
- 存储: 50GB 可用空间
- CUDA: 12.x

## 前置环境

```bash
# 创建并激活虚拟环境
python3 -m venv /tmp/quant-llm/finetune-env
source /tmp/quant-llm/finetune-env/bin/activate

# 安装依赖
pip install akshare pandas numpy unsloth trl transformers datasets torch
```

如需使用数据增强功能（`run.sh generate`），还需本地运行 [ollama](https://ollama.com/) 并拉取模型：

```bash
ollama pull qwen3:14b        # 量化计算种子扩展
ollama pull deepseek-r1:32b  # 推理链增强（可选，耗时较长）
```

## 项目结构

```
/tmp/quant-llm/
├── run.sh                             # 一键执行脚本（入口）
├── README.md                          # 本文件
├── LICENSE
├── .gitignore
│
├── scripts/                           # 所有 Python 脚本
│   ├── crawl_ashare.py                #   A股全量历史行情爬取
│   ├── crawl_multi_market.py          #   期货+ETF+可转债行情爬取
│   ├── generate_training_data.py      #   GitHub量化仓库 → 中文问答对（需ollama）
│   ├── convert_ashare_to_training.py  #   A股行情 → 技术分析训练对
│   ├── convert_all_to_training.py     #   全市场行情 → 训练对（A股+期货+ETF+转债）
│   ├── prepare_dataset.py            #   合并多源指令数据为统一ChatML JSONL
│   ├── fetch_fingpt_data.py          #   FinGPT A股预测数据 → ChatML
│   ├── generate_quant_calculations.py #  量化计算种子扩展（60→500条）
│   ├── add_reasoning_chains.py       #   推理链增强（deepseek-r1:32b）
│   ├── merge_and_retrain.py          #   合并所有数据源 → 最终训练集
│   ├── train.py                      #   QLoRA微调训练脚本（Unsloth）
│   ├── evaluate.py                   #   模型评估（ROUGE-L+结构化指标）
│   └── export_gguf.py                #   导出 GGUF 格式
│
├── training-data/                     # 所有训练数据（.gitignore 忽略）
│   ├── ashare/                        #   A股行情（~5000只，含技术指标）
│   ├── futures/                       #   商品期货（~80个主力合约）
│   ├── etf/                           #   ETF基金（~800只）
│   ├── cbond/                         #   可转债（~400只）
│   ├── github-repos/                  #   GitHub量化策略源码
│   ├── BAAI_IndustryInstruction_Finance-Economics/
│   ├── finance-instruct-500k/
│   ├── quant-trading-instruct/
│   ├── merged_train.jsonl             #   v1 指令数据 (~30k条)
│   ├── quant-github-generated.jsonl   #   GitHub策略问答 (~55条)
│   ├── all_market_train.jsonl         #   全市场行情训练对 (~10k条)
│   ├── fingpt_forecaster.jsonl        #   FinGPT A股预测（数百条）
│   ├── quant_calculations.jsonl       #   量化计算问答（~500条）
│   ├── reasoning_enhanced.jsonl       #   推理链增强（~2000条）
│   └── merged_train_v2.jsonl          #   最终训练集 (~40k+条)
│
├── output/                            # 模型输出
│   ├── quant-qwen2.5-14b-lora/        #   LoRA适配器权重
│   ├── gguf/                          #   GGUF 导出文件
│   └── eval_results.json              #   评估结果
│
├── finetune-env/                      # Python 虚拟环境
└── unsloth_compiled_cache/            # 编译缓存
```

## 流程详解

### Step 1: 数据采集 (`run.sh crawl`)

**A股历史行情** — `scripts/crawl_ashare.py`
- 数据源: 东方财富 API（通过 akshare，免费无需 API key）
- 范围: 主板+创业板+科创板，包含退市股（分层标注，用于风险预警训练）
- 分层标注: normal（正常）/ ST / delisting（退市），每条记录含 status 和 name 字段
- 输出: `training-data/ashare/basic/`（OHLCV）+ `training-data/ashare/advanced/`（含 RSI/MACD/MA）
- 支持断点续传
- 预计耗时: 2-3小时

**期货+ETF+可转债** — `scripts/crawl_multi_market.py`
- 商品期货: ~80个品种（螺纹钢、铁矿石、豆粕等主力连续合约）
- ETF基金: ~800+只（沪深300ETF、行业ETF、跨境ETF等）
- 可转债: ~400只活跃转债
- 支持断点续传
- 预计耗时: 30-60分钟

### Step 2: 数据转换 (`run.sh convert`)

`scripts/convert_all_to_training.py` 将四大市场行情转化为品种专属的技术分析问答对:

| 市场 | 上限 | 专属问答模板 |
|------|------|-------------|
| A股 | 5000条 | 技术面综合分析、趋势判断 |
| 商品期货 | 2000条 | 量价分析（增仓/减仓）、月度季节性统计 |
| ETF基金 | 2000条 | 波动率评估、配置价值分析 |
| 可转债 | 1500条 | 价格区间分析（折价/平价/偏股/高价）、双低策略 |

### Step 2.5: 数据增强 (`run.sh generate`)

| 数据源 | 条数 | 方法 |
|--------|------|------|
| FinGPT A股预测 | 数百条 | HuggingFace 下载 + Llama2→ChatML 转换 |
| 量化计算种子扩展 | ~500条 | 60条手写种子 + qwen3:14b few-shot 扩展 |
| 推理链增强 | ~2000条 | deepseek-r1:32b 为高质量记录添加 `<think>` 推理链 |

### Step 3: 合并训练集 (`run.sh merge`)

| 数据源 | 条数 | 内容 |
|--------|------|------|
| BAAI 中文金融 | ~29,000 | 金融知识、投资分析 |
| A股技术分析 | ≤5,000 | RSI/MACD/均线分析 |
| 期货量价分析 | ≤2,000 | CTA策略、季节性统计 |
| ETF配置分析 | ≤2,000 | 波动率、配置建议 |
| 可转债分析 | ≤1,500 | 双低策略、价格区间 |
| 英文量化指令 | 386 | 策略代码和回测 |
| GitHub策略问答 | 55 | 高质量代码解读 |
| FinGPT A股预测 | ~数百 | 沪深50股票趋势预测 |
| 量化计算 | ~500 | 风险指标/期权定价/组合优化 |
| 推理链增强 | ~2000 | 带 `<think>` 推理过程的高质量对 |
| **合计** | **~43,000+** | |

### Step 4: 模型训练 (`run.sh train`)

`run.sh train` 会自动释放 ollama 占用的 GPU 显存，然后启动训练。

训练参数:
| 参数 | 值 |
|------|-----|
| 基座模型 | unsloth/Qwen2.5-14B-bnb-4bit |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| RSLoRA | True（高rank时更稳定） |
| 目标模块 | q/k/v/o/gate/up/down_proj |
| 学习率 | 2e-4 |
| Batch size | 1（梯度累积8步，等效 batch_size=8） |
| Epoch | 1 |
| 精度 | bf16 |
| 优化器 | AdamW 8bit |
| 序列长度 | 2048 |
| 显存占用 | ~22-23GB |

训练输出: `output/quant-qwen2.5-14b-lora/`（LoRA 适配器权重）

### Step 5: 导出 GGUF (`run.sh export`)

`scripts/export_gguf.py` 将 LoRA checkpoint 合并到基座模型并导出 Q4_K_M 量化 GGUF 格式，可用 ollama 加载。

### Step 6: 模型评估 (`run.sh eval`)

`scripts/evaluate.py` 批量评估微调模型质量：
- 50 条手写测试题（覆盖技术分析、策略代码、量化计算、风控、可转债/ETF/期货）
- 从训练集随机抽 2% 作为 holdout 测试集
- 评估指标：ROUGE-L、平均回复长度、结构化输出率
- 支持 `--baseline` 参数与基座模型对比

### 验证

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/tmp/quant-llm/output/quant-qwen2.5-14b-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

inputs = tokenizer(
    "<|im_start|>system\n你是一个专业的量化交易专家。<|im_end|>\n"
    "<|im_start|>user\n请解释RSI指标的用法<|im_end|>\n"
    "<|im_start|>assistant\n",
    return_tensors="pt"
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 数据格式

所有训练数据统一为 **ChatML JSONL** 格式:

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "模型回答"}
  ]
}
```

## 技术栈

- **基座模型**: [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) (4bit 量化)
- **微调框架**: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) SFTTrainer
- **微调方法**: QLoRA (Quantized Low-Rank Adaptation)
- **数据源**: [akshare](https://github.com/akfamily/akshare) (东方财富 API)、HuggingFace 数据集
- **训练数据生成**: 模板化规则引擎 + 本地大模型辅助 ([ollama](https://ollama.com/) + qwen3:14b)
- **推理链增强**: deepseek-r1:32b 生成 `<think>` 推理过程

## Roadmap

### Phase 2: 模拟盘交易系统（计划中）

训练完成后，将模型接入模拟盘进行实盘验证。

**架构设计**：

```
市场数据 API ──→ 数据处理 ──→ LLM 分析决策 ──→ 交易执行 ──→ 日志记录
                                    ↑                    ↓
                               历史上下文 ←────── 持仓/P&L 状态
```

**核心模块**：

| 模块 | 功能 |
|------|------|
| 数据采集 | 定时拉取日线行情 + 计算技术指标，构造结构化 prompt |
| 决策引擎 | 调用微调模型，输出结构化 JSON 交易指令 |
| 交易执行 | 对接模拟盘 API，执行买入/卖出/持仓 |
| 风控层 | 独立于模型的硬编码规则（仓位上限、止损线、日亏损限制） |
| 日志系统 | 完整记录每笔交易的输入/推理/决策/结果 |

**模型输出格式**（约束为固定 JSON）：

```json
{
  "action": "buy",
  "symbol": "600519",
  "reason": "RSI从超卖区回升至35，MACD底背离，5日均线拐头向上...",
  "position_pct": 0.1,
  "stop_loss_pct": -3,
  "take_profit_pct": 8
}
```

**设计原则**：
- **日频决策**：匹配训练数据粒度（日线级技术分析），不做高频
- **风控独立**：模型只做建议，风控层有一票否决权
- **日志闭环**：交易日志可作为 DPO 训练数据（模型做对的强化，做错的作为负样本）
- **渐进验证**：先小仓位模拟 → 评估胜率/回撤 → 再考虑扩大
