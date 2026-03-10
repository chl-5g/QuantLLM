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
bash /tmp/quant-llm/run.sh merge     # 仅合并训练集
bash /tmp/quant-llm/run.sh train     # 仅训练
bash /tmp/quant-llm/run.sh export    # 导出 GGUF
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

如需使用 `generate_training_data.py`（从 GitHub 仓库生成问答对），还需本地运行 [ollama](https://ollama.com/) 并拉取模型：

```bash
ollama pull qwen3:14b
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
│   ├── merge_and_retrain.py          #   合并指令数据+行情数据 → 最终训练集
│   ├── train.py                      #   QLoRA微调训练脚本（Unsloth）
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
│   └── merged_train_v2.jsonl          #   v2 最终训练集 (~40k条)
│
├── output/                            # 模型输出
│   ├── quant-qwen2.5-14b-lora/        #   LoRA适配器权重
│   └── gguf/                          #   GGUF 导出文件
│
├── finetune-env/                      # Python 虚拟环境
└── unsloth_compiled_cache/            # 编译缓存
```

## 流程详解

### Step 1: 数据采集 (`run.sh crawl`)

**A股历史行情** — `scripts/crawl_ashare.py`
- 数据源: 东方财富 API（通过 akshare，免费无需 API key）
- 范围: 主板+创业板+科创板，排除退市股（约5000+只）
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
| **合计** | **~40,000** | |

### Step 4: 模型训练 (`run.sh train`)

`run.sh train` 会自动释放 ollama 占用的 GPU 显存，然后启动训练。

训练参数:
| 参数 | 值 |
|------|-----|
| 基座模型 | unsloth/Qwen2.5-14B-bnb-4bit |
| LoRA rank | 16 |
| LoRA alpha | 16 |
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
