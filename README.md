# QuantLLM — 量化交易 AI 助手

> **本项目完全由 AI 生成。** 包括所有 Python 脚本、Shell 脚本、数据处理流程、训练配置及本文档，均由 Claude Opus 4.6 在人类指导下自主编写完成。

**⚠️ 本项目仅供研究与教育用途，不构成投资建议。使用前请阅读末尾[免责声明](#免责声明)。**

基于 Qwen2.5-14B 的量化交易领域 QLoRA 微调项目。覆盖 A股、商品期货、ETF基金、可转债四大市场，支持个股评分预测和板块 ETF 轮动策略。

## 快速开始

```bash
# 一键执行全部流程（数据采集 → 转换 → 合并 → 训练）
bash /opt/quant-llm/run.sh

# 或分步执行
bash /opt/quant-llm/run.sh crawl      # 数据采集（A股+多市场）
bash /opt/quant-llm/run.sh recalc     # 重算技术指标（从basic重算，不爬取）
bash /opt/quant-llm/run.sh fund-flow  # 爬取资金流数据
bash /opt/quant-llm/run.sh convert    # 行情数据 → 训练问答对
bash /opt/quant-llm/run.sh predict    # 生成预测性训练数据（实际收益标签）
bash /opt/quant-llm/run.sh generate   # 数据增强（FinGPT+量化计算+推理链）
bash /opt/quant-llm/run.sh merge      # 合并所有数据源 → 最终训练集
bash /opt/quant-llm/run.sh train      # QLoRA 微调训练
bash /opt/quant-llm/run.sh export     # 导出 GGUF 格式
bash /opt/quant-llm/run.sh eval       # 模型评估
bash /opt/quant-llm/run.sh backtest   # 回测验证（对比沪深300）
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
python3 -m venv /opt/quant-llm/finetune-env
source /opt/quant-llm/finetune-env/bin/activate

# 安装全部依赖
pip install -r requirements.txt

# 可选：使用锁定版本（确保完全一致的环境）
# pip install -r requirements-lock.txt
```

如需使用数据增强功能（`run.sh generate`），还需本地运行 [ollama](https://ollama.com/) 并拉取模型：

```bash
ollama pull qwen3:14b        # 量化计算种子扩展
ollama pull deepseek-r1:32b  # 推理链增强（可选，耗时较长）
```

## 配置

所有配置集中在 `config.yaml`，脚本通过 `scripts/_config.py` 读取：

```python
from _config import cfg, MODEL_NAME, MAX_SEQ_LENGTH, DATA_DIR, OUTPUT_DIR
```

主要配置项：数据路径、模型参数、LoRA 配置、训练超参、ollama 地址、评估参数、板块 ETF 列表、回测参数等。详见 `config.yaml` 注释。

## 项目结构

```
/opt/quant-llm/
├── run.sh                             # 一键执行脚本（入口）
├── config.yaml                        # 中心化配置文件
├── requirements.txt                   # Python 依赖清单
├── requirements-lock.txt              # 精确版本锁定
├── README.md                          # 本文件
├── LICENSE
├── .gitignore
│
├── scripts/                           # 所有 Python 脚本
│   ├── _config.py                     #   配置加载器（所有脚本共用）
│   ├── indicators.py                  #   技术指标共享库（28个指标）
│   ├── crawl_ashare.py                #   A股全量历史行情爬取（--recalc 重算指标）
│   ├── crawl_multi_market.py          #   期货+ETF+可转债行情爬取（--recalc）
│   ├── crawl_fund_flow.py             #   板块/个股资金流数据爬取
│   ├── convert_all_to_training.py     #   全市场行情 → 训练对（含增强评分因子）
│   ├── generate_predictive_data.py    #   预测性训练数据（实际收益标签）
│   ├── fetch_fingpt_data.py           #   FinGPT A股预测数据 → ChatML
│   ├── generate_quant_calculations.py #   量化计算种子扩展（60→500条）
│   ├── add_reasoning_chains.py        #   推理链增强（deepseek-r1:32b）
│   ├── merge_and_retrain.py           #   合并所有数据源 → 最终训练集（v4）
│   ├── train.py                       #   QLoRA微调训练（early stopping+验证集）
│   ├── evaluate.py                    #   模型评估（ROUGE-L+结构化+对抗性测试）
│   ├── backtest_signals.py            #   回测系统（个股+ETF轮动 vs 沪深300）
│   ├── export_gguf.py                 #   导出 GGUF 格式
│   ├── rag_build_index.py             #   构建 RAG 检索索引
│   ├── rag_retrieve.py                #   RAG 检索引擎
│   ├── rag_serve.py                   #   RAG 增强推理服务
│   ├── eastmoney_login.py             #   东方财富模拟盘认证
│   ├── eastmoney_keepalive.py         #   模拟盘 session 保活守护
│   └── eastmoney_check.py             #   模拟盘凭据验证
│
├── training-data/                     # 所有训练数据（.gitignore 忽略）
│   ├── ashare/                        #   A股行情（~5000只，含28个技术指标）
│   ├── futures/                       #   商品期货（~80个主力合约）
│   ├── etf/                           #   ETF基金（~800只）
│   ├── cbond/                         #   可转债（~400只）
│   ├── fund_flow/                     #   资金流数据（板块/个股/大盘）
│   ├── predictive_signals.jsonl       #   预测性训练数据（实际收益标签）
│   ├── sector_rotation.jsonl          #   板块轮动训练数据
│   └── merged_train_v4.jsonl          #   最终训练集（v4，含预测数据）
│
├── output/                            # 模型输出
│   ├── quant-qwen2.5-14b-lora-r32/   #   LoRA适配器权重
│   ├── gguf/                          #   GGUF 导出文件
│   ├── rag_index.faiss                #   RAG 检索索引
│   ├── eval_results.json              #   评估结果
│   ├── backtest_results.json          #   回测结果
│   ├── backtest_equity.csv            #   回测权益曲线
│   └── backtest_trades.csv            #   回测交易记录
│
├── finetune-env/                      # Python 虚拟环境
└── unsloth_compiled_cache/            # 编译缓存
```

## 技术指标体系

`scripts/indicators.py` 提供 28 个技术指标，纯 pandas/numpy 实现，不依赖 ta-lib：

| 类别 | 指标 |
|------|------|
| 原有 | RSI(14), MACD(12,26,9), MA(20), Volume MA(5) |
| 多周期均线 | MA(5/10/60/120), EMA(12/26) |
| 动量 | ROC(12), Williams %R(14), CCI(20) |
| 波动率 | ATR(14), Bollinger Bands(20,2), 历史波动率HV(20) |
| 量能 | OBV, MFI(14), VWAP近似, 量变化率 |
| 趋势 | ADX(14) |
| 派生 | 均线排列(bullish/bearish/mixed), OBV趋势(rising/falling/flat) |

## 流程详解

### Step 1: 数据采集 (`run.sh crawl`)

**A股历史行情** — `scripts/crawl_ashare.py`
- 数据源: 东方财富 API（通过 akshare，免费无需 API key）
- 范围: 主板+创业板+科创板，包含退市股（分层标注）
- 输出: `training-data/ashare/basic/`（OHLCV）+ `training-data/ashare/advanced/`（含28个技术指标）
- 支持 `--recalc` 模式（从 basic 重算 advanced，无需联网）

**期货+ETF+可转债** — `scripts/crawl_multi_market.py`
- 商品期货 ~80个品种、ETF基金 ~800+只、可转债 ~400只
- 同样支持 `--recalc` 模式

**资金流数据** — `scripts/crawl_fund_flow.py`
- 板块资金流排名（行业/概念/地域）
- 个股资金流排名（今日/3日/5日/10日）
- 大盘资金流向（北向资金、主力）

### Step 2: 数据转换 (`run.sh convert`)

`scripts/convert_all_to_training.py` 将四大市场行情转化为品种专属的技术分析问答对。交易评分模板（0-100分制）融合以下因子：

| 因子 | 贡献 | 说明 |
|------|------|------|
| RSI(14) | ±20 | 超买/超卖判断 |
| MACD | ±17 | 柱线+金叉死叉 |
| MA位置 | ±8 | 价格vs20日均线 |
| CCI(20) | ±5 | 极值超买超卖 |
| ADX(14) | ±3 | 趋势强度放大/抑制 |
| MFI(14) | ±5 | 资金流确认 |
| BB+RSI | ±4 | 布林带与RSI联动 |
| HV(20) | ±3 | 波动率风险 |
| 均线排列 | ±3 | 多头/空头排列 |
| OBV趋势 | ±3 | 量价配合 |
| 市场环境 | ±10 | 牛/熊/震荡自适应 |
| 选股筛选 | -55~+15 | 换手率/股本/底部检测 |

### Step 2.5: 预测性训练数据 (`run.sh predict`)

**核心改进**：用实际未来收益做标签，而非公式评分。

**Type A — 个股收益预测**（~20000条）：
- 输入: 全部技术指标 + 市场环境
- 标签: 实际 5/10/20 日收益方向（strong_buy/buy/hold/sell/strong_sell）
- 采样范围: 2005-06-01 ~ 2025-06-30（覆盖A股现代史完整周期）

**Type B — 板块轮动预测**（~3000条）：
- 输入: 12个核心板块 ETF 指标对比
- 标签: 实际未来 N 日各板块收益排名
- 板块: 科技/消费/医药/金融/新能源/军工/半导体/证券/有色/地产/基建/传媒

### Step 3: 合并训练集 (`run.sh merge`)

| 数据源 | 预计条数 | 内容 |
|--------|---------|------|
| BAAI 全量中文金融 | ~40,300 | 金融知识底座 |
| 多市场行情分析 | ~8,900 | A股/期货/ETF/可转债技术分析 |
| 预测性训练数据 | ~20,000×2 | 个股收益预测（2x过采样） |
| 板块轮动预测 | ~3,000×2 | ETF板块轮动（2x过采样） |
| FinGPT 预测数据 | 1,230 | 道琼斯30股票趋势预测 |
| 推理链增强 | ~2,000 | 带 `<think>` 推理过程 |
| 量化计算 | ~500 | 风险指标/期权定价/组合优化 |
| **预计合计** | **~100,000** | |

### Step 4: 模型训练 (`run.sh train`)

训练参数（详见 `config.yaml`）:

| 参数 | 值 |
|------|-----|
| 基座模型 | unsloth/Qwen2.5-14B-bnb-4bit |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| RSLoRA | True |
| 目标模块 | q/k/v/o/gate/up/down_proj |
| 学习率 | 2e-4 |
| Batch size | 1（梯度累积8步，等效8） |
| Epoch | 3（配合 early stopping） |
| 精度 | bf16 |
| 序列长度 | 2048 |

### Step 5: 回测验证 (`run.sh backtest`)

训练完成后，必须通过回测验证策略有效性。

**两套策略**：
1. **个股策略**（日频）：评分筛选 → 买入Top N → 止损/止盈 → T+1约束
2. **板块ETF轮动**（周频）：板块评分 → 超配Top3 → 周度调仓

**回测规则**：
- 初始资金 10 万，佣金万2.5，印花税千1（卖出），滑点万3
- T+1 约束、单票10%仓位上限、总仓位80%上限
- 止损-5%/只，组合止损-30%
- 基准：沪深300 ETF (510300) 买入持有

**输出指标**：年化收益、夏普比率、最大回撤、胜率、盈亏比、Calmar比率、超额收益、信息比率

### RAG 检索增强

```
用户查询 → bge-large-zh-v1.5 编码 → FAISS 检索 top-3 → 注入 system prompt → ollama 推理
```

- FAISS IndexFlatIP，~200MB 索引
- 含 `[MARKET_DATA]` 的查询跳过 RAG（直接走模型评分）
- 配置见 `config.yaml` → `rag:` 段

## 模拟盘交易系统

训练+回测验证后，接入东方财富模拟盘进行实盘模拟。

**双模型协作架构**：

```
市场数据 → 技术指标计算 → 微调模型批量评分 → 筛选Top10 → Claude审核 → 风控 → 执行
                                                              ↑
                                              Claude Sonnet 宏观研判
```

**牛熊市环境感知**（5维度融合评分）：

| 维度 | 指标 | 牛市信号 | 熊市信号 |
|------|------|---------|---------|
| 趋势位置 | 价格 vs MA120 | 偏离 >5% | 偏离 <-5% |
| 趋势方向 | MA120 斜率（20日） | >+1% | <-1% |
| 量能确认 | 近20日均量 vs 前60日 | 放量 >1.3x | — |
| 波动率 | 近期 vs 长期标准差 | 低波动 <0.7x | 高波动 >1.8x |
| 价格动量 | 60日收益率 | >+15% | <-15% |

**选股筛选**（震荡/熊市强制生效）：
- 换手率 >= 3%（否则强制扣30分）
- 总股本 < 20亿股（否则强制扣25分）
- 底部信号检测（RSI<35 / MA偏离<-10% / 距52周低点<15% / MACD底背离）

## 技术栈

- **基座模型**: [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) (4bit 量化)
- **微调框架**: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) SFTTrainer
- **微调方法**: QLoRA (Quantized Low-Rank Adaptation)
- **数据源**: [akshare](https://github.com/akfamily/akshare) (东方财富 API)、HuggingFace 数据集
- **技术指标**: 28 个自研指标（纯 pandas/numpy，无 ta-lib 依赖）
- **训练数据**: 模板化规则引擎 + 预测性标签（实际收益） + 本地大模型辅助
- **回测系统**: 走步验证，T+1约束，对比沪深300基准
- **RAG**: FAISS + bge-large-zh-v1.5
- **模拟盘**: 东方财富 Playwright 自动化 + Claude API 审核

## 免责声明

本项目仅供研究与教育用途。模型输出不构成任何投资建议，不保证盈利。量化交易存在市场风险，使用者需自行承担交易风险和损失。请在充分理解相关风险后谨慎使用。
