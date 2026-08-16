# QuantLLM — A股量化多 Agent 交易系统

> **⚠️ 本项目仅供研究与教育用途，不构成投资建议。**

基于 Qwen3.8-27B 微调 + LangGraph 多 Agent 工作流的 A 股量化交易系统。参考 [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) 的多分析师协作架构，结合 A 股特有的反转效应（全技术因子 IC 为负）和市场环境感知，覆盖 A 股、商品期货、ETF 基金、可转债四大市场。

## 架构概览

```
                     ┌──────────────────────────────────┐
                     │         LangGraph 工作流           │
                     └──────────────┬───────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        │               │           │           │               │
  技术面分析师     基本面分析师   资金面分析师   情绪面分析师   估值分析师
 (28指标+反转)   (PE/PB/ROE)   (北向/主力)   (逆向投资)    (安全边际)
        │               │           │           │               │
        └───────────────┴───────────┼───────────┴───────────────┘
                                    │
                          ┌─────────▼─────────┐
                          │    市场环境分析     │
                          │  (牛/熊/震荡检测)   │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │     风控管理       │
                          │   (波动率+仓位)     │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼─────────┐
                          │     组合经理       │
                          │   (最终买卖决策)    │
                          └───────────────────┘
```

## 快速开始

### 多 Agent 交易决策（新）

```bash
# 单只股票全分析师
cd /opt/quant-llm
source finetune-env/bin/activate
python -m src.main --ticker 600519.SH

# 多只股票 + 显示推理过程
python -m src.main --ticker 600519.SH,000858.SZ,300750.SZ --show-reasoning

# 指定日期范围
python -m src.main --ticker 600519.SH --start-date 2025-06-01 --end-date 2025-12-31

# 使用特定分析师
python -m src.main --ticker 600519.SH --analysts technical_analyst,fundamentals_analyst

# 列出所有分析师
python -m src.main --list-analysts
```

### 模型训练流水线（原有）

```bash
# 一键执行全部流程（数据采集 → 转换 → 合并 → 训练）
bash /opt/quant-llm/run.sh

# 或分步执行
bash /opt/quant-llm/run.sh crawl      # 数据采集
bash /opt/quant-llm/run.sh convert    # 行情 → 训练问答对
bash /opt/quant-llm/run.sh predict    # 预测性训练数据
bash /opt/quant-llm/run.sh factors    # 构建个股因子
bash /opt/quant-llm/run.sh merge      # 合并训练集
bash /opt/quant-llm/run.sh train      # QLoRA 微调
bash /opt/quant-llm/run.sh export     # 导出 GGUF
bash /opt/quant-llm/run.sh eval       # 模型评估
bash /opt/quant-llm/run.sh backtest   # 回测验证
bash /opt/quant-llm/run.sh trade-live # 实盘交易
```

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

# 安装依赖
pip install -r requirements.txt
pip install langgraph langchain-core pydantic  # 多 Agent 工作流
```

如需使用数据增强功能，还需本地运行 ollama：

```bash
ollama pull qwen3.8:27b      # 数据增强模型（模型名由 .env 配置，参考 .env.example）
```

模型配置统一写在 `.env`（OLLAMA_URL / OLLAMA_GENERATION_MODEL / OLLAMA_LIVE_RANK_MODEL / OLLAMA_REASONING_MODEL），改模型只改这里。

## 项目结构

```
/opt/quant-llm/
├── src/                               # 多 Agent 交易系统（新）
│   ├── main.py                        #   CLI 入口
│   ├── agents/
│   │   ├── technicals.py              #   技术面分析师（28指标+反转策略）
│   │   ├── fundamentals.py            #   基本面分析师（PE/PB/ROE）
│   │   ├── fund_flow.py              #   资金面分析师（北向/主力）
│   │   ├── sentiment.py              #   情绪面分析师（逆向投资）
│   │   ├── valuation.py              #   估值分析师（安全边际）
│   │   ├── regime.py                 #   市场环境分析师（牛/熊/震荡）
│   │   ├── risk_manager.py           #   风控管理（波动率+仓位限制）
│   │   └── portfolio_manager.py      #   组合经理（最终决策）
│   ├── graph/
│   │   ├── state.py                  #   AgentState 状态定义
│   │   └── workflow.py              #   LangGraph 工作流编排
│   ├── tools/
│   │   └── api.py                   #   数据获取（本地缓存+akshare）
│   └── utils/
│       ├── analysts.py              #   分析师注册中心
│       └── llm.py                   #   LLM 调用（ollama/OpenAI兼容）
│
├── scripts/                           # 训练链路脚本（原有）
│   ├── _config.py                     #   配置加载器
│   ├── indicators.py                  #   28个技术指标
│   ├── crawl_ashare.py               #   A股历史行情爬取
│   ├── convert_all_to_training.py     #   行情→训练对
│   ├── generate_predictive_data.py    #   预测性训练数据
│   ├── build_stock_factors.py         #   个股因子构建
│   ├── merge_and_retrain.py           #   合并训练集
│   ├── train.py                       #   QLoRA微调
│   ├── evaluate.py                    #   模型评估
│   ├── backtest_signals.py            #   回测系统
│   ├── trade_live_qwen.py             #   实盘决策
│   └── trade_session_runner.py        #   交易调度器
│
├── run.sh                             # 一键执行脚本
├── config.yaml                        # 中心化配置
├── requirements.txt                   # Python 依赖
│
├── training-data/                     # 训练数据
│   ├── ashare/                        #   A股行情（~5000只）
│   ├── futures/                       #   商品期货
│   ├── etf/                           #   ETF基金
│   ├── cbond/                         #   可转债
│   ├── factors/                       #   因子数据
│   └── merged_train_v4.jsonl          #   最终训练集
│
├── output/                            # 模型输出
├── logs/                              # 日志
└── finetune-env/                      # Python 虚拟环境
```

## 多 Agent 系统详解

### 6 大分析师

| 分析师 | 核心能力 | A股特色 |
|--------|---------|---------|
| **技术面** | 28个指标，反转策略 | 基于 IC 因子分析：RSI(-0.40), 20d趋势(-0.46) |
| **基本面** | PE/PB/ROE 估值 | 分行业PE对比，破净检测 |
| **资金面** | 北向资金、主力流向 | 跟踪"聪明钱"动向 |
| **情绪面** | 恐慌贪婪、涨跌停 | 逆向投资：恐慌时贪婪 |
| **估值** | 多维估值，安全边际 | PE分位数+PB分位数+行业对比 |
| **市场环境** | 5维regime检测 | 牛(95%仓位)/熊(30%仓位)/震荡(50%仓位) |

### 风控体系

- **波动率调整**：年化波动率 <15%→仓位上限25%，>50%→仓位上限10%
- **单票上限**：10%
- **T+1 约束**：当日买入次日才能卖出
- **环境自适应**：牛市10只/震荡5只/熊市3只

### 工作流

```
start → [6个分析师并行] → 风控管理 → 组合经理 → 最终决策
```

所有分析师独立并行运行，信号汇总到风控层计算仓位上限，最终由组合经理做出 buy/sell/hold 决策。

## 训练系统详解

### 技术指标体系

`scripts/indicators.py` 提供 28 个技术指标，纯 pandas/numpy 实现：

| 类别 | 指标 |
|------|------|
| 原有 | RSI(14), MACD(12,26,9), MA(20), Volume MA(5) |
| 多周期均线 | MA(5/10/60/120), EMA(12/26) |
| 动量 | ROC(12), Williams %R(14), CCI(20) |
| 波动率 | ATR(14), Bollinger Bands(20,2), HV(20) |
| 量能 | OBV, MFI(14), VWAP近似 |
| 趋势 | ADX(14) |

### 核心策略：A股反转效应

基于 IC 因子分析实证：**A股所有主要技术因子 IC 均为负值**，趋势跟踪失效，反转/均值回归是主导效应。

| 因子组 | IC_IR | 策略含义 |
|--------|-------|---------|
| RSI | -0.40 | 超卖看多、超买看空 |
| 20d趋势 | -0.46 | 20日跌幅越大，反转概率越高 |
| HV波动率 | -0.42 | 高波动=风险，低波动=机会 |
| MA20偏离 | -0.38 | 偏离均值越远越可能回归 |

### 回测结果（2005-2025，反转策略 v4）

| 指标 | 个股策略 | 沪深300 |
|------|---------|---------|
| 总收益率 | **+1201%** | +122% |
| 年化收益率 | **12.99%** | 6.08% |
| 夏普比率 | 0.383 | 0.302 |
| 胜率 | 39.3% | — |
| 超额收益 | **+1079%** | — |

## 实盘交易

接入东方财富模拟盘，双层决策架构：

```
规则引擎初筛（Top50） → Qwen Skills 精排 → 风控层 → 东方财富API执行
```

- 执行时段：09:30-11:30, 13:00-15:00
- 调度频率：每5分钟一轮
- 交易成本：佣金万2.5 + 印花税千1(卖出) + 滑点万3

详细的交易架构、风控体系和执行细节见原有 [README（训练系统）](#模型训练流水线原有) 部分。

## 技术栈

- **LangGraph**: 多 Agent 工作流编排
- **基座模型**: Qwen3.8-27B (4bit 量化)
- **微调框架**: Unsloth + TRL SFTTrainer
- **数据源**: akshare (东方财富 API)、HuggingFace
- **技术指标**: 28 个自研指标（纯 pandas/numpy）
- **回测系统**: 走步验证，T+1 约束
- **LLM 推理**: ollama (本地) / OpenAI 兼容接口

## 免责声明

本项目仅供研究与教育用途。模型输出不构成任何投资建议，不保证盈利。量化交易存在市场风险，使用者需自行承担交易风险和损失。
