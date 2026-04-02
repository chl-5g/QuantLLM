# QuantLLM — 量化交易 AI 助手

> **⚠️ 本项目仅供研究与教育用途，不构成投资建议。使用前请阅读末尾[免责声明](#免责声明)。**

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
bash /opt/quant-llm/run.sh trade-live # 双层实盘决策与交易日志
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
├── scripts/                           # 生产链路 Python 脚本
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
│   ├── qwen_skills.py                 #   Qwen Skills（结构化精排，JSON 约束）
│   ├── eastmoney_http_api.py          #   东方财富模拟盘 HTTP API 客户端
│   ├── trade_execution.py             #   执行核心（API直连、风控、幂等、限频、两阶段执行）
│   ├── trade_live_qwen.py             #   双层实盘入口（规则初筛 -> Qwen 精排 -> 执行）
│   └── trade_session_runner.py        #   交易时段自动调度器（含竞价 dry-run 策略）
│
├── archive/                           # 历史与工具归档（不参与生产链路）
│   ├── legacy/                        #   旧版备份脚本/配置（.bak 等）
│   └── tools/
│       └── scripts/                   #   一次性/测试脚本，按用途分组
│           ├── data/                  #   数据清洗与 trapboost 工具
│           ├── eval/                  #   评估对比与对抗测试工具
│           ├── rag/                   #   RAG 附加处理工具
│           └── test/                  #   独立测试脚本
│
├── logs/                              # 交易与策略日志（按 YYYY-MM-DD 分目录）
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
│   ├── quant-qwen2.5-14b-v3/         #   V3 LoRA适配器权重
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

`scripts/convert_all_to_training.py` 将四大市场行情转化为品种专属的技术分析问答对。交易评分系统基于 IC 分析验证的**反转策略 v2**（A 股所有技术因子 IC 为负，反转效应主导）：

| 评分组 | 上限 | 核心因子（IC_IR） | 逻辑 |
|--------|------|-------------------|------|
| 动量反转 | ±20 | RSI(-0.40), 20d趋势(-0.46), ROC(-0.39) | 超买看空、超卖看多 |
| 趋势确认 | ±12 | MA20偏离(-0.38), CCI(-0.30), MFI(-0.31) | 偏离均值回归 |
| 波动率 | ±10 | HV(-0.42), BB位置(-0.35) | 高波动=风险 |
| 量能 | ±8 | 量比(-0.21), 5d趋势(-0.18) | 放量见顶 |
| 困境反转 | ±15 | 价格分位数、量能萎缩度、底部反弹 | 极度超跌→反转 |
| 市场环境 | ±8 | 沪深300 MA120 regime | 牛/熊/震荡自适应 |
| 趋势过滤 | -8 | 个股 MA120 | 下跌趋势扣分 |
| 选股筛选 | -35 | 换手率/股本/底部检测 | 低流动性扣分 |

### Step 2.5: 预测性训练数据 (`run.sh predict`)

**核心改进**：用实际未来收益做标签，而非公式评分。

**Type A — 个股收益预测**（32,705条，2x过采样至65,410条）：
- 输入: 全部技术指标 + 市场环境
- 标签: 实际 5/10/20 日收益方向（strong_buy/buy/hold/sell/strong_sell）
- 采样范围: 2005-06-01 ~ 2025-06-30（覆盖A股现代史完整周期）
- 每只股票采样8个日期，覆盖牛/熊/震荡多种行情

**Type B — 板块轮动预测**（186条，2x过采样至372条）：
- 输入: 12个核心板块 ETF 指标对比
- 标签: 实际未来 N 日各板块收益排名
- 板块: 科技/消费/医药/金融/新能源/军工/半导体/证券/有色/地产/基建/传媒

### Step 3: 合并训练集 (`run.sh merge`)

| 数据源 | 预计条数 | 内容 |
|--------|---------|------|
| BAAI 全量中文金融 | 40,290 | 金融知识底座 |
| 预测性训练数据 | 65,410 | 个股收益预测（32k×2x过采样） |
| 多市场行情分析 | 8,855 | A股/期货/ETF/可转债技术分析 |
| 推理链增强 | 1,820 | 带 `<think>` 推理过程 |
| FinGPT 预测数据 | 1,230 | 道琼斯30股票趋势预测 |
| R1 金融推理 | 1,000 | DeepSeek-R1 金融推理数据 |
| 板块轮动预测 | 372 | ETF板块轮动（186×2x过采样） |
| 量化计算 | 350 | 风险指标/期权定价/组合优化 |
| GitHub 量化 | 38 | 量化策略代码问答 |
| **合计** | **119,365** | 超长样本已过滤（>8192字符） |

### Step 4: 模型训练 (`run.sh train`)

训练参数（详见 `config.yaml`）:

| 参数 | 值 |
|------|-----|
| 基座模型 | unsloth/Qwen2.5-14B-bnb-4bit |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| RSLoRA | True |
| 目标模块 | q/k/v/o/gate/up/down_proj |
| 学习率 | 1e-4（cosine decay） |
| Batch size | 1（梯度累积8步，等效8） |
| Epoch | 3（配合 early stopping） |
| 精度 | bf16 |
| 序列长度 | 2048 |
| 梯度裁剪 | max_grad_norm=1.0 |
| 权重衰减 | 0.01 |
| Checkpoint | 每2000步保存，保留最近3个 |

### Step 5: 回测验证 (`run.sh backtest`)

训练完成后，必须通过回测验证策略有效性。

**两套策略**：
1. **个股策略**（日频）：评分筛选 → 买入Top N → 止损/止盈 → T+1约束
2. **板块ETF轮动**（周频）：板块评分 → 超配Top3 → 周度调仓

**回测规则**：
- 初始资金 10 万，佣金万2.5，印花税千1（卖出），滑点万3
- T+1 约束、单票10%仓位上限
- 止损-20%/只（反转策略需宽容忍度），持仓5天保护期
- 组合止损-30%（触发后60天冷却期，非永久停止）
- 熊市自动减仓（仓位上限降至25%），震荡市66%
- 基准：沪深300 ETF (510300) 买入持有
- 市场环境判定：沪深300 MA120 多维度 regime 检测

**回测结果（2005-2025，反转策略 v4）**：

| 指标 | 个股策略 | 沪深300 |
|------|---------|---------|
| 总收益率 | **+1201%** | +122% |
| 年化收益率 | **12.99%** | 6.08% |
| 夏普比率 | 0.383 | 0.302 |
| 胜率 | 39.3% | — |
| 盈亏比 | 1.122 | — |
| 超额收益 | **+1079%** | — |

**输出指标**：年化收益、夏普比率、最大回撤、胜率、盈亏比、Calmar比率、超额收益、信息比率

### Step 6: RAG 检索增强

```
用户查询 → bge-large-zh-v1.5 编码 → FAISS 检索 top-3 → 注入 system prompt → ollama 推理
```

- FAISS IndexFlatIP，~200MB 索引
- 含 `[MARKET_DATA]` 的查询跳过 RAG（直接走模型评分）
- 配置见 `config.yaml` → `rag:` 段

## 模拟盘交易系统

训练+回测验证后，已接入东方财富模拟盘执行链路。

### 交易策略架构

```
                        ┌─────────────────────────┐
                        │   沪深300 MA120 Regime   │
                        │  5维度融合 → 牛/熊/震荡   │
                        └────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  第1层: 规则引擎（秒级）       │
                    │  compute_score 初筛 → Top 50  │
                    └────────────┬────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  第2层: Qwen Skills（分钟级）  │
                    │  StockRankSkill 精排 Top N    │
                    │  输出: rank/action/reason     │
                    └────────────┬────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │         执行计划构建                   │
              │  score驱动仓位 + 环境自适应持仓上限      │
              └──────────────────┬──────────────────┘
                                 │
         ┌───────────────────────▼───────────────────────┐
         │              trade_execution 风控层             │
         │  单票上限 | 总仓位上限 | 日内限频 | 幂等去重      │
         │  kill switch | 失败记录 | 两阶段执行(先卖后买)    │
         └───────────────────────┬───────────────────────┘
                                 │
              ┌──────────────────▼──────────────────┐
              │   东方财富 HTTP API 直连执行           │
              │   实时行情(双源容灾) → 下单 → 回执核验  │
              └─────────────────────────────────────┘
```

### 核心策略：A股反转效应

本项目策略基于 IC 因子分析的实证发现：**A股所有主要技术因子 IC 均为负值**，即趋势跟踪在 A 股失效，反转/均值回归才是主导效应。

| 因子组 | 代表因子 | IC_IR | 策略含义 |
|--------|---------|-------|---------|
| 动量反转 | RSI(-0.40), 20d趋势(-0.46) | -0.40~-0.46 | 超买看空、超卖看多 |
| 波动率 | HV(-0.42), BB位置(-0.35) | -0.35~-0.42 | 高波动=风险，低波动=机会 |
| 趋势确认 | MA20偏离(-0.38), CCI(-0.30) | -0.30~-0.38 | 偏离越大越可能回归 |
| 量能 | 量比(-0.21), 5d趋势(-0.18) | -0.18~-0.21 | 放量见顶，缩量筑底 |

**选股逻辑**：寻找超卖（RSI低）、偏离均值（MA偏离大）、缩量（量能萎缩）的股票，等待均值回归。

### 市场环境感知（5维度融合评分）

基于沪深300指数的多维度 regime 检测，决定仓位水平和选股门槛：

| 维度 | 指标 | 牛市信号(+1) | 熊市信号(-1) |
|------|------|-------------|-------------|
| 趋势位置 | 价格 vs MA120 | 偏离 >5% | 偏离 <-5% |
| 趋势方向 | MA120 斜率（20日） | >+1% | <-1% |
| 量能确认 | 近20日均量 vs 前60日 | 放量 >1.3x | — |
| 波动率 | 近期 vs 长期标准差 | 低波动 <0.7x | 高波动 >1.8x |
| 价格动量 | 60日收益率 | >+15% | <-15% |

**Regime 分数范围**: -5 ~ +5，阈值判定：`≥2 牛市` / `≤-2 熊市` / `其余震荡`

**环境自适应持仓策略**：

| 环境 | 目标总仓位 | 最大持仓数 | 选股门槛 |
|------|-----------|-----------|---------|
| 牛市 | 95% | 10只 | buy + strong_buy |
| 震荡 | 50% | 5只 | buy + strong_buy |
| 熊市 | 30% | 3只 | strong_buy only（已实现） |

### 仓位管理

- **Score 驱动仓位分配**：按 `score^power` 归一化权重分配目标仓位，高分股拿更大仓位
- **单票上限**: 10%（`risk_control.max_position_pct`）
- **总仓位上限**: 80%（被 regime 进一步约束）
- **最小调仓阈值**: 1%（避免频繁微调）

### 风控体系

**事前风控**（下单前）：
- 单票仓位上限 10%，总仓位上限 80%
- 日内交易笔数上限 20
- 幂等去重（同一标的同方向同日不重复下单，含失败记录）
- Kill switch 文件存在即阻断所有交易

**事中风控**（执行时）：
- 实时行情双源容灾（push2 主源 + push2delay 备用），取价失败跳过不盲报
- 两阶段执行：先卖出释放资金，再买入
- 当日未完成委托去重（不在同一标的同方向堆积委托）
- 账户强制绑定（`config.yaml` 指定 zjzh，防串单）

**事后风控**：
- 全量交易明细落盘 `trade_records.jsonl`
- 每日策略日志+执行日志归档 `logs/YYYY-MM-DD/`
- 交易日志积累3个月后做 DPO 偏好优化

### 交易执行细节

| 项目 | 配置 |
|------|------|
| 执行方式 | 东方财富模拟盘 HTTP API 直连（Playwright 已停用） |
| 调度频率 | 每5分钟一轮（`trade_session_runner.py`） |
| 预热时段 | 09:15-09:25 仅 dry-run |
| 执行时段 | 09:30-11:30, 13:00-15:00 |
| 交易成本 | 佣金万2.5 + 印花税千1(卖出) + 滑点万3 |

### 回测结果（2005-2025，反转策略 v4-trapboost）

| 指标 | 个股策略 | 沪深300 |
|------|---------|---------|
| 总收益率 | **+1201%** | +122% |
| 年化收益率 | **12.99%** | 6.08% |
| 夏普比率 | 0.383 | 0.302 |
| 胜率 | 39.3% | — |
| 盈亏比 | 1.122 | — |
| 超额收益 | **+1079%** | — |

## 当前进展（2026-04-02）

- v4-trapboost 模型已训练完成，门禁评估通过（holdout + 核心题 + 对抗题 + 评分方向 6/6）
- 执行层已从 Playwright 切换为 HTTP API 直连
- 交易调度器 `trade_session_runner.py` 已作为守护进程运行
- 修复 `_extract_json` JSON 解析（Qwen Skills 不再 fallback 到规则引擎）
- 修复幂等去重（failed 订单也记录，防止反复重试）
- 修复 `sanitize_plan` 支持 sell/strong_sell action

### 已完成里程碑（原 TODO 完成项汇总）

- P0：修复 Qwen fallback、幂等去重、sell action 吞单；完成两阶段执行链路与 Qwen 精排效果验证
- P1：完成信号层/执行层分离、Regime 连续化、非对称买卖、熊市 strong_buy 限制、止盈止损优化、每周 3-10 优质股清单、基本面/北向因子、涨跌停处理、T+1 强化
- P2：完成 GGUF 导出、DPO 偏好对生成、日报生成、监控告警、策略-持仓对账、回测与实盘收益对比
- P3（除实盘切换）：完成资金管理（2A 提示止盈、70%A 暂停）、合规审计与异常检测、波动率目标化仓位调节

## TODO — 策略优化路线

> 每周维护更新，上次更新：2026-04-02

### P0 — 验证与修复（当前阶段，仅未完成）

- [ ] **模拟盘实跑验证** — 目标：跑赢沪深300至少3个月（从2026-04-03开始计）
### P3 — 实盘准备（仅未完成）

- [ ] **实盘切换** — 模拟盘跑赢沪深300三个月后，换支持 QMT 的券商（国金/华鑫）

### P4 — 工程化与稳定性（新增）

- [ ] **接入 P2/P3 定时任务编排** — 将日报/监控/对账/回测实盘对比纳入统一调度（cron/systemd），不再依赖手工触发
- [ ] **run.sh 增补新命令入口** — 增加 dpo-build / daily-report / monitor / reconcile / compare 子命令，统一运维入口
- [ ] **因子数据生产链补齐** — 建立 `stock_factors_latest.json` 的采集、清洗、落盘、质量校验流程，确保基本面/北向因子持续生效
- [ ] **监控告警通道实装** — webhook 告警补齐分级、重试、静默窗口和去重，减少噪音并避免漏报
- [ ] **手动覆盖机制加 schema 校验** — 对 `trade_live.OVERRIDE.json` 做 JSON Schema 校验与启动前检查，防误配
- [ ] **审计/异常日志轮转归档** — 为 `audit_events.jsonl`/`anomaly_events.jsonl`/`trade_records.jsonl` 增加留存周期与压缩归档
- [ ] **回测-实盘对比升级为滚动看板** — 增加 7/30/90 天滚动收益、回撤、胜率、超额收益指标
- [ ] **新增最小测试集 + CI** — 建立关键路径单元测试并接入 GitHub Actions（计划生成、幂等、风控阻断、override）
- [ ] **运维脚本去硬编码网络配置** — 移除 `watch_training.sh` 等脚本中的硬编码代理地址，改为配置/环境变量
- [ ] **交易凭据安全治理** — 明确 Cookie 凭据过期检测、自动刷新、权限最小化和审计策略

## 技术栈

- **基座模型**: [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B) (4bit 量化)
- **微调框架**: [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) SFTTrainer
- **微调方法**: QLoRA (Quantized Low-Rank Adaptation)
- **数据源**: [akshare](https://github.com/akfamily/akshare) (东方财富 API)、HuggingFace 数据集
- **技术指标**: 28 个自研指标（纯 pandas/numpy，无 ta-lib 依赖）
- **训练数据**: 模板化规则引擎 + 预测性标签（实际收益） + 本地大模型辅助
- **回测系统**: 走步验证，T+1约束，对比沪深300基准
- **RAG**: FAISS + bge-large-zh-v1.5
- **模拟盘**: 东方财富 HTTP API 直连 + Qwen Skills 精排 + 交易时段自动调度器

## 免责声明

本项目仅供研究与教育用途。模型输出不构成任何投资建议，不保证盈利。量化交易存在市场风险，使用者需自行承担交易风险和损失。请在充分理解相关风险后谨慎使用。
