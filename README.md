# QuantLLM — 量化交易 AI 助手

> **本项目仅供研究与教育用途，不构成投资建议。使用前请阅读末尾[免责声明](#免责声明)。**

基于 Qwen2.5-14B QLoRA 微调的量化交易垂类模型。覆盖 A股、商品期货、ETF基金、可转债四大市场。已接入东方财富模拟盘，实现从信号生成到自动交易的完整闭环。

**核心发现**：A 股所有主要技术因子 IC 均为负值 → 趋势跟踪失效，反转/均值回归才是主导效应。

## 快速开始

```bash
# 一键全流程（数据采集 → 转换 → 合并 → 训练）
bash /opt/quant-llm/run.sh

# 分步执行
bash /opt/quant-llm/run.sh crawl      # 数据采集（A股+多市场）
bash /opt/quant-llm/run.sh recalc     # 重算技术指标（从basic重算，不爬取）
bash /opt/quant-llm/run.sh fund-flow  # 爬取资金流数据
bash /opt/quant-llm/run.sh convert    # 行情 → 训练问答对
bash /opt/quant-llm/run.sh predict    # 预测性训练数据（实际收益标签）
bash /opt/quant-llm/run.sh generate   # 数据增强（FinGPT+量化计算+推理链）
bash /opt/quant-llm/run.sh merge      # 合并所有数据源 → 最终训练集
bash /opt/quant-llm/run.sh train      # QLoRA 微调
bash /opt/quant-llm/run.sh export     # 导出 GGUF（供 ollama 本地推理）
bash /opt/quant-llm/run.sh eval       # 模型评估
bash /opt/quant-llm/run.sh backtest   # 回测验证（对比沪深300）
bash /opt/quant-llm/run.sh trade-live # 双层实盘决策（规则+Qwen精排）
```

## 硬件要求

| 项目 | 最低配置 |
|------|----------|
| GPU | NVIDIA RTX A5000 24GB（或同等显存） |
| 内存 | 64GB+ |
| 存储 | 50GB 可用 |
| CUDA | 12.x |

## 环境配置

```bash
python3 -m venv /opt/quant-llm/finetune-env
source /opt/quant-llm/finetune-env/bin/activate
pip install -r requirements.txt

# 数据增强需本地 ollama
ollama pull qwen3:14b        # 种子扩展 + 预热精排
ollama pull qwen3:4b-nothink # 盘中精排（低延迟）
```

所有配置集中在 `config.yaml`，脚本通过 `scripts/_config.py` 读取。

## 项目结构

```
/opt/quant-llm/
├── run.sh                             # 一键执行入口
├── config.yaml                        # 中心化配置
├── requirements.txt                   # Python 依赖
├── scripts/                           # 生产脚本（28个，共11k行）
│   ├── _config.py                     #   配置加载器
│   ├── indicators.py                  #   28个技术指标（纯pandas/numpy）
│   ├── crawl_ashare.py                #   A股行情爬取（支持 --recalc）
│   ├── crawl_multi_market.py          #   期货+ETF+可转债
│   ├── crawl_fund_flow.py             #   资金流数据
│   ├── convert_all_to_training.py     #   行情 → 训练问答对
│   ├── generate_predictive_data.py    #   预测性数据（真实收益标签）
│   ├── fetch_fingpt_data.py           #   FinGPT数据
│   ├── generate_quant_calculations.py #   量化计算种子扩展
│   ├── add_reasoning_chains.py        #   推理链增强
│   ├── merge_and_retrain.py           #   合并训练集
│   ├── train.py                       #   QLoRA 微调
│   ├── evaluate.py                    #   模型评估
│   ├── export_gguf.py                 #   GGUF 导出
│   ├── backtest_signals.py            #   回测系统
│   ├── rag_build_index.py             #   RAG 索引构建
│   ├── rag_retrieve.py                #   RAG 检索
│   ├── rag_serve.py                   #   RAG 推理服务
│   ├── qwen_skills.py                 #   Qwen Skills 精排
│   ├── eastmoney_http_api.py          #   东方财富 HTTP API
│   ├── trade_execution.py             #   交易执行核心（风控+幂等）
│   ├── trade_live_qwen.py             #   双层实盘入口
│   ├── trade_session_runner.py        #   交易时段调度器
│   ├── reconcile_positions.py         #   策略-券商持仓对账
│   ├── monitor_alerts.py              #   监控告警
│   ├── generate_daily_report.py       #   日报生成
│   ├── compare_backtest_live.py       #   回测vs实盘对比
│   └── dpo_build_pairs.py             #   DPO偏好对构建
├── training-data/                     # 训练数据（.gitignore）
│   ├── ashare/                        #   A股（4173只，含28个技术指标）
│   ├── futures/                       #   期货
│   ├── etf/                           #   ETF
│   ├── cbond/                         #   可转债
│   ├── predictive_signals.jsonl       #   个股预测（32705条）
│   ├── sector_rotation.jsonl          #   板块轮动（186条）
│   └── merged_train_v4.jsonl          #   最终训练集（61556条）
├── output/
│   ├── quant-qwen2.5-14b-v4-trapboost/  # LoRA 适配器权重
│   ├── rag_index.faiss                #   RAG 索引
│   ├── trade_logs/                    #   交易日志+信号计划
│   │   ├── plans/                     #     每日信号计划
│   │   ├── trade_records.jsonl        #     全量交易明细
│   │   └── executed_orders.jsonl      #     幂等去重记录
│   ├── backtest_results.json          #   回测结果
│   └── eval_results.json              #   评估结果
├── logs/                              # 运行日志（按YYYY-MM-DD归档）
├── docs/                              # 文档
├── archive/                           # 历史归档（不参与生产）
└── finetune-env/                      # Python 虚拟环境
```

## 训练流水线

### 数据采集

- **A 股**：东方财富 API（akshare），主板+创业板+科创板，含退市股
- **多市场**：期货~80品种、ETF~800只、可转债~400只
- **资金流**：板块/个股/大盘资金流排名

### 评分系统（反转策略 v2）

基于 IC 因子分析验证，所有评分因子按**反转逻辑**设计：

| 因子组 | 权重上限 | 核心因子（IC_IR） | 策略含义 |
|--------|---------|-------------------|---------|
| 动量反转 | ±20 | RSI(-0.40), 20d趋势(-0.46), ROC(-0.39) | 超买看空、超卖看多 |
| 趋势确认 | ±12 | MA20偏离(-0.38), CCI(-0.30), MFI(-0.31) | 偏离均值回归 |
| 波动率 | ±10 | HV(-0.42), BB位置(-0.35) | 高波动=风险 |
| 量能 | ±8 | 量比(-0.21), 5d趋势(-0.18) | 放量见顶 |
| 困境反转 | ±15 | 价格分位数、量能萎缩、底部反弹 | 极度超跌→反转 |
| 市场环境 | ±8 | 沪深300 regime | 牛/熊/震荡自适应 |

### 训练数据

| 数据源 | 条数 | 内容 |
|--------|------|------|
| BAAI 中文金融 | 40,290 | 金融知识底座 |
| 多市场技术分析 | 8,855 | A股/期货/ETF/可转债 |
| 推理链增强 | 1,820 | 带 `<think>` 推理过程 |
| FinGPT 预测 | 1,230 | 道琼斯30趋势预测 |
| R1 金融推理 | 1,000 | DeepSeek-R1 推理 |
| 量化计算 | 350 | 风险指标/期权定价 |
| **合计（merged_train_v4）** | **61,556** | 含预测标签+trapboost |

> 预测性数据（32,705条个股 + 186条板块轮动）在 merge 阶段整合，标签为实际未来收益。

### 训练参数

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-14B (4bit) |
| LoRA rank / alpha | 32 / 32 |
| 目标模块 | q/k/v/o/gate/up/down_proj |
| 学习率 | 1e-4（cosine decay） |
| 等效 batch | 8（bs=1, grad_accum=8） |
| Epoch | 3（early stopping patience=20） |
| 序列长度 | 2048 |

### 回测结果（2005-2025）

| 指标 | 个股反转策略 | 沪深300 |
|------|-------------|---------|
| 总收益率 | **+1201%** | +122% |
| 年化收益率 | **12.99%** | 6.08% |
| 夏普比率 | 0.383 | 0.302 |
| 胜率 | 39.3% | — |
| 盈亏比 | 1.122 | — |

## 模拟盘交易系统

### 架构

```
09:00-09:25 Preopen（每日一次）
    沪深300 Regime 检测（5维度→连续分 0-100）
        ↓
    规则引擎初筛（compute_score → Top 50）
        ↓
    Qwen Skills 精排（qwen3:14b → Top N，输出 rank/action/reason）
        ↓
    生成当日信号计划（signal_plan_YYYYMMDD.json）
        + 每周买入清单（weekly_buylist_YYYY-WNN.json）

09:30-15:00 Live（每5分钟轮询）
    读取当日信号计划（不重新计算）
        ↓
    执行计划构建（score驱动仓位 + regime自适应 + 波动率目标化）
        ↓
    风控层（单票上限/总仓位/限频/幂等/kill switch/T+1）
        ↓
    两阶段执行：先卖后买
        ↓
    东方财富 HTTP API 直连（双源行情容灾 → 下单 → 回执核验）
```

**信号与执行分离**：preopen 阶段用大模型（qwen3:14b）精排一次生成计划，盘中轮询只执行已有计划（qwen3:4b-nothink），避免反转策略下的高频无效重算。

### 市场环境感知

基于沪深300的 5 维度 regime 检测，原始分 [-5, +5] 映射为连续分 [0, 100]：

| 维度 | 牛市(+1) | 熊市(-1) |
|------|----------|----------|
| 价格 vs MA120 | >+5% | <-5% |
| MA120 斜率 | >+1% | <-1% |
| 量能趋势 | 放量>1.3x | — |
| 波动率 | 低波<0.7x | 高波>1.8x |
| 60日动量 | >+15% | <-15% |

**连续化仓位**：

| Regime 分数 | 目标仓位 | 最大持仓 | 选股门槛 |
|-------------|---------|---------|----------|
| 0-30（极端熊市） | 5-20% | 2只 | 仅 strong_buy |
| 30-50（温和熊市） | 30-50% | 3只 | buy + strong_buy |
| 50-70（震荡/温和牛） | 50-70% | 5只 | buy + strong_buy |
| 70-100（强牛市） | 80-95% | 10只 | buy + strong_buy |

### 策略特性

- **非对称买卖**：低位不抢（RSI<42 或 regime≤50 时等 20 日趋势回升确认），高位分批卖出
- **波动率目标化**：按候选池 HV20 中位数缩放总仓位，目标年化波动率 15%
- **基本面过滤**：PE/PB/ROE 叠加（需因子文件），排除价值陷阱（PE≤0 或 ROE≤0 扣分）
- **北向资金因子**：净流入 +3 / 净流出 -3

### 风控体系

| 层级 | 措施 |
|------|------|
| 事前 | 单票上限 10%，总仓位上限 80%（受 regime 约束），日内上限 20 笔 |
| 事前 | 幂等去重（日期\|broker\|symbol\|side），failed 允许 3 次重试 |
| 事前 | kill switch 文件阻断、手动覆盖机制 |
| 事中 | 双源行情容灾，取价失败跳过不盲报 |
| 事中 | T+1 约束（当日买入不可卖出），委托去重（不堆积同方向委托） |
| 事中 | 两阶段执行（先卖出释放资金，再买入） |
| 事后 | 止损 -12%/只，移动止盈（浮盈 12% 后回撤 6% 触发） |
| 事后 | 组合止损 -30%（暂停交易），净值翻倍止盈提示 |
| 事后 | 异常检测（单笔 >12 万、日买入 >30 万、失败率 >60%） |
| 事后 | 全量交易明细落盘，按日归档 |

### 运维工具

| 工具 | 用途 |
|------|------|
| `reconcile_positions.py` | 策略持仓 vs 券商实际持仓对账 |
| `monitor_alerts.py` | 连续失败/长时间无交易告警（支持 webhook） |
| `generate_daily_report.py` | 日报生成（Markdown + 可选 DOCX） |
| `compare_backtest_live.py` | 回测 vs 实盘收益对比 |
| `dpo_build_pairs.py` | 从交易日志构建 DPO 偏好对 |

## RAG 检索增强

```
用户查询 → bge-large-zh-v1.5 编码 → FAISS top-3 → 注入 prompt → ollama 推理
```

- FAISS IndexFlatIP，score_threshold=0.35
- 含 `[MARKET_DATA]` 的查询跳过 RAG

## TODO — 策略优化路线

> 每周维护更新，上次更新：2026-04-03

### P0 — 验证与修复（当前阶段）

- [ ] **模拟盘实跑验证** — 目标：跑赢沪深300至少3个月（从2026-04-03开始计）
- [ ] **节假日 API 替换** — timor.tech 已返回 403，当前靠 weekday fallback，遇调休日会误判。替换为 `chinese_calendar` 库或其他可靠数据源
- [ ] **config.yaml broker 统一** — `trade_live.broker` 写的是 `eastmoney_paper`，runner 实际用 `--broker eastmoney_sim`，应统一为 `eastmoney_sim` 避免混淆

### P1 — 因子数据补齐（影响评分质量）

- [ ] **基本面因子生产链** — `training-data/factors/stock_factors_latest.json` 不存在，PE/PB/ROE/北向资金因子全为 0，评分完全依赖技术面。需建立采集→清洗→落盘流程
- [ ] **因子数据定时更新** — 因子文件需每日/每周更新，纳入 cron 或 systemd timer

### P3 — 实盘准备

- [ ] **实盘切换** — 模拟盘跑赢沪深300三个月后，换支持 QMT 的券商（国金/华鑫）

### P4 — 工程化与稳定性

- [ ] **定时任务编排** — 日报/监控/对账/回测对比纳入 cron/systemd 统一调度
- [ ] **run.sh 增补命令** — dpo-build / daily-report / monitor / reconcile / compare
- [ ] **监控告警通道实装** — webhook 分级、重试、静默窗口、去重
- [ ] **日志轮转归档** — audit/anomaly/trade_records 留存周期与压缩
- [ ] **最小测试集 + CI** — 关键路径单元测试接入 GitHub Actions
- [ ] **交易凭据安全治理** — Cookie 过期检测、自动刷新、审计策略

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 基座模型 | Qwen2.5-14B (4bit) |
| 微调 | Unsloth + TRL SFTTrainer, QLoRA r=32 |
| 数据源 | akshare（东方财富 API）、HuggingFace |
| 技术指标 | 28 个自研（纯 pandas/numpy，无 ta-lib） |
| 训练数据 | 规则引擎 + 真实收益标签 + 本地大模型辅助 |
| 回测 | 走步验证，T+1 约束，对比沪深300 |
| RAG | FAISS + bge-large-zh-v1.5 |
| 精排 | Qwen3 Skills（qwen3:14b/4b-nothink） |
| 模拟盘 | 东方财富 HTTP API 直连 + 交易时段调度器 |
| 运行环境 | PyTorch 2.6 + CUDA 12.4, RTX A5000 24GB |

## 免责声明

本项目仅供研究与教育用途。模型输出不构成任何投资建议，不保证盈利。量化交易存在市场风险，使用者需自行承担交易风险和损失。
