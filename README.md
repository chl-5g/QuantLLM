# 量化交易垂类大模型

> 基于 Qwen2.5 的量化交易领域 QLoRA 微调项目
> 状态：首轮训练已完成，数据持续扩充中

---

## 项目概况

**目标**：微调一个专精量化交易的大语言模型，覆盖策略生成、因子分析、回测解读、风险归因、代码调试等任务。

**基础设施**：
- 训练：宿主机 RTX A5000 24GB，Unsloth + QLoRA，环境 `/root/finetune-env`
- 编排/服务：k8s 集群（master 8GB + worker×3 各 2GB）
- 基础模型：Qwen2.5-14B（4bit 量化）
- 训练数据格式：JSONL（ChatML 对话格式）

---

## 目录结构

```
/tmp/training-data/
├── README.md                          # 本文件
├── .gitignore
├── run.sh                             # 自动化运行入口
├── scripts/                           # Python 运行脚本
│   ├── crawl_ashare.py                # A股数据爬取（akshare）
│   ├── convert_ashare_to_training.py  # A股数据 → 训练JSONL
│   ├── generate_training_data.py      # 用大模型批量生成训练对
│   ├── prepare_dataset.py             # 数据集合并/预处理
│   └── train.py                       # QLoRA 微调训练脚本
├── data/                              # 训练数据集
│   ├── merged_train.jsonl             # 合并后训练集（30101条）
│   ├── quant-github-generated.jsonl   # GitHub量化代码生成的训练对（55条）
│   └── quant-trading-instruct/        # 量化交易指令数据集
├── ashare/                            # A股爬取数据（不上传git）
├── github-repos/                      # 量化策略代码仓库（不上传git）
├── BAAI_IndustryInstruction_Finance-Economics/  # 公开数据集（不上传git）
├── finance-instruct-500k/             # 公开数据集（不上传git）
├── output/                            # 模型产出（不上传git）
└── unsloth_compiled_cache/            # 编译缓存（不上传git）
```

---

## 当前进展

### 已完成
- [x] 微调环境搭建（PyTorch 2.6 + cu124 + Unsloth）
- [x] A股数据爬取（4150只股票基本面+进阶数据）
- [x] 公开数据集收集（BAAI金融指令、finance-instruct-500k）
- [x] GitHub量化代码仓库收集（backtesting.py、QuantsPlaybook等）
- [x] 训练数据合并（30101条 JSONL）
- [x] 首轮 QLoRA 训练完成（Qwen2.5-14B，checkpoint-1000）

### 待完成
- [ ] 训练数据质量审核与清洗
- [ ] 扩充手工构建的高质量指令数据
- [ ] 模型评估（量化问答准确率、代码可运行率）
- [ ] 迭代训练（数据扩充后重训）
- [ ] 模型部署为推理服务

---

## 训练数据来源

| 来源 | 类型 | 数量 | 状态 |
|------|------|------|------|
| BAAI 金融经济指令 | 公开数据集 | - | 已下载 |
| finance-instruct-500k | 公开数据集 | - | 已下载 |
| A股基本面数据 | 爬取 → 转训练对 | 4150只 | 已完成 |
| GitHub量化仓库 | 代码 → 训练对 | 55条 | 待扩充 |
| 手工构建 | 自有策略/分析 | 0 | 待开始 |
| **合并训练集** | **merged_train.jsonl** | **30101条** | **已用于首轮训练** |

### 训练样本类型

| 任务 | 输入示例 | 输出示例 |
|------|----------|----------|
| 策略生成 | "写一个基于动量因子的选股策略" | 完整 Python 代码 + 说明 |
| 因子分析 | IC 序列数据 | 有效性分析文本 |
| 回测解读 | Sharpe/MaxDD/换手率 | 策略评价 |
| 风险归因 | 持仓列表 | 风格暴露分析 |
| 信号过滤 | 市场状态描述 | 适用/失效因子判断 |
| 代码调试 | 报错的回测代码 | 修复后代码 + 原因 |

---

## 工作流架构

```
数据采集 → 数据处理 → 训练集构建 → [模型训练] → 模型评估 → 模型服务
                                      ↑
                               宿主机 RTX A5000
                               /root/finetune-env
                            (不进 k8s，由 k8s Job 触发)
```

### Workflow 1：数据采集

**职责**：从各数据源定时拉取原始数据，存入共享存储。

**数据源**：
| 类型 | 来源 |
|------|------|
| 行情/因子数据 | tushare、akshare |
| 财报/公告 | 交易所官网、东方财富 |
| 量化代码 | GitHub 量化策略仓库 |
| 研报/文章 | 雪球、掘金量化、知乎量化 |

**k8s 资源**：`CronJob`（定时触发）→ `Job`（执行采集）

### Workflow 2：数据处理

**职责**：对原始数据进行去重、清洗、格式化、质量过滤。

**任务细节**：
- 文本去重（MinHash 或精确哈希）
- 代码格式校验（语法检查、可运行性验证）
- 质量打分过滤（长度、结构完整性）
- 转为统一中间格式

**k8s 资源**：`Job`（可拆分多个并行 worker）

### Workflow 3：训练集构建（JSONL）

**职责**：将处理后的原始数据转化为 instruction-output 对，输出 JSONL。

**k8s 资源**：`Job`（调用 ollama API 批量生成候选样本）

### Workflow 4：模型训练

**运行位置**：宿主机（不在 k8s 内）
- 环境：`/root/finetune-env`（PyTorch 2.6+cu124，Unsloth）
- GPU：RTX A5000 24GB
- 方式：QLoRA，4bit 量化

### Workflow 5：模型评估

**评估维度**：
- 量化问答准确率（领域知识）
- 策略代码生成可运行率
- 回测结果解读质量（人工抽检）
- 与基础模型的对比得分

### Workflow 6：模型服务

**服务方式**：ollama 加载微调后的模型，对外暴露 REST API
**k8s 资源**：`Deployment` + `Service`

---

## k8s 资源分配

```
┌──────────────────────────────────────────────────────────────┐
│  master (8GB)                                                │
│  ├── 工作流调度（CronJob 控制器）                             │
│  ├── 数据采集 CronJob（轻量，<200MB）                        │
│  └── 模型服务 Deployment（ollama，~5-6GB）                   │
├──────────────────────────────────────────────────────────────┤
│  worker1 (2GB)  数据处理 Job（pandas/polars，CPU 密集）      │
│  worker2 (2GB)  训练集构建 Job（调用 ollama API 生成样本）   │
│  worker3 (2GB)  模型评估 Job + 策略回测 Job                 │
└──────────────────────────────────────────────────────────────┘
          │
          │ SSH 触发
          ▼
   宿主机（RTX A5000 24GB）
   └── Workflow 4：模型训练（Unsloth + QLoRA）
```

---

## 快速开始

```bash
cd /tmp/training-data

# 完整流程（爬取 → 转换 → 生成 → 合并 → 训练）
./run.sh all

# 或分步执行
./run.sh crawl      # 爬取A股数据
./run.sh convert    # 转换A股数据为训练格式
./run.sh generate   # 用大模型生成GitHub训练对
./run.sh merge      # 合并所有数据集
./run.sh train      # QLoRA 微调训练
```

---

## 待决策事项

- [ ] 训练数据质量审核标准
- [ ] 共享存储方案（NFS / hostPath / PVC）
- [ ] 是否引入 Argo Workflows，还是先用简单 Job
- [ ] 模型评估基准数据集如何构建
- [ ] 训练数据目标规模

---

## 下一步行动

1. 审核现有 30k 训练数据质量，清洗低质量样本
2. 扩充 GitHub 量化代码生成的训练对（目前仅 55 条）
3. 开始手工构建高质量指令数据（目标 500+ 条）
4. 建立评估基准，量化评估首轮模型
5. 迭代训练，提升模型质量
