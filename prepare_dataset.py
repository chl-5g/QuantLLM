#!/usr/bin/env python3
"""
合并所有训练数据为统一的 ChatML JSONL 格式
"""
import json
import re
from datasets import load_from_disk

OUTPUT = "/tmp/training-data/merged_train.jsonl"
SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"

records = []

# === 1. GitHub 加工数据 (55条，高质量中文) ===
print("加载 GitHub 加工数据...")
with open("/tmp/training-data/quant-github-generated.jsonl", "r") as f:
    for line in f:
        r = json.loads(line)
        records.append(r["messages"])
print(f"  {len(records)} 条")

# === 2. BAAI 中文金融数据 (筛选中文 + 量化相关) ===
print("加载 BAAI 中文金融数据...")
ds = load_from_disk("/tmp/training-data/BAAI_IndustryInstruction_Finance-Economics")
baai_count = 0

# 量化/金融关键词
quant_keywords = [
    '量化', '回测', '因子', '对冲', '策略', '选股', '动量', '均线', '止损', '仓位',
    '夏普', '最大回撤', '收益率', '波动率', '风险', '投资组合', '套利', '期货', '期权',
    '股票', '债券', '基金', '交易', '技术分析', '基本面', '财务', '估值', '市盈率',
    'alpha', 'beta', 'sharpe', 'var', '资产配置', '风控', '杠杆', '做空', '做多',
    '金融工程', '衍生品', '隐含波动率', 'black-scholes', '蒙特卡洛',
    '利率', '通胀', '央行', '货币政策', '财政政策', '宏观经济', '信用',
]

for i in range(len(ds)):
    convs = ds[i]["conversations"]
    if len(convs) < 2:
        continue

    # 只要中文数据
    text = convs[0]["value"] + convs[1]["value"]
    if not re.search(r'[\u4e00-\u9fff]{10,}', text):
        continue

    # 关键词匹配（放宽：只要包含任意金融关键词）
    text_lower = text.lower()
    if not any(kw in text_lower for kw in quant_keywords):
        continue

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for c in convs:
        role = "user" if c["from"] == "human" else "assistant"
        messages.append({"role": role, "content": c["value"]})

    # 过滤回答太短的
    if len(convs[1]["value"]) < 50:
        continue

    records.append(messages)
    baai_count += 1

print(f"  筛选出 {baai_count} 条中文金融数据")

# === 3. Quant-Trading-Instruct (386条，英文代码) ===
print("加载 Quant-Trading-Instruct...")
ds2 = load_from_disk("/tmp/training-data/quant-trading-instruct")
qt_count = 0
for i in range(len(ds2)):
    r = ds2[i]
    if len(r["answer"]) < 100:
        continue
    messages = [
        {"role": "system", "content": "You are a professional quantitative trading expert skilled in strategy development and backtesting."},
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["answer"]}
    ]
    records.append(messages)
    qt_count += 1
print(f"  {qt_count} 条")

# === 写出 ===
print(f"\n总计: {len(records)} 条")
with open(OUTPUT, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")

print(f"已保存到 {OUTPUT}")

# 统计
lens = []
for r in records:
    total_len = sum(len(m["content"]) for m in r)
    lens.append(total_len)
lens.sort()
print(f"对话总长度 - 中位数: {lens[len(lens)//2]}, P90: {lens[len(lens)*9//10]}, 最长: {lens[-1]}")
