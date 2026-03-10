#!/usr/bin/env python3
"""
BAAI IndustryInstruction 全量中文金融数据提取
将所有中文数据（不仅限于关键词匹配）导出为 ChatML JSONL
"""

import json
import os
import re
from datasets import load_from_disk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "training-data")
INPUT_PATH = os.path.join(DATA_DIR, "BAAI_IndustryInstruction_Finance-Economics")
OUTPUT_PATH = os.path.join(DATA_DIR, "baai_zh_full.jsonl")
SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"

ds = load_from_disk(INPUT_PATH)
print(f"BAAI 总数据: {len(ds)} 条")

count = 0
skipped_short = 0
skipped_no_zh = 0

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for i in range(len(ds)):
        convs = ds[i]["conversations"]
        if len(convs) < 2:
            continue

        text = convs[0]["value"] + convs[1]["value"]

        # 只要中文数据（至少10个连续中文字符）
        if not re.search(r'[\u4e00-\u9fff]{10,}', text):
            skipped_no_zh += 1
            continue

        # 过滤回答太短的
        if len(convs[1]["value"]) < 50:
            skipped_short += 1
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for c in convs:
            role = "user" if c["from"] == "human" else "assistant"
            messages.append({"role": role, "content": c["value"]})

        record = {"messages": messages, "source": "baai_zh_finance"}
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

print(f"\n提取完成:")
print(f"  中文数据: {count} 条 → {OUTPUT_PATH}")
print(f"  跳过(非中文): {skipped_no_zh}")
print(f"  跳过(回答太短): {skipped_short}")
