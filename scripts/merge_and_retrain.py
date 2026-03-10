#!/usr/bin/env python3
"""
合并所有训练数据源，准备最终微调
v1: GitHub问答 + BAAI金融 + 英文量化指令 (~30k)
v2: 多市场行情技术分析对 (A股+期货+ETF+可转债, ~10k)
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "training-data")
OUTPUT = os.path.join(DATA_DIR, "merged_train_v2.jsonl")

sources = [
    (os.path.join(DATA_DIR, "merged_train.jsonl"), "v1_mixed"),
    (os.path.join(DATA_DIR, "all_market_train.jsonl"), "multi_market_technical"),
]

total = 0
with open(OUTPUT, "w", encoding="utf-8") as out:
    for path, label in sources:
        count = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    count += 1
        except FileNotFoundError:
            print(f"[SKIP] {path} 不存在")
            continue
        print(f"[{label}] {count} 条")
        total += count

print(f"\n合并完成：{total} 条 → {OUTPUT}")
