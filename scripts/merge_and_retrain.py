#!/usr/bin/env python3
"""
合并所有训练数据源，准备最终微调
v1: GitHub问答 + BAAI金融(关键词过滤) + 英文量化指令 (~30k)
v2: 多市场行情技术分析对 (A股+期货+ETF+可转债, ~10k)
v3: FinGPT + 量化计算 + 推理链增强 (可选)
v4: BAAI全量中文金融 (~40k，替代v1中的BAAI子集)
"""

import json
import os
import hashlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "training-data")
OUTPUT = os.path.join(DATA_DIR, "merged_train_v2.jsonl")

sources = [
    (os.path.join(DATA_DIR, "baai_zh_full.jsonl"), "baai_zh_finance"),
    (os.path.join(DATA_DIR, "quant-github-generated.jsonl"), "github_quant"),
    (os.path.join(DATA_DIR, "all_market_train.jsonl"), "multi_market_technical"),
    (os.path.join(DATA_DIR, "fingpt_forecaster.jsonl"), "fingpt_forecaster"),
    (os.path.join(DATA_DIR, "quant_calculations.jsonl"), "quant_calculations"),
    (os.path.join(DATA_DIR, "reasoning_enhanced.jsonl"), "reasoning_enhanced"),
]


def user_hash(record):
    """用 user message 内容做 hash，用于去重"""
    for msg in record.get("messages", []):
        if msg["role"] == "user":
            return hashlib.md5(msg["content"].encode()).hexdigest()
    return None


total = 0
seen_hashes = set()
dedup_count = 0
source_counts = {}  # 按数据源统计

with open(OUTPUT, "w", encoding="utf-8") as out:
    for path, label in sources:
        count = 0
        skipped = 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    h = user_hash(record)

                    # reasoning_enhanced 的记录与已有数据有重叠，按 hash 去重
                    # 保留 reasoning 版本（后加载覆盖前面的）
                    if label == "reasoning_enhanced" and h and h in seen_hashes:
                        record["source"] = label
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        dedup_count += 1
                        continue

                    if h:
                        if h in seen_hashes and label != "reasoning_enhanced":
                            skipped += 1
                            continue
                        seen_hashes.add(h)

                    record["source"] = label
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
        except FileNotFoundError:
            print(f"[SKIP] {path} 不存在（可选数据源，跳过）")
            continue
        source_counts[label] = count
        msg = f"[{label}] {count} 条"
        if skipped:
            msg += f" (去重跳过 {skipped})"
        print(msg)
        total += count

print(f"\n合并完成：{total} 条 → {OUTPUT}")
if dedup_count:
    print(f"  推理链增强覆盖: {dedup_count} 条")

# 数据平衡性分析
print(f"\n{'=' * 60}")
print("数据平衡性分析")
print(f"{'=' * 60}")
print(f"{'数据源':<30s} {'条数':>6s} {'占比':>8s} {'状态'}")
print("-" * 60)
for label, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    pct = cnt / total * 100
    if pct > 50:
        status = "⚠ 占比过高，可能主导训练"
    elif pct < 2:
        status = "⚠ 占比过低，考虑过采样"
    else:
        status = ""
    print(f"{label:<30s} {cnt:>6d} {pct:>7.1f}% {status}")
print("-" * 60)
print(f"{'合计':<30s} {total:>6d} {'100.0%':>8s}")
