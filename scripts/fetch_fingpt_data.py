#!/usr/bin/env python3
"""
从 HuggingFace 下载 FinGPT A股预测数据集，转换为 ChatML JSONL 格式
数据源：FinGPT forecaster 系列数据集（沪深50等）
输出：training-data/fingpt_forecaster.jsonl
"""

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "training-data", "fingpt_forecaster.jsonl")
SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长技术分析、因子分析、趋势判断和风险管理。"

# 按优先级尝试的数据集名称
DATASET_CANDIDATES = [
    "FinGPT/fingpt-forecaster-sz50-20230201-20240101-1-2-08",
    "FinGPT/fingpt-forecaster-sz50-20220601-20230601-1-2-08",
    "FinGPT/fingpt-forecaster-dow30-202305-202405",
]


def parse_llama2_prompt(prompt_text):
    """解析 Llama-2 格式的 prompt，提取 system 和 user 部分"""
    # 格式: [INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
    sys_match = re.search(r'<<SYS>>\s*(.*?)\s*<</SYS>>', prompt_text, re.DOTALL)
    system = sys_match.group(1).strip() if sys_match else ""

    # user 部分在 <</SYS>> 之后、[/INST] 之前
    user_match = re.search(r'<</SYS>>\s*(.*?)\s*\[/INST\]', prompt_text, re.DOTALL)
    if not user_match:
        # 备选：没有 SYS 标签的情况
        user_match = re.search(r'\[INST\]\s*(.*?)\s*\[/INST\]', prompt_text, re.DOTALL)
    user = user_match.group(1).strip() if user_match else prompt_text.strip()

    return system, user


def main():
    print("=" * 60)
    print("FinGPT A股预测数据 → ChatML 转换")
    print("=" * 60)

    # 尝试加载数据集
    from datasets import load_dataset

    ds = None
    used_name = None
    for name in DATASET_CANDIDATES:
        try:
            print(f"\n尝试加载: {name}")
            ds = load_dataset(name)
            used_name = name
            print(f"  成功！")
            break
        except Exception as e:
            print(f"  失败: {e}")

    if ds is None:
        print("\n所有候选数据集都无法加载，退出")
        return

    # 获取训练集
    if "train" in ds:
        data = ds["train"]
    else:
        # 取第一个可用的 split
        split_name = list(ds.keys())[0]
        data = ds[split_name]
        print(f"  使用 split: {split_name}")

    print(f"  原始记录数: {len(data)}")
    print(f"  字段: {list(data.features.keys())}")

    # 转换
    records = []
    skipped = 0

    for item in data:
        prompt = item.get("prompt", "")
        answer = item.get("answer", "")

        if len(answer) < 20:
            skipped += 1
            continue

        _, user_content = parse_llama2_prompt(prompt)

        if not user_content or len(user_content) < 10:
            skipped += 1
            continue

        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ]
        }
        records.append(record)

    # 写出
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"转换完成！")
    print(f"  数据集: {used_name}")
    print(f"  转换: {len(records)} 条")
    print(f"  跳过: {skipped} 条")
    print(f"  输出: {OUTPUT_FILE}")

    # 对话长度统计
    if records:
        lens = sorted(sum(len(m["content"]) for m in r["messages"]) for r in records)
        print(f"  对话长度 - 中位数: {lens[len(lens)//2]}, P90: {lens[len(lens)*9//10]}, 最长: {lens[-1]}")


if __name__ == "__main__":
    main()
