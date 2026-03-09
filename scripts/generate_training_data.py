#!/usr/bin/env python3
"""
从 GitHub 量化仓库提取策略代码，用本地大模型生成中文训练问答对。
输出格式：ChatML JSONL
"""

import json
import os
import glob
import requests
import time
import sys

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:14b"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "quant-github-generated.jsonl")

# ============================================================
# 第一步：收集策略代码文件
# ============================================================

def collect_strategy_files():
    """收集所有策略相关的 Python 文件"""
    files = []

    # quant-trading: 每个 backtest 文件都是独立策略
    base = os.path.join(PROJECT_ROOT, "github-repos", "quant-trading-master")
    for f in glob.glob(f"{base}/*.py"):
        if "backtest" in f.lower() or "pattern" in f.lower() or "vix" in f.lower():
            files.append(("quant-trading", f))

    # backtesting.py: examples 目录
    base2 = os.path.join(PROJECT_ROOT, "github-repos", "backtesting.py-master", "doc", "examples")
    for f in glob.glob(f"{base2}/*.py"):
        files.append(("backtesting.py", f))

    return files

# ============================================================
# 第二步：为每个代码文件生成多种任务的问答对
# ============================================================

TASK_TEMPLATES = [
    {
        "task": "code_explain",
        "prompt": """你是一个量化交易专家和编程教师。请阅读以下量化交易策略代码，然后生成一个高质量的中文问答对。

**要求**：
- 问题：用自然的中文提问，请求解释这段策略的核心逻辑
- 回答：详细解释策略原理、关键参数、买卖信号的触发条件，用通俗易懂的中文

**代码**：
```python
{code}
```

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"question": "你的问题", "answer": "你的回答"}}"""
    },
    {
        "task": "strategy_improve",
        "prompt": """你是一个量化交易专家。请阅读以下策略代码，生成一个关于"如何改进这个策略"的中文问答对。

**要求**：
- 问题：询问如何改进或优化这个策略
- 回答：给出 3-5 个具体的改进方向（如参数优化、风控、信号过滤等），每个方向给出具体建议

**代码**：
```python
{code}
```

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"question": "你的问题", "answer": "你的回答"}}"""
    },
    {
        "task": "backtest_analysis",
        "prompt": """你是一个量化交易专家。请阅读以下策略代码，生成一个关于"如何分析该策略的回测结果"的中文问答对。

**要求**：
- 问题：询问如何评估这个策略的回测表现
- 回答：说明应该关注哪些指标（Sharpe、MaxDD、胜率、盈亏比等），以及这类策略常见的风险点

**代码**：
```python
{code}
```

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"question": "你的问题", "answer": "你的回答"}}"""
    },
    {
        "task": "code_rewrite",
        "prompt": """你是一个量化交易专家和 Python 程序员。请阅读以下策略代码，生成一个"用 backtrader 框架重写该策略"的中文问答对。

**要求**：
- 问题：请求用 backtrader 框架实现类似策略
- 回答：给出完整的 backtrader 策略类代码（包含 __init__ 和 next 方法），并加中文注释

**代码**：
```python
{code}
```

请严格按以下 JSON 格式输出，不要输出其他内容：
{{"question": "你的问题", "answer": "你的回答"}}"""
    },
]

def call_ollama(prompt, max_retries=2):
    """调用本地 ollama API"""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 2048}
                },
                timeout=120
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return content
        except Exception as e:
            if attempt < max_retries:
                print(f"  重试 ({attempt+1}/{max_retries}): {e}")
                time.sleep(3)
            else:
                print(f"  调用失败: {e}")
                return None

def parse_qa(response_text):
    """从大模型回复中提取 JSON 问答对"""
    if not response_text:
        return None

    # 尝试直接 parse
    text = response_text.strip()

    # 去掉 markdown code block
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # 去掉 think 标签
    if "<think>" in text:
        text = text.split("</think>")[-1].strip()

    # 找 JSON 对象
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end])
            if "question" in obj and "answer" in obj:
                if len(obj["answer"]) > 50:  # 过滤太短的回答
                    return obj
        except json.JSONDecodeError:
            pass
    return None

def process_file(source, filepath, out_f):
    """处理单个策略文件，生成多个问答对"""
    filename = os.path.basename(filepath)

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()
    except Exception as e:
        print(f"  读取失败: {e}")
        return 0

    # 跳过太短或太长的文件
    if len(code) < 100:
        return 0
    if len(code) > 8000:
        code = code[:8000] + "\n# ... (代码截断)"

    count = 0
    for template in TASK_TEMPLATES:
        prompt = template["prompt"].format(code=code)
        print(f"  [{template['task']}] 生成中...", end=" ", flush=True)

        response = call_ollama(prompt)
        qa = parse_qa(response)

        if qa:
            record = {
                "messages": [
                    {"role": "system", "content": "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "metadata": {
                    "source": source,
                    "file": filename,
                    "task": template["task"]
                }
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            count += 1
            print(f"OK ({len(qa['answer'])} chars)")
        else:
            print("SKIP (解析失败)")

    return count

# ============================================================
# 主流程
# ============================================================

def main():
    files = collect_strategy_files()
    print(f"找到 {len(files)} 个策略文件")

    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for i, (source, filepath) in enumerate(files):
            filename = os.path.basename(filepath)
            print(f"\n[{i+1}/{len(files)}] {source}/{filename}")
            count = process_file(source, filepath, out_f)
            total += count
            print(f"  生成 {count} 条")

    print(f"\n{'='*60}")
    print(f"总计生成 {total} 条训练数据")
    print(f"输出文件: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
