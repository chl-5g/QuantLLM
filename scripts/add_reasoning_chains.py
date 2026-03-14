#!/usr/bin/env python3
"""
推理链增强脚本
用 qwen3:14b 思考模式为高质量训练数据添加 <think> 推理链
输入：merged_train_v2.jsonl 中的高质量记录
输出：training-data/reasoning_enhanced.jsonl
"""

import json
import os
import hashlib
import requests
import time

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:14b"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_ROOT, "training-data", "merged_train_v2.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "training-data", "reasoning_enhanced.jsonl")
PROGRESS_FILE = os.path.join(PROJECT_ROOT, "training-data", ".reasoning_progress.json")

SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"

# 量化关键词，用于筛选高质量记录
QUANT_KEYWORDS = [
    "策略", "回测", "因子", "alpha", "夏普", "波动率", "收益率", "风险",
    "均线", "MACD", "RSI", "技术分析", "量化", "对冲", "套利", "期权",
    "组合", "仓位", "止损", "止盈", "交易", "K线", "趋势", "动量",
    "价值", "成长", "市盈率", "ROE", "财务", "基本面", "资产配置",
    "久期", "凸性", "利率", "债券", "可转债", "ETF", "期货",
    "VaR", "最大回撤", "信息比率", "IC", "协整", "GARCH",
]

# 交易决策类关键词 — 包含这些的记录跳过推理链增强
# 依据 StockBench 发现：推理微调不优于指令微调（在交易决策场景）
DECISION_SKIP_KEYWORDS = [
    "买入", "卖出", "做多", "做空", "建仓", "加仓", "减仓", "清仓",
    "操作建议", "交易信号", "追高", "抄底", "逢低", "止盈止损",
    "应该买", "应该卖", "是否买入", "是否卖出",
    "综合判断", "短期趋势", "涨停", "跌停",
    "action", "buy", "sell", "hold",
]

# 适合推理链的知识解释类关键词
REASONING_PREFER_KEYWORDS = [
    "计算", "公式", "推导", "原理", "解释", "为什么", "如何理解",
    "区别", "比较", "VaR", "夏普", "久期", "凸性", "Black-Scholes",
    "凯利", "IC", "ICIR", "协整", "Hurst", "GARCH", "波动率建模",
    "风险度量", "组合优化", "因子分析", "统计检验",
]

MAX_RECORDS = 2000


def call_ollama(messages, max_retries=2, timeout=120):
    """调用 qwen3:14b 思考模式，返回 (thinking, content) 元组"""
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.6, "num_predict": 2048, "num_ctx": 4096},
                },
                timeout=(10, timeout),  # (connect_timeout, read_timeout)
            )
            resp.raise_for_status()
            msg = resp.json()["message"]
            # qwen3 思考模式：thinking 在单独字段，content 是最终回答
            thinking = msg.get("thinking", "") or ""
            content = msg.get("content", "") or ""
            return thinking, content
        except Exception as e:
            if attempt < max_retries:
                print(f"  重试 ({attempt+1}/{max_retries}): {e}")
                time.sleep(5)
            else:
                print(f"  调用失败: {e}")
                return None, None


def extract_think_block(text):
    """提取 <think>...</think> 块"""
    if not text or "<think>" not in text:
        return None, text

    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_end < 0:
        return None, text

    think_content = text[think_start + 7:think_end].strip()
    remaining = text[think_end + 8:].strip()
    return think_content, remaining


def record_hash(record):
    """用 user message 的 hash 标识记录"""
    user_msg = ""
    for msg in record.get("messages", []):
        if msg["role"] == "user":
            user_msg = msg["content"]
            break
    return hashlib.md5(user_msg.encode()).hexdigest()[:12]


def select_records(input_file, max_count):
    """筛选高质量记录（仅知识解释/量化计算类，跳过交易决策类）"""
    candidates = []
    skipped_decision = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            messages = record.get("messages", [])

            # 需要有 system + user + assistant
            if len(messages) < 3:
                continue

            assistant_msg = ""
            user_msg = ""
            for msg in messages:
                if msg["role"] == "assistant":
                    assistant_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]

            # 回答长度 > 200
            if len(assistant_msg) < 200:
                continue

            combined = user_msg + assistant_msg

            # 跳过交易决策类（StockBench 发现：推理微调不优于指令微调）
            decision_hits = sum(1 for kw in DECISION_SKIP_KEYWORDS if kw in combined)
            if decision_hits >= 2:
                skipped_decision += 1
                continue

            # 包含量化关键词
            keyword_count = sum(1 for kw in QUANT_KEYWORDS if kw in combined)
            if keyword_count < 2:
                continue

            # 优先推理链适合的主题（加权）
            reasoning_bonus = sum(1 for kw in REASONING_PREFER_KEYWORDS if kw in combined)
            score = keyword_count + reasoning_bonus * 2

            candidates.append((score, record))

    # 按分数排序，取前 max_count 条
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [r for _, r in candidates[:max_count]]
    print(f"  候选记录: {len(candidates)}, 跳过交易决策类: {skipped_decision}, 选中: {len(selected)}")
    return selected


def main():
    print("=" * 60)
    print("推理链增强 (qwen3:14b 思考模式)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"输入文件不存在: {INPUT_FILE}")
        return

    # 加载进度
    progress = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
        print(f"已有进度: {len(progress)} 条")

    # 筛选记录
    print("\n筛选高质量记录...")
    records = select_records(INPUT_FILE, MAX_RECORDS)

    # 处理
    results = []
    skipped = 0
    failed = 0
    reused = 0

    for i, record in enumerate(records):
        rhash = record_hash(record)

        # 检查进度
        if rhash in progress:
            if progress[rhash] == "__skip__":
                skipped += 1
                continue
            results.append(progress[rhash])
            reused += 1
            continue

        messages = record["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        original_answer = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        # 构造 prompt（/think 触发 qwen3 思考模式）
        r1_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"/think\n{user_msg}"},
        ]

        if (i - reused) % 10 == 0:
            print(f"\n[{i+1}/{len(records)}] 处理中 (复用{reused}, 新生成{len(results)-reused}, 跳过{skipped}, 失败{failed})")

        think_content, r1_answer = call_ollama(r1_messages)
        if think_content is None:
            failed += 1
            continue

        if not think_content or len(think_content) < 100:
            skipped += 1
            continue

        # 拼接：<think>推理</think> + 原始回答
        enhanced_answer = f"<think>\n{think_content}\n</think>\n\n{original_answer}"

        enhanced_record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": enhanced_answer},
            ],
            "metadata": {"source": "reasoning_enhanced", "original_hash": rhash},
        }

        results.append(enhanced_record)
        progress[rhash] = enhanced_record

        # 每10条保存一次进度
        if (i - reused) % 10 == 9:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False)
            print(f"  进度已保存 ({len(progress)} 条)")

        time.sleep(0.5)

    # 最终保存进度
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False)

    # 写出
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"推理链增强完成！")
    print(f"  输入: {len(records)} 条")
    print(f"  成功: {len(results)} 条 (复用{reused}, 新生成{len(results)-reused})")
    print(f"  跳过: {skipped} 条 (推理链太短)")
    print(f"  失败: {failed} 条")
    print(f"  输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
