"""
Qwen3 Skills 系统 - 结构化任务调用框架。
类似 RAG 中的 cross-encoder reranker：规则引擎粗排 -> LLM 精排。
"""

import json
import os
import re

import jsonschema

from _config import call_ollama, cfg

# ============================================================
# StockRankSkill - top50 -> top10 精排
# ============================================================

STOCK_RANK_SYSTEM = """你是专业量化分析师。根据市场环境和候选股票的技术指标，从候选列表中精选最优的{top_n}只股票。

要求：
1. 综合考虑市场环境（牛市/熊市/震荡）调整选股偏好
2. 牛市偏趋势跟随，熊市/震荡偏困境反转和防御
3. 关注多因子共振：RSI超卖+布林下轨+量能萎缩=强反转信号
4. 只输出一个JSON对象，格式：{{"rankings": [...]}}
5. 每个元素：rank(int), symbol(str), score(number), action("strong_buy"/"buy"/"hold"), reason(str), risk_factors(list[str])"""

STOCK_RANK_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["rankings"],
    "properties": {
        "rankings": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["rank", "symbol", "score", "action", "reason"],
                "properties": {
                    "rank": {"type": "integer", "minimum": 1},
                    "symbol": {"type": "string"},
                    "score": {"type": "number"},
                    "action": {"enum": ["strong_buy", "buy", "hold"]},
                    "reason": {"type": "string"},
                    "risk_factors": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def _extract_json(text):
    """只提取首个完整 JSON 对象，忽略尾部污染字符块。"""
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty", text, 0)

    # 去掉 think 标签
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

    # 去掉 markdown 代码块外壳
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    decoder = json.JSONDecoder()

    # 从首个 '{' 开始尝试 raw_decode，只保留第一个完整对象
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in starts:
        try:
            obj, _ = decoder.raw_decode(text[s:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _is_ranking_item(obj):
    if not isinstance(obj, dict):
        return False
    required = {"rank", "symbol", "score", "action", "reason"}
    return required.issubset(set(obj.keys()))


def _extract_ranking_items(text, max_items=200):
    """
    容错解析：当模型没有返回完整 {"rankings":[...]} 包装时，
    直接从文本里抓取一组 ranking 对象。
    """
    src = (text or "").strip()
    if not src:
        return []
    src = re.sub(r"</?think>", "", src, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", src, re.DOTALL | re.IGNORECASE)
    if m:
        src = m.group(1).strip()

    items = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(src) and len(items) < max_items:
        start = src.find("{", idx)
        if start == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(src[start:])
            if _is_ranking_item(obj):
                items.append(obj)
            idx = start + max(consumed, 1)
        except Exception:
            idx = start + 1
    return items


def call_skill_stock_rank(market_regime, candidates, top_n=None):
    """
    精排入口。
    candidates: list of dict，每个含 symbol, score, 及关键指标
    返回解析后的 rankings list，失败返回 None
    """
    skill_cfg = cfg.get("skills", {}).get("stock_rank", {})
    if top_n is None:
        top_n = skill_cfg.get("top_n", 10)

    system = STOCK_RANK_SYSTEM.format(top_n=top_n)
    user_msg = json.dumps({"market_regime": market_regime, "top_n": top_n, "candidates": candidates}, ensure_ascii=False)

    env_model = os.getenv("STOCK_RANK_MODEL", "").strip()
    model = (
        env_model
        or skill_cfg.get("model")
        or cfg.get("ollama", {}).get("live_rank_model")
        or cfg["ollama"]["generation_model"]
    )

    # assistant prefill 保证 JSON 开头（思考模式已由 think=False 关闭）
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": '{"rankings": ['},
    ]

    num_gpu = skill_cfg.get("num_gpu")
    extra = {}
    if num_gpu is not None:
        extra["num_gpu"] = num_gpu

    attempts = int(skill_cfg.get("attempts", 2))
    ollama_timeout = int(skill_cfg.get("timeout", cfg.get("ollama", {}).get("timeout", 180)))
    ollama_retries = int(skill_cfg.get("ollama_max_retries", 1))

    for attempt in range(attempts):
        raw = call_ollama(
            model=model,
            messages=messages,
            temperature=skill_cfg.get("temperature", 0),
            num_predict=skill_cfg.get("num_predict", 1024),
            seed=skill_cfg.get("seed", 42),
            timeout=ollama_timeout,
            max_retries=ollama_retries,
            think=False,
            **extra,
        )
        if raw is None:
            continue

        # 拼回 prefill 前缀，并只保留首个 JSON 对象
        full_json = '{"rankings": [' + raw
        try:
            data = _extract_json(full_json)
            if isinstance(data, dict) and "rankings" in data:
                jsonschema.validate(data, STOCK_RANK_OUTPUT_SCHEMA)
                data["rankings"] = data["rankings"][:top_n]
                return data["rankings"]

            # 常见异常：模型直接输出 ranking item 序列，缺少 {"rankings": ...} 包装
            rescued = _extract_ranking_items(raw)
            if not rescued and _is_ranking_item(data):
                rescued = [data]
            if rescued:
                wrapped = {"rankings": rescued[:top_n]}
                jsonschema.validate(wrapped, STOCK_RANK_OUTPUT_SCHEMA)
                return wrapped["rankings"]
            raise jsonschema.ValidationError("no usable rankings payload")
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            print(f"  [SKILL] 输出校验失败 (attempt {attempt+1}): {e}")
            continue

    print("  [SKILL] stock_rank 调用失败")
    return None

