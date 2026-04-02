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
5. 每个元素：rank(int), symbol(str), score(number), action("strong_buy"/"buy"/"hold"/"sell"/"strong_sell"), reason(str), risk_factors(list[str])"""

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
                    "action": {"enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"]},
                    "reason": {"type": "string"},
                    "risk_factors": {"type": "array", "items": {"type": "string"}},
                },
            },
        }
    },
}


def _extract_json(text):
    """提取 JSON 对象：优先完整解析，回退到截断尾部后重试。"""
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty", text, 0)

    # 去掉 think 标签
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

    # 去掉 markdown 代码块外壳
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()

    # 优先尝试完整 json.loads（处理 prefill 拼接后的完整 JSON）
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 回退：截断尾部垃圾后重试（找最后一个 ]} 闭合）
    idx = text.rfind("]}")
    if idx > 0:
        try:
            obj = json.loads(text[:idx + 2])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 最终回退：raw_decode 从首个 '{' 开始
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for s in starts:
        try:
            obj, _ = decoder.raw_decode(text[s:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise json.JSONDecodeError("No JSON object found", text, 0)


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

    # assistant prefill 绕过 qwen3 的 thinking 模式
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
            **extra,
        )
        if raw is None:
            continue

        # 拼回 prefill 前缀，并只保留首个 JSON 对象
        full_json = '{"rankings": [' + raw
        try:
            data = _extract_json(full_json)
            jsonschema.validate(data, STOCK_RANK_OUTPUT_SCHEMA)
            data["rankings"] = data["rankings"][:top_n]
            return data["rankings"]
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            print(f"  [SKILL] 输出校验失败 (attempt {attempt+1}): {e}")
            continue

    print("  [SKILL] stock_rank 调用失败")
    return None

