"""
Qwen3 Skills 系统 — 结构化任务调用框架。
类似 RAG 中的 cross-encoder reranker：规则引擎粗排 → LLM 精排。
"""

import json
import re
import jsonschema
from _config import cfg, call_ollama

# ============================================================
# StockRankSkill — top50 → top10 精排
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
                    "risk_factors": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }
}


def _extract_json(text):
    """从可能包含 markdown 代码块的文本中提取 JSON。"""
    # 尝试直接解析
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    # 提取 ```json ... ``` 块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 提取第一个 { ... } 块
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON found", text, 0)


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
    user_msg = json.dumps({
        "market_regime": market_regime,
        "top_n": top_n,
        "candidates": candidates
    }, ensure_ascii=False)

    model = cfg["ollama"]["generation_model"]

    for attempt in range(2):
        raw = call_ollama(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ],
            temperature=skill_cfg.get("temperature", 0),
            num_predict=skill_cfg.get("num_predict", 4096),
            seed=skill_cfg.get("seed", 42),
            think=False,  # 关闭 thinking，结构化输出不需要推理链
        )
        if raw is None:
            continue

        try:
            data = _extract_json(raw)
            jsonschema.validate(data, STOCK_RANK_OUTPUT_SCHEMA)
            data["rankings"] = data["rankings"][:top_n]
            return data["rankings"]
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            print(f"  [SKILL] 输出校验失败 (attempt {attempt+1}): {e}")
            continue

    print("  [SKILL] stock_rank 调用失败")
    return None
