"""估值分析师 — 多维度估值，计算内在价值与安全边际"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class ValuationSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="估值信号方向")
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    pe_valuation: float = Field(description="PE估值得分")
    pb_valuation: float = Field(description="PB估值得分")
    safety_margin: float = Field(description="安全边际 -100~100")
    intrinsic_value_gap: float = Field(description="内在价值差距百分比")
    composite_score: float = Field(description="综合估值评分 (-100~100)")
    reasoning: str = Field(description="分析逻辑")


class ValuationOutput(BaseModel):
    decisions: dict[str, ValuationSignal] = Field(description="每只股票的估值信号")


def valuation_analyst_agent(state: AgentState, agent_id: str = "valuation_analyst_agent"):
    """多维估值分析，计算安全边际"""
    data = state["data"]
    tickers = data["tickers"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    valuation_data = {}
    for ticker in tickers:
        vd = _get_valuation(ticker, data)
        if vd:
            valuation_data[ticker] = vd

    if not valuation_data:
        return _empty_result(state, tickers, agent_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股估值分析师，评估每只股票是否被低估/高估。\n"
         "评估方法：\n"
         "1. PE分位数法：当前PE vs 历史PE分位数，低于25分位→低估\n"
         "2. PB分位数法：当前PB vs 历史PB分位数，破净(PB<1)→关注\n"
         "3. 行业对比：PE vs 行业中位数\n"
         "4. 安全边际：(内在价值-当前价格)/当前价格，>20%→看多\n"
         "A股特征：银行地产PB<1常见，科技PE>50常见，需结合行业判断\n"
         "输出信号bullish/bearish/neutral，confidence 0-100。"),
        ("human",
         "以下股票的估值数据，请逐一给出信号：\n\n{valuations}\n\n"
         "返回JSON：{{\"decisions\": {{\"TICKER\": {{\"signal\":\"...\",\"confidence\":int,"
         "\"pe_valuation\":float,\"pb_valuation\":float,\"safety_margin\":float,"
         "\"intrinsic_value_gap\":float,\"composite_score\":float,\"reasoning\":\"...\"}} }}}}"),
    ])

    prompt_data = {"valuations": json.dumps(valuation_data, ensure_ascii=False, indent=2)}
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        decisions = {}
        for t in tickers:
            decisions[t] = ValuationSignal(
                signal="neutral", confidence=50, pe_valuation=0, pb_valuation=0,
                safety_margin=0, intrinsic_value_gap=0, composite_score=0,
                reasoning="估值数据不足，默认中性"
            )
        return ValuationOutput(decisions=decisions)

    result = call_llm(prompt_msgs, ValuationOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    if show_reasoning:
        show_agent_reasoning(result.model_dump(), "估值分析师")

    state["data"]["analyst_signals"][agent_id] = {
        t: s.model_dump() for t, s in result.decisions.items()
    }
    message = HumanMessage(content=json.dumps(result.model_dump()), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}


def _get_valuation(ticker: str, data: dict) -> dict | None:
    factors = data.get("factors", {}).get(ticker, {})
    prices = data.get("market_data", {}).get(ticker, {})
    if not factors:
        return None

    pe_ttm = factors.get("pe_ttm")
    pb = factors.get("pb")
    roe = factors.get("roe")
    sector_pe = factors.get("sector_pe")

    close_arr = prices.get("close", []) if prices else []
    if len(close_arr) >= 252:
        pe_history = [close_arr[i] / (factors.get("eps") or 1) for i in range(-252, 0)] if factors.get("eps") else []
    else:
        pe_history = []

    return {
        "ticker": ticker,
        "pe_ttm": pe_ttm,
        "pb": pb,
        "roe": roe,
        "sector_pe": sector_pe,
        "pe_percentile": _calc_percentile(pe_history, pe_ttm) if pe_history else None,
        "market_cap": factors.get("market_cap"),
    }


def _calc_percentile(history, current):
    if not history or current is None:
        return None
    return round(sum(1 for v in history if v <= current) / len(history) * 100, 1)


def _empty_result(state, tickers, agent_id):
    state["data"]["analyst_signals"][agent_id] = {
        t: {"signal": "neutral", "confidence": 0, "reasoning": "无估值数据"}
        for t in tickers
    }
    message = HumanMessage(content=json.dumps({"error": "no data"}), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}
