"""基本面分析师 — PE/PB/ROE 估值与财务健康度"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class FundamentalSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="基本面信号方向")
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    pe_score: float = Field(description="PE估值得分")
    pb_score: float = Field(description="PB估值得分")
    roe_score: float = Field(description="ROE盈利能力得分")
    growth_score: float = Field(description="成长性得分")
    composite_score: float = Field(description="综合基本面评分 (-100~100)")
    reasoning: str = Field(description="分析逻辑")


class FundamentalOutput(BaseModel):
    decisions: dict[str, FundamentalSignal] = Field(description="每只股票的基本面信号")


def fundamentals_analyst_agent(state: AgentState, agent_id: str = "fundamentals_analyst_agent"):
    """分析基本面数据，基于PE/PB/ROE生成信号"""
    data = state["data"]
    tickers = data["tickers"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    fundamental_data = {}
    for ticker in tickers:
        fd = _get_fundamental_data(ticker, data)
        if fd:
            fundamental_data[ticker] = fd

    if not fundamental_data:
        return _empty_result(state, tickers, agent_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股基本面分析师。评估标准：\n"
         "- PE<15且行业合理→低估看多；PE>50或负值→高估看空\n"
         "- PB<1→破净，可能看多（需结合ROE）；PB>5→高估\n"
         "- ROE>15%→盈利能力强；ROE<5%→盈利能力弱\n"
         "- 营收/利润增长>20%→高成长加分\n"
         "输出信号bullish/bearish/neutral，confidence 0-100。"),
        ("human",
         "以下股票的基本面数据，请逐一给出信号：\n\n{fundamentals}\n\n"
         "返回JSON：{{\"decisions\": {{\"TICKER\": {{\"signal\":\"...\",\"confidence\":int,"
         "\"pe_score\":float,\"pb_score\":float,\"roe_score\":float,\"growth_score\":float,"
         "\"composite_score\":float,\"reasoning\":\"...\"}} }}}}"),
    ])

    prompt_data = {"fundamentals": json.dumps(fundamental_data, ensure_ascii=False, indent=2)}
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        decisions = {}
        for t in tickers:
            decisions[t] = FundamentalSignal(
                signal="neutral", confidence=50, pe_score=0, pb_score=0,
                roe_score=0, growth_score=0, composite_score=0,
                reasoning="基本面数据不足，默认中性"
            )
        return FundamentalOutput(decisions=decisions)

    result = call_llm(prompt_msgs, FundamentalOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    if show_reasoning:
        show_agent_reasoning(result.model_dump(), "基本面分析师")

    state["data"]["analyst_signals"][agent_id] = {
        t: s.model_dump() for t, s in result.decisions.items()
    }
    message = HumanMessage(content=json.dumps(result.model_dump()), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}


def _get_fundamental_data(ticker: str, data: dict) -> dict | None:
    factors = data.get("factors", {}).get(ticker, {})
    if not factors:
        return None
    return {
        "ticker": ticker,
        "pe_ttm": factors.get("pe_ttm", None),
        "pb": factors.get("pb", None),
        "roe": factors.get("roe", None),
        "market_cap": factors.get("market_cap", None),
        "revenue_growth": factors.get("revenue_growth", None),
        "profit_growth": factors.get("profit_growth", None),
        "debt_ratio": factors.get("debt_ratio", None),
    }


def _empty_result(state, tickers, agent_id):
    state["data"]["analyst_signals"][agent_id] = {
        t: {"signal": "neutral", "confidence": 0, "reasoning": "无基本面数据"}
        for t in tickers
    }
    message = HumanMessage(content=json.dumps({"error": "no data"}), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}
