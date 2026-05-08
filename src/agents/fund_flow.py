"""资金面分析师 — 北向资金、主力资金流向"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class FundFlowSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="资金面信号方向")
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    northbound_score: float = Field(description="北向资金得分")
    main_force_score: float = Field(description="主力资金得分")
    institutional_score: float = Field(description="机构资金得分")
    composite_score: float = Field(description="综合资金面评分 (-100~100)")
    reasoning: str = Field(description="分析逻辑")


class FundFlowOutput(BaseModel):
    decisions: dict[str, FundFlowSignal] = Field(description="每只股票的资金面信号")


def fund_flow_analyst_agent(state: AgentState, agent_id: str = "fund_flow_analyst_agent"):
    """分析资金流向，跟踪聪明钱"""
    data = state["data"]
    tickers = data["tickers"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    flow_data = {}
    for ticker in tickers:
        fd = _get_fund_flow(ticker, data)
        if fd:
            flow_data[ticker] = fd

    if not flow_data:
        return _empty_result(state, tickers, agent_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股资金面分析师，跟踪聪明钱的动向。\n"
         "- 北向资金连续净流入→看多信号（外资看好）\n"
         "- 主力资金持续净流入→看多信号（大资金布局）\n"
         "- 主力流出而北向流入→中性（分歧）\n"
         "- 北向大幅流出→看空信号\n"
         "输出信号bullish/bearish/neutral，confidence 0-100。"),
        ("human",
         "以下股票的资金流向数据，请逐一给出信号：\n\n{fund_flows}\n\n"
         "返回JSON：{{\"decisions\": {{\"TICKER\": {{\"signal\":\"...\",\"confidence\":int,"
         "\"northbound_score\":float,\"main_force_score\":float,\"institutional_score\":float,"
         "\"composite_score\":float,\"reasoning\":\"...\"}} }}}}"),
    ])

    prompt_data = {"fund_flows": json.dumps(flow_data, ensure_ascii=False, indent=2)}
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        decisions = {}
        for t in tickers:
            decisions[t] = FundFlowSignal(
                signal="neutral", confidence=50, northbound_score=0,
                main_force_score=0, institutional_score=0, composite_score=0,
                reasoning="资金流数据不足，默认中性"
            )
        return FundFlowOutput(decisions=decisions)

    result = call_llm(prompt_msgs, FundFlowOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    if show_reasoning:
        show_agent_reasoning(result.model_dump(), "资金面分析师")

    state["data"]["analyst_signals"][agent_id] = {
        t: s.model_dump() for t, s in result.decisions.items()
    }
    message = HumanMessage(content=json.dumps(result.model_dump()), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}


def _get_fund_flow(ticker: str, data: dict) -> dict | None:
    flows = data.get("fund_flows", {}).get(ticker, {})
    if not flows:
        return None
    return {
        "ticker": ticker,
        "northbound_net_5d": flows.get("northbound_net_5d", None),
        "main_force_net_today": flows.get("main_force_net_today", None),
        "main_force_net_5d": flows.get("main_force_net_5d", None),
        "super_large_net": flows.get("super_large_net", None),
        "large_net": flows.get("large_net", None),
        "medium_net": flows.get("medium_net", None),
        "small_net": flows.get("small_net", None),
        "turnover_rate": flows.get("turnover_rate", None),
    }


def _empty_result(state, tickers, agent_id):
    state["data"]["analyst_signals"][agent_id] = {
        t: {"signal": "neutral", "confidence": 0, "reasoning": "无资金流数据"}
        for t in tickers
    }
    message = HumanMessage(content=json.dumps({"error": "no data"}), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}
