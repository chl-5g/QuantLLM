"""情绪面分析师 — 市场情绪与逆向投资"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class SentimentSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="情绪面信号方向")
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    fear_greed_score: float = Field(description="恐慌贪婪得分")
    limit_up_down_score: float = Field(description="涨跌停统计得分")
    turnover_score: float = Field(description="换手率情绪得分")
    composite_score: float = Field(description="综合情绪评分 (-100~100)")
    reasoning: str = Field(description="分析逻辑")


class SentimentOutput(BaseModel):
    decisions: dict[str, SentimentSignal] = Field(description="每只股票的情绪面信号")


def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """分析市场情绪，逆向投资——恐慌时贪婪，贪婪时恐慌"""
    data = state["data"]
    tickers = data["tickers"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    sentiment_data = {}
    for ticker in tickers:
        sd = _get_sentiment(ticker, data)
        if sd:
            sentiment_data[ticker] = sd

    if not sentiment_data:
        return _empty_result(state, tickers, agent_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股情绪面分析师，信奉逆向投资。\n"
         "- 极度恐慌(恐慌指数>70)→看多机会（别人恐惧我贪婪）\n"
         "- 极度贪婪(贪婪指数>70)→看空信号（别人贪婪我恐惧）\n"
         "- 大量跌停→恐慌蔓延→可能接近底部→谨慎看多\n"
         "- 大量涨停→过度乐观→可能接近顶部→谨慎看空\n"
         "- 换手率异常高+下跌→恐慌抛售→看多\n"
         "输出信号bullish/bearish/neutral，confidence 0-100。"),
        ("human",
         "以下股票的情绪面数据，请逐一给出信号：\n\n{sentiments}\n\n"
         "返回JSON：{{\"decisions\": {{\"TICKER\": {{\"signal\":\"...\",\"confidence\":int,"
         "\"fear_greed_score\":float,\"limit_up_down_score\":float,\"turnover_score\":float,"
         "\"composite_score\":float,\"reasoning\":\"...\"}} }}}}"),
    ])

    prompt_data = {"sentiments": json.dumps(sentiment_data, ensure_ascii=False, indent=2)}
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        decisions = {}
        for t in tickers:
            decisions[t] = SentimentSignal(
                signal="neutral", confidence=50, fear_greed_score=0,
                limit_up_down_score=0, turnover_score=0, composite_score=0,
                reasoning="情绪面数据不足，默认中性"
            )
        return SentimentOutput(decisions=decisions)

    result = call_llm(prompt_msgs, SentimentOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    if show_reasoning:
        show_agent_reasoning(result.model_dump(), "情绪面分析师")

    state["data"]["analyst_signals"][agent_id] = {
        t: s.model_dump() for t, s in result.decisions.items()
    }
    message = HumanMessage(content=json.dumps(result.model_dump()), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}


def _get_sentiment(ticker: str, data: dict) -> dict | None:
    mkt = data.get("market_sentiment", {})
    stock_sent = data.get("stock_sentiment", {}).get(ticker, {})
    return {
        "ticker": ticker,
        "fear_greed_index": mkt.get("fear_greed_index", 50),
        "limit_up_count": mkt.get("limit_up_count", 0),
        "limit_down_count": mkt.get("limit_down_count", 0),
        "advance_decline_ratio": mkt.get("advance_decline_ratio", 1.0),
        "turnover_rate": stock_sent.get("turnover_rate", None),
        "change_pct": stock_sent.get("change_pct", None),
        "volume_ratio": stock_sent.get("volume_ratio", 1.0),
    }


def _empty_result(state, tickers, agent_id):
    state["data"]["analyst_signals"][agent_id] = {
        t: {"signal": "neutral", "confidence": 0, "reasoning": "无情绪面数据"}
        for t in tickers
    }
    message = HumanMessage(content=json.dumps({"error": "no data"}), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}
