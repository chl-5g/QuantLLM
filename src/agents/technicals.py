"""技术面分析师 — A股反转效应专用"""

import json
import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class TechnicalSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"] = Field(description="技术面信号方向")
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    rsi_score: float = Field(description="RSI 反转得分")
    trend_score: float = Field(description="趋势回归得分")
    volume_score: float = Field(description="量能得分")
    volatility_score: float = Field(description="波动率得分")
    composite_score: float = Field(description="综合技术评分 (-100~100)")
    reasoning: str = Field(description="分析逻辑")


class TechnicalOutput(BaseModel):
    decisions: dict[str, TechnicalSignal] = Field(description="每只股票的技术面信号")


def technical_analyst_agent(state: AgentState, agent_id: str = "technical_analyst_agent"):
    """分析技术指标，基于A股反转效应生成信号"""
    data = state["data"]
    tickers = data["tickers"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    # 读取技术指标数据
    technical_data = {}
    for ticker in tickers:
        indicators = _get_technical_indicators(ticker, data)
        if indicators:
            technical_data[ticker] = indicators

    if not technical_data:
        return _empty_result(state, tickers, agent_id)

    # 用 LLM 综合判断
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股技术面分析师，专注小盘股反转策略。核心选股条件：\n"
         "1. 市值<10亿（小盘股，弹性大）\n"
         "2. 历史底部（价格处于60日低位区间，下跌空间有限）\n"
         "3. 涨幅<20%（尚未启动，避免追高）\n"
         "4. 换手率>3%或量比>1.5（有资金关注，流动性充足）\n\n"
         "A股所有技术因子IC均为负值——超卖看多、超买看空。\n"
         "评分逻辑：RSI<30看多(超卖)，RSI>70看空(超买)；价格远低于MA20看多(均值回归)；\n"
         "缩量筑底看多，放量冲顶看空；低波动看多，高波动看空。\n"
         "满足4个选股条件且技术面超卖的股票给予高置信度看多信号。\n"
         "输出信号：bullish(看多)/bearish(看空)/neutral(中性)，confidence 0-100。"),
        ("human",
         "以下股票的技术指标数据，请逐一给出技术面信号：\n\n{indicators}\n\n"
         "返回JSON：{{\"decisions\": {{\"TICKER\": {{\"signal\":\"...\",\"confidence\":int,"
         "\"rsi_score\":float,\"trend_score\":float,\"volume_score\":float,"
         "\"volatility_score\":float,\"composite_score\":float,\"reasoning\":\"...\"}} }}}}"),
    ])

    prompt_data = {"indicators": json.dumps(technical_data, ensure_ascii=False, indent=2)}
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        decisions = {}
        for t in tickers:
            decisions[t] = TechnicalSignal(
                signal="neutral", confidence=50, rsi_score=0, trend_score=0,
                volume_score=0, volatility_score=0, composite_score=0,
                reasoning="技术指标数据不足，默认中性"
            )
        return TechnicalOutput(decisions=decisions)

    result = call_llm(prompt_msgs, TechnicalOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    if show_reasoning:
        show_agent_reasoning(result.model_dump(), "技术面分析师")

    state["data"]["analyst_signals"][agent_id] = {
        t: s.model_dump() for t, s in result.decisions.items()
    }

    message = HumanMessage(content=json.dumps(result.model_dump()), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}


def _get_technical_indicators(ticker: str, data: dict) -> dict | None:
    """从现有技术指标数据中提取"""
    market_data = data.get("market_data", {}).get(ticker, {})
    if not market_data:
        return None

    closes = market_data.get("close", [])
    if len(closes) < 20:
        return None

    close_arr = np.array(closes, dtype=float)
    last_close = close_arr[-1]
    ma20 = np.mean(close_arr[-20:])
    ma60 = np.mean(close_arr[-60:]) if len(close_arr) >= 60 else ma20

    # RSI 14
    deltas = np.diff(close_arr[-15:])
    gains = np.sum(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    losses = -np.sum(deltas[deltas < 0]) if np.any(deltas < 0) else 0
    avg_gain = gains / 14
    avg_loss = losses / 14
    rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

    # 波动率
    returns = np.diff(close_arr[-20:]) / close_arr[-20:-1]
    volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.3

    # 量比
    volumes = market_data.get("volume", [])
    vol_arr = np.array(volumes, dtype=float)
    vol_ratio = vol_arr[-1] / np.mean(vol_arr[-20:]) if len(vol_arr) >= 20 else 1.0

    # 价格分位数
    pct_20d = (last_close - np.min(close_arr[-20:])) / (np.max(close_arr[-20:]) - np.min(close_arr[-20:]) + 1e-8)
    ma20_deviation = (last_close - ma20) / ma20 * 100

    return {
        "ticker": ticker,
        "latest_close": round(float(last_close), 2),
        "ma20": round(float(ma20), 2),
        "ma60": round(float(ma60), 2),
        "rsi_14": round(float(rsi), 1),
        "ma20_deviation_pct": round(float(ma20_deviation), 2),
        "price_pct_20d": round(float(pct_20d), 2),
        "annualized_volatility": round(float(volatility), 2),
        "volume_ratio": round(float(vol_ratio), 2),
        "trend": "up" if last_close > ma60 else "down",
    }


def _empty_result(state, tickers, agent_id):
    state["data"]["analyst_signals"][agent_id] = {
        t: {"signal": "neutral", "confidence": 0, "reasoning": "无技术指标数据"}
        for t in tickers
    }
    message = HumanMessage(content=json.dumps({"error": "no data"}), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}
