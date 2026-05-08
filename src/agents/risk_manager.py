"""风控管理 Agent — 波动率调整 + 仓位限制"""

import json
import numpy as np
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import AgentState, show_agent_reasoning


class RiskAnalysis(BaseModel):
    remaining_position_limit: float = Field(description="剩余仓位上限(元)")
    current_price: float = Field(description="当前价格")
    position_limit_pct: float = Field(description="仓位上限百分比")
    daily_volatility: float = Field(description="日波动率")
    annualized_volatility: float = Field(description="年化波动率")
    reasoning: str = Field(description="风控逻辑")


def risk_management_agent(state: AgentState, agent_id: str = "risk_management_agent"):
    """波动率调整 + 仓位上限计算"""
    data = state["data"]
    portfolio = data["portfolio"]
    tickers = data["tickers"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    # 获取 regime 信号
    regime_signal = data.get("analyst_signals", {}).get("regime_analyst_agent", {})
    target_position_pct = regime_signal.get("target_position_pct", 50.0) / 100.0
    max_positions = regime_signal.get("max_positions", 5)

    cash = float(portfolio.get("cash", 100000.0))
    total_value = cash
    for t, pos in portfolio.get("positions", {}).items():
        price = data.get("market_data", {}).get(t, {}).get("close", [0])[-1:][0] if data.get("market_data", {}).get(t, {}).get("close") else 0
        total_value += pos.get("long", 0) * price

    # 剩余可用于新仓位的资金
    position_budget = total_value * target_position_pct / max(max_positions, 1)

    risk_analysis = {}
    current_prices = {}

    for ticker in tickers:
        market_data = data.get("market_data", {}).get(ticker, {})
        closes = market_data.get("close", []) if market_data else []

        if len(closes) < 20:
            risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "position_limit_pct": 0.0,
                "daily_volatility": 0.05,
                "annualized_volatility": 0.25,
                "reasoning": "价格数据不足"
            }
            current_prices[ticker] = 0.0
            continue

        close_arr = np.array(closes, dtype=float)
        current_price = close_arr[-1]
        current_prices[ticker] = float(current_price)

        # 计算波动率
        returns = np.diff(close_arr[-60:]) / close_arr[-61:-1]
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.025
        annual_vol = daily_vol * np.sqrt(252)

        # 波动率调整仓位上限
        if annual_vol < 0.15:
            vol_multiplier = 1.25
        elif annual_vol < 0.30:
            vol_multiplier = 1.0 - (annual_vol - 0.15) * 0.5
        elif annual_vol < 0.50:
            vol_multiplier = 0.75 - (annual_vol - 0.30) * 0.5
        else:
            vol_multiplier = 0.50

        vol_multiplier = max(0.25, min(1.25, vol_multiplier))
        base_limit_pct = 0.20
        position_limit_pct = base_limit_pct * vol_multiplier

        # A股特殊限制
        position_limit_pct = min(position_limit_pct, 0.10)  # 单票不超10%

        # 可用仓位
        pos_limit_value = total_value * position_limit_pct
        remaining = min(pos_limit_value, max(cash / max(max_positions, 1), 0))
        # 不能超过 position_budget
        remaining = min(remaining, position_budget)

        risk_analysis[ticker] = {
            "remaining_position_limit": round(float(remaining), 2),
            "current_price": round(float(current_price), 2),
            "position_limit_pct": round(float(position_limit_pct), 4),
            "daily_volatility": round(daily_vol, 4),
            "annualized_volatility": round(annual_vol, 4),
            "reasoning": (
                f"年化波动率{annual_vol:.1%}，波动率乘数{vol_multiplier:.2f}，"
                f"仓位上限{position_limit_pct:.1%}，总资产{total_value:.0f}，"
                f"目标总仓位{target_position_pct:.0%}，最大持仓{max_positions}只"
            ),
        }

    state["data"]["current_prices"] = current_prices
    state["data"]["analyst_signals"][agent_id] = risk_analysis

    if show_reasoning:
        show_agent_reasoning(risk_analysis, "风控管理 Agent")

    message = HumanMessage(content=json.dumps(risk_analysis), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}
