"""组合管理 Agent — 汇总所有信号，做出最终交易决策"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class TradeDecision(BaseModel):
    action: Literal["buy", "sell", "hold"] = Field(description="交易动作")
    quantity: int = Field(description="交易股数", ge=0)
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    target_weight: float = Field(description="目标仓位权重")
    reasoning: str = Field(description="决策逻辑")


class PortfolioManagerOutput(BaseModel):
    decisions: dict[str, TradeDecision] = Field(description="每只股票的交易决策")


def portfolio_management_agent(state: AgentState, agent_id: str = "portfolio_manager"):
    """汇总所有分析师信号，做出最终买卖决策"""
    data = state["data"]
    portfolio = data["portfolio"]
    tickers = data["tickers"]
    analyst_signals = data.get("analyst_signals", {})
    show_reasoning = state["metadata"].get("show_reasoning", False)

    # 提取风控数据
    risk_agent_id = "risk_management_agent"
    risk_data = analyst_signals.get(risk_agent_id, {})
    current_prices = data.get("current_prices", {})

    # 汇总各分析师信号
    position_limits = {}
    max_shares = {}
    signals_by_ticker = {}

    for ticker in tickers:
        r = risk_data.get(ticker, {})
        position_limits[ticker] = r.get("remaining_position_limit", 0.0)
        price = current_prices.get(ticker, 0.0)
        if price > 0:
            max_shares[ticker] = int(position_limits[ticker] // price)
        else:
            max_shares[ticker] = 0

        # 收集所有非风控分析师的信号
        ticker_signals = {}
        for agent, sigs in analyst_signals.items():
            if agent.startswith("risk_management"):
                continue
            if isinstance(sigs, dict) and ticker in sigs:
                s = sigs[ticker]
                sig = s.get("signal") or s.get("regime")
                conf = s.get("confidence")
                if sig is not None and conf is not None:
                    ticker_signals[agent] = {"sig": sig, "conf": conf}

        signals_by_ticker[ticker] = ticker_signals

    # 计算允许的动作（风控约束）
    allowed_actions = _compute_allowed(tickers, current_prices, max_shares, portfolio)

    # 预填充纯 hold
    prefilled = {}
    tickers_for_llm = []
    for t in tickers:
        aa = allowed_actions.get(t, {"hold": 0})
        if set(aa.keys()) == {"hold"}:
            prefilled[t] = TradeDecision(
                action="hold", quantity=0, confidence=100,
                target_weight=0.0, reasoning="无可执行交易（资金不足或无信号支持）"
            )
        else:
            tickers_for_llm.append(t)

    if not tickers_for_llm:
        return _finalize(state, prefilled, agent_id)

    # 精简信号
    compact_signals = {}
    for t in tickers_for_llm:
        agents = {}
        for agent, payload in signals_by_ticker.get(t, {}).items():
            agents[agent] = {"sig": payload["sig"], "conf": payload["conf"]}
        compact_signals[t] = agents

    compact_allowed = {t: allowed_actions[t] for t in tickers_for_llm}

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股组合经理。根据多位分析师的信号和风控约束，为每只股票做出最终交易决策。\n"
         "规则：\n"
         "- 只能选择allowed中列出的动作和数量\n"
         "- 多位分析师一致看多+风控允许→buy\n"
         "- 多位分析师一致看空or风控不足→sell\n"
         "- 信号分歧→hold\n"
         "- A股T+1，当日买入次日才能卖，买入时需谨慎\n"
         "- 仅返回JSON，reasoning简短(≤100字)。"),
        ("human",
         "信号汇总：\n{signals}\n\n"
         "允许操作：\n{allowed}\n\n"
         "返回JSON：{{\"decisions\": {{\"TICKER\": {{"
         "\"action\":\"buy/sell/hold\",\"quantity\":int,"
         "\"confidence\":int,\"target_weight\":float,\"reasoning\":\"...\"}} }}}}"),
    ])

    prompt_data = {
        "signals": json.dumps(compact_signals, ensure_ascii=False),
        "allowed": json.dumps(compact_allowed, ensure_ascii=False),
    }
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        decisions = dict(prefilled)
        for t in tickers_for_llm:
            decisions[t] = TradeDecision(
                action="hold", quantity=0, confidence=0,
                target_weight=0.0, reasoning="LLM调用失败，默认hold"
            )
        return PortfolioManagerOutput(decisions=decisions)

    result = call_llm(prompt_msgs, PortfolioManagerOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    # 合并预填充
    merged = dict(prefilled)
    merged.update(result.decisions)

    return _finalize(state, merged, agent_id, show_reasoning)


def _compute_allowed(tickers, current_prices, max_shares, portfolio):
    """计算每只股票允许的操作和最大数量"""
    allowed = {}
    cash = float(portfolio.get("cash", 0.0))
    positions = portfolio.get("positions", {})

    for ticker in tickers:
        price = float(current_prices.get(ticker, 0.0))
        pos = positions.get(ticker, {"long": 0, "long_cost_basis": 0.0})
        long_shares = int(pos.get("long", 0) or 0)
        max_qty = int(max_shares.get(ticker, 0) or 0)

        actions = {"hold": 0}

        # 卖出
        if long_shares > 0:
            actions["sell"] = long_shares

        # 买入
        if cash > 0 and price > 0:
            max_buy_cash = int(cash // price)
            max_buy = max(0, min(max_qty, max_buy_cash))
            if max_buy > 0:
                actions["buy"] = max_buy

        # 过滤掉数量为0的动作(保留hold)
        pruned = {"hold": 0}
        for k, v in actions.items():
            if k != "hold" and v > 0:
                pruned[k] = v
        allowed[ticker] = pruned

    return allowed


def _finalize(state, decisions, agent_id, show_reasoning=False):
    """写入最终决策到 state 并返回"""
    output = {t: d.model_dump() for t, d in decisions.items()}
    state["data"]["final_decisions"] = output

    if show_reasoning:
        show_agent_reasoning(output, "组合经理 (最终决策)")

    message = HumanMessage(
        content=json.dumps(output, ensure_ascii=False),
        name=agent_id,
    )
    return {"messages": state["messages"] + [message], "data": state["data"]}
