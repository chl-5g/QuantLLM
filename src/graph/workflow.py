"""LangGraph 工作流 — 多 Agent 编排"""

import os

from langgraph.graph import END, StateGraph

from src.graph.state import AgentState
from src.agents.risk_manager import risk_management_agent
from src.agents.portfolio_manager import portfolio_management_agent
from src.utils.analysts import get_analyst_nodes


def start_node(state: AgentState):
    """入口节点"""
    return state


def create_workflow(selected_analysts: list[str] | None = None):
    """创建多 Agent 工作流"""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start_node)

    analyst_nodes = get_analyst_nodes()

    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    # 添加选中的分析师节点
    for key in selected_analysts:
        if key in analyst_nodes:
            node_name, node_func = analyst_nodes[key]
            workflow.add_node(node_name, node_func)
            workflow.add_edge("start_node", node_name)

    # 风控和组合管理（始终添加）
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # 所有分析师 → 风控
    for key in selected_analysts:
        if key in analyst_nodes:
            node_name = analyst_nodes[key][0]
            workflow.add_edge(node_name, "risk_management_agent")

    # 风控 → 组合管理 → END
    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow


def run_quant_llm(
    tickers: list[str],
    start_date: str,
    end_date: str,
    initial_cash: float = 100000.0,
    show_reasoning: bool = False,
    selected_analysts: list[str] | None = None,
    model_name: str = os.environ.get("OLLAMA_GENERATION_MODEL", "qwen3.8:27b"),
    model_provider: str = "ollama",
):
    """运行 QuantLLM 多 Agent 交易系统"""
    from langchain_core.messages import HumanMessage
    from src.tools.api import load_market_data
    from src.tools.screener import screen_tickers

    # 加载数据
    loaded = load_market_data(tickers, start_date, end_date)

    # 硬编码选股筛选
    tickers_original = list(tickers)
    tickers = screen_tickers(tickers, loaded, start_date, end_date)
    dropped = set(tickers_original) - set(tickers)
    if dropped:
        print(f"\n  硬编码筛选: {len(tickers_original)} → {len(tickers)} 只")
        for t in sorted(dropped):
            print(f"    ✗ {t} 淘汰（不满足：市值<15亿、60日底部30%%、涨幅<20%%、量比>0.8、非ST）")
    if not tickers:
        print("  ⚠ 无股票通过筛选")
        return {"decisions": {}, "analyst_signals": {}}

    # 构建初始持仓
    positions = {}
    for t in tickers:
        closes = loaded["market_data"].get(t, {}).get("close", [])
        positions[t] = {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0}

    portfolio = {
        "cash": initial_cash,
        "positions": positions,
    }

    # 创建工作流
    workflow = create_workflow(selected_analysts)
    agent = workflow.compile()

    final_state = agent.invoke({
        "messages": [
            HumanMessage(content="基于提供的数据做出交易决策。")
        ],
        "data": {
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
            "market_data": loaded.get("market_data", {}),
            "factors": loaded.get("factors", {}),
            "fund_flows": loaded.get("fund_flows", {}),
            "market_sentiment": loaded.get("market_sentiment", {}),
            "index_data": loaded.get("index_data", {}),
        },
        "metadata": {
            "show_reasoning": show_reasoning,
            "model_name": model_name,
            "model_provider": model_provider,
        },
    })

    return {
        "decisions": final_state["data"].get("final_decisions", {}),
        "analyst_signals": final_state["data"].get("analyst_signals", {}),
    }
