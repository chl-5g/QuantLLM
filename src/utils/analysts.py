"""分析师配置 — A股市场专用版本"""

from src.agents.technicals import technical_analyst_agent
from src.agents.fundamentals import fundamentals_analyst_agent
from src.agents.fund_flow import fund_flow_analyst_agent
from src.agents.sentiment import sentiment_analyst_agent
from src.agents.valuation import valuation_analyst_agent
from src.agents.regime import regime_analyst_agent

ANALYST_CONFIG = {
    "technical_analyst": {
        "display_name": "技术面分析师",
        "description": "基于28个技术指标，专注A股反转效应",
        "investing_style": "超卖看多、超买看空，利用A股反转效应捕捉均值回归机会",
        "agent_func": technical_analyst_agent,
        "order": 0,
    },
    "fundamentals_analyst": {
        "display_name": "基本面分析师",
        "description": "PE/PB/ROE估值，财务健康度评估",
        "investing_style": "低估值+高ROE选股，关注安全边际",
        "agent_func": fundamentals_analyst_agent,
        "order": 1,
    },
    "fund_flow_analyst": {
        "display_name": "资金面分析师",
        "description": "北向资金、主力资金流向分析",
        "investing_style": "跟踪聪明钱，识别资金异动和主力动向",
        "agent_func": fund_flow_analyst_agent,
        "order": 2,
    },
    "sentiment_analyst": {
        "display_name": "情绪面分析师",
        "description": "市场情绪、涨跌停统计、恐慌贪婪指数",
        "investing_style": "逆向投资，市场恐慌时贪婪、贪婪时恐慌",
        "agent_func": sentiment_analyst_agent,
        "order": 3,
    },
    "valuation_analyst": {
        "display_name": "估值分析师",
        "description": "DCF/PE/PB多维度估值，计算内在价值",
        "investing_style": "只买价格远低于内在价值的标的，保留充足安全边际",
        "agent_func": valuation_analyst_agent,
        "order": 4,
    },
    "regime_analyst": {
        "display_name": "市场环境分析师",
        "description": "沪深300多维度市场环境检测",
        "investing_style": "牛市重仓、震荡半仓、熊市轻仓，顺应大势",
        "agent_func": regime_analyst_agent,
        "order": 5,
    },
}


def get_analyst_nodes() -> dict:
    """返回 {key: (node_name, agent_func)} 映射"""
    return {key: (f"{key}_agent", cfg["agent_func"])
            for key, cfg in ANALYST_CONFIG.items()}


def get_analysts_list() -> list[dict]:
    """返回分析师列表，供前端/API使用"""
    return [
        {
            "key": key,
            "display_name": cfg["display_name"],
            "description": cfg["description"],
            "investing_style": cfg["investing_style"],
            "order": cfg["order"],
        }
        for key, cfg in sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    ]
