"""市场环境分析师 — 沪深300多维度 regime 检测"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.llm import call_llm


class RegimeSignal(BaseModel):
    regime: Literal["bull", "bear", "consolidation"] = Field(description="市场环境")
    regime_score: int = Field(description="Regime分数 -5~+5", ge=-5, le=5)
    confidence: int = Field(description="置信度 0-100", ge=0, le=100)
    target_position_pct: float = Field(description="建议仓位百分比")
    max_positions: int = Field(description="最大持仓数")
    trend_score: float = Field(description="趋势得分")
    volume_score: float = Field(description="量能得分")
    volatility_score: float = Field(description="波动率得分")
    momentum_score: float = Field(description="动量得分")
    reasoning: str = Field(description="分析逻辑")


class RegimeOutput(BaseModel):
    market_regime: RegimeSignal = Field(description="市场环境评估")


def regime_analyst_agent(state: AgentState, agent_id: str = "regime_analyst_agent"):
    """检测市场环境——决定仓位水平"""
    data = state["data"]
    show_reasoning = state["metadata"].get("show_reasoning", False)

    regime_data = _get_regime_data(data)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是A股市场环境分析师。基于沪深300多维度指标判断牛/熊/震荡。\n"
         "5维度评估：趋势位置(MA120偏离)、趋势方向(MA120斜率)、量能(20d vs 60d)、\n"
         "波动率(近期vs长期)、价格动量(60d收益)。\n"
         "分数范围-5~+5：≥2牛市(95%仓位,10只)、≤-2熊市(30%仓位,3只)、其余震荡(50%仓位,5只)。"),
        ("human",
         "当前市场环境数据：\n\n{regime_data}\n\n"
         "返回JSON：{{\"market_regime\": {{\"regime\":\"bull/bear/consolidation\","
         "\"regime_score\":int,\"confidence\":int,\"target_position_pct\":float,"
         "\"max_positions\":int,\"trend_score\":float,\"volume_score\":float,"
         "\"volatility_score\":float,\"momentum_score\":float,\"reasoning\":\"...\"}} }}"),
    ])

    prompt_data = {"regime_data": json.dumps(regime_data, ensure_ascii=False, indent=2)}
    prompt_msgs = prompt.invoke(prompt_data)

    def make_default():
        return RegimeOutput(market_regime=RegimeSignal(
            regime="consolidation", regime_score=0, confidence=50,
            target_position_pct=50.0, max_positions=5,
            trend_score=0, volume_score=0, volatility_score=0, momentum_score=0,
            reasoning="市场数据不足，默认震荡市50%仓位"
        ))

    result = call_llm(prompt_msgs, RegimeOutput, agent_id, state, default_factory=make_default)
    if result is None:
        result = make_default()

    if show_reasoning:
        show_agent_reasoning(result.model_dump(), "市场环境分析师")

    state["data"]["analyst_signals"][agent_id] = result.market_regime.model_dump()

    message = HumanMessage(content=json.dumps(result.model_dump()), name=agent_id)
    return {"messages": state["messages"] + [message], "data": state["data"]}


def _get_regime_data(data: dict) -> dict:
    """从市场数据计算 regime 指标"""
    csi300 = data.get("index_data", {}).get("CSI300", {})
    closes = csi300.get("close", [])
    volumes = csi300.get("volume", [])

    if len(closes) < 120:
        return {"status": "insufficient_data", "data_points": len(closes)}

    import numpy as np
    close_arr = np.array(closes, dtype=float)
    ma120 = np.mean(close_arr[-120:])
    last_close = close_arr[-1]

    # 趋势位置
    trend_position = (last_close - ma120) / ma120 * 100

    # MA120 斜率 (20日)
    ma120_series = [np.mean(close_arr[i-120:i]) for i in range(120, len(close_arr) + 1)]
    if len(ma120_series) >= 21:
        ma120_slope = (ma120_series[-1] - ma120_series[-21]) / ma120_series[-21] * 100
    else:
        ma120_slope = 0

    # 量能确认
    vol_arr = np.array(volumes, dtype=float)
    vol_20d = np.mean(vol_arr[-20:]) if len(vol_arr) >= 20 else 1
    vol_60d = np.mean(vol_arr[-60:]) if len(vol_arr) >= 60 else vol_20d
    volume_ratio = vol_20d / vol_60d if vol_60d > 0 else 1

    # 波动率
    returns = np.diff(close_arr[-60:]) / close_arr[-61:-1]
    recent_vol = np.std(returns[-20:]) if len(returns) >= 20 else 0
    long_vol = np.std(returns) if len(returns) > 10 else recent_vol
    vol_ratio = recent_vol / long_vol if long_vol > 0 else 1

    # 价格动量
    momentum_60d = (close_arr[-1] - close_arr[-61]) / close_arr[-61] * 100 if len(close_arr) >= 61 else 0

    # 维度得分
    trend_pos_score = 1 if trend_position > 5 else (-1 if trend_position < -5 else 0)
    trend_dir_score = 1 if ma120_slope > 1 else (-1 if ma120_slope < -1 else 0)
    volume_score = 1 if volume_ratio > 1.3 else 0
    vol_score = 1 if vol_ratio < 0.7 else (-1 if vol_ratio > 1.8 else 0)
    momentum_score = 1 if momentum_60d > 15 else (-1 if momentum_60d < -15 else 0)

    regime_score = trend_pos_score + trend_dir_score + volume_score + vol_score + momentum_score

    return {
        "csi300_latest": round(float(last_close), 2),
        "ma120": round(float(ma120), 2),
        "trend_position_pct": round(float(trend_position), 2),
        "ma120_slope_pct": round(float(ma120_slope), 2),
        "volume_ratio": round(float(volume_ratio), 2),
        "volatility_ratio": round(float(vol_ratio), 2),
        "momentum_60d_pct": round(float(momentum_60d), 2),
        "dimension_scores": {
            "trend_position": trend_pos_score,
            "trend_direction": trend_dir_score,
            "volume": volume_score,
            "volatility": vol_score,
            "momentum": momentum_score,
        },
        "computed_regime_score": regime_score,
    }
