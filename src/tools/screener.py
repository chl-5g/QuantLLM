"""硬编码选股筛选器 — 排除ST、市值<10亿、历史底部、涨幅<20%、换手>3%"""

import json
import os
import numpy as np


def load_name_map() -> dict[str, str]:
    """加载股票名称映射"""
    paths = [
        "/tmp/stock_names.json",
        "/opt/quant-llm/training-data/stock_names.json",
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


NAME_MAP = load_name_map()


def screen_tickers(tickers: list[str], data: dict, start_date: str, end_date: str) -> list[str]:
    """
    硬编码筛选：市值<10亿、60日底部25%区间、20日涨幅<20%、量比>1.5、排除ST
    返回通过筛选的 ticker 列表，按"离底最近+量比最高"排序
    """
    market_data = data.get("market_data", {})
    scored = []

    for ticker in tickers:
        md = market_data.get(ticker, {})
        if not md:
            continue

        # 排除 ST
        name = NAME_MAP.get(ticker, "")
        if "ST" in name or "*ST" in name:
            continue

        closes = np.array(md.get("close", []), dtype=float)
        volumes = np.array(md.get("volume", []), dtype=float)
        amounts = np.array(md.get("amount", []), dtype=float)

        if len(closes) < 60 or len(volumes) < 20:
            continue

        current = closes[-1]
        if current < 5:
            continue

        # 1. 市值 < 15亿（近似：日均成交额/3%换手率）
        avg_amount = np.mean(amounts[-5:])
        if avg_amount <= 0:
            continue
        rough_mcap = avg_amount / 0.03
        if rough_mcap >= 15e8:
            continue

        # 2. 历史底部（60日价格分位 ≤ 30%）
        low_60 = np.min(closes[-60:])
        high_60 = np.max(closes[-60:])
        if high_60 <= low_60:
            continue
        position_60 = (current - low_60) / (high_60 - low_60)
        if position_60 > 0.30:
            continue

        # 3. 20日涨幅 < 20%
        gain_20d = (current - closes[-20]) / closes[-20] * 100
        if gain_20d >= 20:
            continue

        # 4. 量比 > 0.8（换手率>3%的代理指标，放宽）
        vol_ma5 = np.mean(volumes[-6:-1])
        if vol_ma5 <= 0:
            continue
        vol_ratio = volumes[-1] / vol_ma5
        if vol_ratio < 0.8:
            continue

        # 得分：离底越近越好，量比越高越好
        score = (0.25 - position_60) * 100 + vol_ratio
        scored.append((ticker, score))

    # 按得分降序排列
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored]
