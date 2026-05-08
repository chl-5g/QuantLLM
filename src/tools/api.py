"""数据获取工具 — 基于本地缓存 + akshare"""

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, '/opt/quant-llm/scripts')


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取A股历史行情，优先本地缓存"""
    cache_dirs = [
        "/opt/quant-llm/training-data/ashare/advanced",
        "/opt/quant-llm/training-data/ashare/basic",
    ]

    for cache_dir in cache_dirs:
        cache_file = os.path.join(cache_dir, f"{ticker}.jsonl")
        if not os.path.exists(cache_file):
            # 尝试不同后缀
            for ext in [".jsonl", ".json"]:
                alt = os.path.join(cache_dir, f"{ticker}{ext}")
                if os.path.exists(alt):
                    cache_file = alt
                    break
            else:
                continue

        try:
            rows = []
            with open(cache_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        row_date = row.get("date", "")[:10]
                        if start_date <= row_date <= end_date:
                            close = row.get("close_adj", row.get("close", 0))
                            rows.append({
                                "date": row_date,
                                "open": row.get("open", 0),
                                "high": row.get("high", 0),
                                "low": row.get("low", 0),
                                "close": float(close),
                                "volume": row.get("volume", 0),
                            })
            if rows:
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                df.sort_values("date", inplace=True)
                return df
        except Exception:
            pass

    return pd.DataFrame()


def get_multi_price_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """批量获取多只股票行情"""
    result = {}
    for t in tickers:
        df = get_price_data(t, start_date, end_date)
        if not df.empty:
            result[t] = df
    return result


def get_stock_factors(tickers: list[str]) -> dict[str, dict]:
    """读取股票因子数据"""
    factor_file = "/opt/quant-llm/training-data/factors/stock_factors_latest.json"
    factors = {}
    try:
        if os.path.exists(factor_file):
            with open(factor_file) as f:
                all_factors = json.load(f)
            for t in tickers:
                if t in all_factors:
                    factors[t] = all_factors[t]
    except Exception as e:
        print(f"  [factors] 读取失败: {e}")
    return factors


def get_fund_flows(tickers: list[str]) -> dict[str, dict]:
    """获取个股资金流向"""
    flow_dir = "/opt/quant-llm/training-data/fund_flow"
    flows = {}
    for t in tickers:
        for suffix in ["", ".SH", ".SZ"]:
            fpath = os.path.join(flow_dir, f"{t}{suffix}.json")
            if os.path.exists(fpath):
                try:
                    with open(fpath) as f:
                        flows[t] = json.load(f)
                    break
                except Exception:
                    pass
    return flows


def get_market_sentiment() -> dict:
    """获取市场情绪（基于本地数据估算）"""
    return {
        "fear_greed_index": 50,
        "limit_up_count": 0,
        "limit_down_count": 0,
        "advance_decline_ratio": 1.0,
    }


def get_index_data(index_code: str = "CSI300", start_date: str = None, end_date: str = None) -> dict:
    """获取沪深300指数行情"""
    index_file = "/opt/quant-llm/training-data/index/csi300.jsonl"
    if not os.path.exists(index_file):
        return {"close": [], "volume": [], "dates": []}

    rows = []
    try:
        with open(index_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                d = row.get("date", "")[:10]
                if start_date and d < start_date:
                    continue
                if end_date and d > end_date:
                    continue
                rows.append({"date": d, "close": float(row.get("close", 0)), "volume": row.get("volume", 0)})
    except Exception:
        pass

    if rows:
        return {
            "close": [r["close"] for r in rows],
            "volume": [r["volume"] for r in rows],
            "dates": [r["date"] for r in rows],
        }
    return {"close": [], "volume": [], "dates": []}


def load_market_data(tickers: list[str], start_date: str, end_date: str) -> dict:
    """一站式加载所有需要的数据"""
    print(f"\n加载 {len(tickers)} 只股票数据 [{start_date} ~ {end_date}]...")

    print("  [1/5] 获取行情数据...")
    price_dfs = get_multi_price_data(tickers, start_date, end_date)
    market_data = {}
    for t, df in price_dfs.items():
        if not df.empty:
            market_data[t] = {
                "open": df["open"].tolist(),
                "close": df["close"].tolist(),
                "high": df["high"].tolist(),
                "low": df["low"].tolist(),
                "volume": df["volume"].tolist(),
                "dates": [str(d).split("T")[0] for d in df["date"]],
            }

    print("  [2/5] 获取因子数据...")
    factors = get_stock_factors(tickers)

    print("  [3/5] 获取资金流数据...")
    fund_flows = get_fund_flows(tickers)

    print("  [4/5] 获取市场情绪...")
    sentiment = get_market_sentiment()

    print("  [5/5] 获取指数数据...")
    index_data = {"CSI300": get_index_data("CSI300", start_date, end_date)}

    print(f"  完成: {len(market_data)}只有行情, {len(factors)}只有因子\n")
    return {
        "market_data": market_data,
        "factors": factors,
        "fund_flows": fund_flows,
        "market_sentiment": sentiment,
        "index_data": index_data,
    }
