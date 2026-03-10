#!/usr/bin/env python3
"""
多市场历史行情数据爬取：商品期货 + ETF基金 + 可转债
技术指标计算 + 数据清洗，与 crawl_ashare.py 同架构

输出：
  /tmp/quant-llm/futures/basic/      期货基础行情
  /tmp/quant-llm/futures/advanced/   期货进阶因子
  /tmp/quant-llm/etf/basic/          ETF基础行情
  /tmp/quant-llm/etf/advanced/       ETF进阶因子
  /tmp/quant-llm/cbond/basic/        可转债基础行情
  /tmp/quant-llm/cbond/advanced/     可转债进阶因子

数据源：akshare（免费，无需 API key）
"""

import os
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

import akshare as ak
import pandas as pd
import numpy as np
import json
import time
import sys
from datetime import datetime

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "training-data")
START_DATE = "19900101"
END_DATE = datetime.now().strftime("%Y%m%d")
SLEEP_BETWEEN = 0.3
PROGRESS_FILE = os.path.join(OUTPUT_BASE, "multi_market_progress.json")

# ============================================================
# 技术指标（复用 crawl_ashare.py 的逻辑）
# ============================================================

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_ma(series, period):
    return series.rolling(window=period, min_periods=period).mean()

def add_technical_indicators(df):
    df = df.copy()
    df["rsi_14"] = calc_rsi(df["close"], 14)
    macd_line, signal_line, histogram = calc_macd(df["close"])
    df["macd_line"] = macd_line
    df["signal_line"] = signal_line
    df["macd_histogram"] = histogram
    df["volume_ma_5"] = calc_ma(df["volume"], 5)
    df["close_ma_20"] = calc_ma(df["close"], 20)
    return df

def clean_data(df):
    if df is None or df.empty:
        return None
    df = df[~((df["open"] == 0) & (df["close"] == 0))]
    df = df[df["volume"] > 0]
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        return None
    return df

# ============================================================
# 进度管理
# ============================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {
        "futures": {"done": [], "failed": [], "records": 0},
        "etf": {"done": [], "failed": [], "records": 0},
        "cbond": {"done": [], "failed": [], "records": 0},
    }

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False)

# ============================================================
# 通用写出函数
# ============================================================

def write_basic(filepath, symbol, df):
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "symbol": symbol,
                "date": row["date"],
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
                "amount": float(row.get("amount", 0)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_advanced(filepath, symbol, df):
    df_adv = add_technical_indicators(df)
    df_adv = df_adv.dropna(subset=["rsi_14", "macd_line", "close_ma_20"])
    with open(filepath, "w", encoding="utf-8") as f:
        for _, row in df_adv.iterrows():
            record = {
                "symbol": symbol,
                "date": row["date"],
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
                "rsi_14": round(float(row["rsi_14"]), 1),
                "macd_line": round(float(row["macd_line"]), 2),
                "signal_line": round(float(row["signal_line"]), 2),
                "macd_histogram": round(float(row["macd_histogram"]), 2),
                "volume_ma_5": int(row["volume_ma_5"]) if pd.notna(row["volume_ma_5"]) else 0,
                "close_ma_20": round(float(row["close_ma_20"]), 2),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# 1. 商品期货
# ============================================================

def crawl_futures(progress):
    print("\n" + "=" * 60)
    print("商品期货行情采集")
    print("=" * 60)

    basic_dir = os.path.join(OUTPUT_BASE, "futures", "basic")
    adv_dir = os.path.join(OUTPUT_BASE, "futures", "advanced")
    os.makedirs(basic_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    prog = progress["futures"]

    # 获取主力合约列表
    print("获取期货品种列表...")
    try:
        info = ak.futures_display_main_sina()
        symbols = info["symbol"].tolist()
        names = dict(zip(info["symbol"], info["name"]))
    except Exception as e:
        print(f"获取期货列表失败: {e}")
        return

    print(f"共 {len(symbols)} 个品种")
    already = len(prog["done"])
    if already > 0:
        print(f"已完成 {already} 个，从断点继续")

    for i, sym in enumerate(symbols):
        if sym in prog["done"] or sym in prog["failed"]:
            continue

        name = names.get(sym, "")
        try:
            df = ak.futures_zh_daily_sina(symbol=sym)
            if df is None or df.empty:
                prog["failed"].append(sym)
                print(f"  [{len(prog['done'])}/{len(symbols)}] {sym} {name} - SKIP (无数据)")
                continue

            # 标准化列名（已经是英文: date,open,high,low,close,volume,hold,settle）
            required = ["date", "open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    prog["failed"].append(sym)
                    continue

            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            df["amount"] = 0.0

            df = clean_data(df)
            if df is None:
                prog["failed"].append(sym)
                continue

            symbol_label = f"FUT.{sym}"
            write_basic(os.path.join(basic_dir, f"{symbol_label}.jsonl"), symbol_label, df)
            write_advanced(os.path.join(adv_dir, f"{symbol_label}.jsonl"), symbol_label, df)

            count = len(df)
            prog["done"].append(sym)
            prog["records"] += count
            print(f"  [{len(prog['done'])}/{len(symbols)}] {symbol_label} {name} - OK ({count}条)")

        except Exception as e:
            prog["failed"].append(sym)
            print(f"  [{len(prog['done'])}/{len(symbols)}] {sym} {name} - ERROR: {e}")

        time.sleep(SLEEP_BETWEEN)

    save_progress(progress)
    print(f"期货采集完成: {len(prog['done'])} 成功, {len(prog['failed'])} 失败, {prog['records']:,} 条记录")

# ============================================================
# 2. ETF 基金
# ============================================================

def crawl_etf(progress):
    print("\n" + "=" * 60)
    print("ETF 基金行情采集")
    print("=" * 60)

    basic_dir = os.path.join(OUTPUT_BASE, "etf", "basic")
    adv_dir = os.path.join(OUTPUT_BASE, "etf", "advanced")
    os.makedirs(basic_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    prog = progress["etf"]

    print("获取 ETF 列表（可能较慢）...")
    try:
        info = ak.fund_etf_spot_em()
        codes = info["代码"].tolist()
        names = dict(zip(info["代码"], info["名称"]))
    except Exception as e:
        print(f"获取 ETF 列表失败: {e}")
        return

    print(f"共 {len(codes)} 只 ETF")
    already = len(prog["done"])
    if already > 0:
        print(f"已完成 {already} 只，从断点继续")

    for i, code in enumerate(codes):
        if code in prog["done"] or code in prog["failed"]:
            continue

        name = names.get(code, "")
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="hfq"
            )

            if df is None or df.empty:
                prog["failed"].append(code)
                continue

            col_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"
            }
            df = df.rename(columns=col_map)

            required = ["date", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                prog["failed"].append(code)
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            if "amount" not in df.columns:
                df["amount"] = 0.0
            else:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

            df = clean_data(df)
            if df is None:
                prog["failed"].append(code)
                continue

            symbol_label = f"ETF.{code}"
            write_basic(os.path.join(basic_dir, f"{symbol_label}.jsonl"), symbol_label, df)
            write_advanced(os.path.join(adv_dir, f"{symbol_label}.jsonl"), symbol_label, df)

            count = len(df)
            prog["done"].append(code)
            prog["records"] += count

            if (len(prog["done"])) % 50 == 0 or i < 5:
                print(f"  [{len(prog['done'])}/{len(codes)}] {symbol_label} {name} - OK ({count}条)")

        except Exception as e:
            prog["failed"].append(code)

        if (len(prog["done"]) + len(prog["failed"])) % 100 == 0:
            save_progress(progress)

        time.sleep(SLEEP_BETWEEN)

    save_progress(progress)
    print(f"ETF采集完成: {len(prog['done'])} 成功, {len(prog['failed'])} 失败, {prog['records']:,} 条记录")

# ============================================================
# 3. 可转债
# ============================================================

def crawl_cbond(progress):
    print("\n" + "=" * 60)
    print("可转债行情采集")
    print("=" * 60)

    basic_dir = os.path.join(OUTPUT_BASE, "cbond", "basic")
    adv_dir = os.path.join(OUTPUT_BASE, "cbond", "advanced")
    os.makedirs(basic_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)

    prog = progress["cbond"]

    print("获取可转债列表...")
    try:
        info = ak.bond_zh_hs_cov_spot()
        symbols = info["symbol"].tolist()
        names = dict(zip(info["symbol"], info["name"]))
    except Exception as e:
        print(f"获取可转债列表失败: {e}")
        return

    print(f"共 {len(symbols)} 只可转债")
    already = len(prog["done"])
    if already > 0:
        print(f"已完成 {already} 只，从断点继续")

    for i, sym in enumerate(symbols):
        if sym in prog["done"] or sym in prog["failed"]:
            continue

        name = names.get(sym, "")
        try:
            df = ak.bond_zh_hs_cov_daily(symbol=sym)

            if df is None or df.empty:
                prog["failed"].append(sym)
                continue

            # 列名已经是英文: date, open, high, low, close, volume
            required = ["date", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                prog["failed"].append(sym)
                continue

            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            df["amount"] = 0.0

            df = clean_data(df)
            if df is None:
                prog["failed"].append(sym)
                continue

            # 标准化 symbol: CB.sh113009
            symbol_label = f"CB.{sym}"
            write_basic(os.path.join(basic_dir, f"{symbol_label}.jsonl"), symbol_label, df)
            write_advanced(os.path.join(adv_dir, f"{symbol_label}.jsonl"), symbol_label, df)

            count = len(df)
            prog["done"].append(sym)
            prog["records"] += count

            if (len(prog["done"])) % 50 == 0 or i < 5:
                print(f"  [{len(prog['done'])}/{len(symbols)}] {symbol_label} {name} - OK ({count}条)")

        except Exception as e:
            prog["failed"].append(sym)

        if (len(prog["done"]) + len(prog["failed"])) % 100 == 0:
            save_progress(progress)

        time.sleep(SLEEP_BETWEEN)

    save_progress(progress)
    print(f"可转债采集完成: {len(prog['done'])} 成功, {len(prog['failed'])} 失败, {prog['records']:,} 条记录")

# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("多市场历史数据采集（期货 + ETF + 可转债）")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    start_time = time.time()
    progress = load_progress()

    crawl_futures(progress)
    crawl_etf(progress)
    crawl_cbond(progress)

    save_progress(progress)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("全部采集完成！")
    print(f"  期货: {len(progress['futures']['done'])} 只, {progress['futures']['records']:,} 条")
    print(f"  ETF:  {len(progress['etf']['done'])} 只, {progress['etf']['records']:,} 条")
    print(f"  可转债: {len(progress['cbond']['done'])} 只, {progress['cbond']['records']:,} 条")
    total_records = sum(p["records"] for p in progress.values())
    print(f"  总记录: {total_records:,}")
    print(f"  耗时: {elapsed/60:.1f} 分钟")

    # 写统计文件
    stats = {
        "futures": {"count": len(progress["futures"]["done"]), "records": progress["futures"]["records"]},
        "etf": {"count": len(progress["etf"]["done"]), "records": progress["etf"]["records"]},
        "cbond": {"count": len(progress["cbond"]["done"]), "records": progress["cbond"]["records"]},
        "total_records": total_records,
        "elapsed_minutes": round(elapsed / 60, 1),
    }
    with open(os.path.join(OUTPUT_BASE, "multi_market_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
