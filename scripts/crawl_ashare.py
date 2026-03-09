#!/usr/bin/env python3
"""
A股全量历史行情数据爬取 + 技术指标计算 + 数据清洗
输出：
  1. /tmp/training-data/ashare/basic/     基础行情 JSONL
  2. /tmp/training-data/ashare/advanced/  进阶因子 JSONL
  3. /tmp/training-data/ashare/stats.json 统计信息

数据源：akshare（免费，无需 API key）
"""

import os
# 绕过代理，直连东方财富 API
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

import akshare as ak
import pandas as pd
import numpy as np
import json
import time
import sys
from datetime import datetime, timedelta

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "ashare")
BASIC_DIR = os.path.join(OUTPUT_BASE, "basic")
ADVANCED_DIR = os.path.join(OUTPUT_BASE, "advanced")
PROGRESS_FILE = os.path.join(OUTPUT_BASE, "progress.json")
START_DATE = "19900101"  # 尽可能早
END_DATE = datetime.now().strftime("%Y%m%d")
SLEEP_BETWEEN = 0.3  # 请求间隔，避免被封

os.makedirs(BASIC_DIR, exist_ok=True)
os.makedirs(ADVANCED_DIR, exist_ok=True)

# ============================================================
# 技术指标计算（纯 pandas 实现，不依赖 ta-lib）
# ============================================================

def calc_rsi(series, period=14):
    """计算 RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    """计算 MACD"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calc_ma(series, period):
    """简单移动平均"""
    return series.rolling(window=period, min_periods=period).mean()

def add_technical_indicators(df):
    """给 DataFrame 添加技术指标列"""
    df = df.copy()
    df["rsi_14"] = calc_rsi(df["close"], 14)
    macd_line, signal_line, histogram = calc_macd(df["close"])
    df["macd_line"] = macd_line
    df["signal_line"] = signal_line
    df["macd_histogram"] = histogram
    df["volume_ma_5"] = calc_ma(df["volume"], 5)
    df["close_ma_20"] = calc_ma(df["close"], 20)
    return df

# ============================================================
# 数据清洗
# ============================================================

def clean_data(df):
    """数据清洗"""
    if df is None or df.empty:
        return None

    # 删除全为 0 的行（停牌）
    df = df[~((df["open"] == 0) & (df["close"] == 0))]

    # 删除成交量为 0 的行
    df = df[df["volume"] > 0]

    # 删除重复日期
    df = df.drop_duplicates(subset=["date"], keep="last")

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return None

    return df

# ============================================================
# 股票代码标准化
# ============================================================

def normalize_symbol(code, market=None):
    """将 akshare 的代码转成标准格式 000001.SZ / 600519.SH"""
    code = str(code).zfill(6)
    if market:
        suffix = "SH" if "sh" in str(market).lower() else "SZ"
    else:
        if code.startswith(("6", "9")):
            suffix = "SH"
        elif code.startswith(("0", "2", "3")):
            suffix = "SZ"
        elif code.startswith("8") or code.startswith("4"):
            suffix = "BJ"
        else:
            suffix = "SZ"
    return f"{code}.{suffix}"

# ============================================================
# 进度管理
# ============================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"done": [], "failed": [], "total_records": 0}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False)

# ============================================================
# 单只股票处理
# ============================================================

def process_stock(symbol_code, progress):
    """获取并处理单只股票数据"""
    symbol = normalize_symbol(symbol_code)

    # 跳过已处理的
    if symbol in progress["done"]:
        return 0

    try:
        # akshare 获取日线数据（后复权）
        df = ak.stock_zh_a_hist(
            symbol=symbol_code,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="hfq"
        )

        if df is None or df.empty:
            progress["failed"].append(symbol)
            return 0

        # 标准化列名
        col_map = {
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"
        }
        df = df.rename(columns=col_map)

        # 确保需要的列存在
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                progress["failed"].append(symbol)
                return 0

        # 类型转换
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        else:
            df["amount"] = 0.0

        # 清洗
        df = clean_data(df)
        if df is None:
            progress["failed"].append(symbol)
            return 0

        # === 输出基础行情 ===
        basic_file = os.path.join(BASIC_DIR, f"{symbol}.jsonl")
        with open(basic_file, "w", encoding="utf-8") as f:
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
                    "close_adj": round(float(row["close"]), 2),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # === 输出进阶因子 ===
        df_adv = add_technical_indicators(df)
        # 去掉前面 NaN 行（指标需要 warmup）
        df_adv = df_adv.dropna(subset=["rsi_14", "macd_line", "close_ma_20"])

        adv_file = os.path.join(ADVANCED_DIR, f"{symbol}.jsonl")
        with open(adv_file, "w", encoding="utf-8") as f:
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

        count = len(df)
        progress["done"].append(symbol)
        progress["total_records"] += count
        return count

    except Exception as e:
        progress["failed"].append(symbol)
        return 0

# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("A股全量历史数据采集")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print("=" * 60)

    # 获取所有 A 股代码
    print("\n获取 A 股股票列表...")
    try:
        stock_info = ak.stock_zh_a_spot_em()
        codes = stock_info["代码"].tolist()
        names = dict(zip(stock_info["代码"], stock_info["名称"]))
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        # 备用方案
        try:
            stock_info = ak.stock_info_a_code_name()
            codes = stock_info["code"].tolist()
            names = dict(zip(stock_info["code"], stock_info["name"]))
        except:
            print("备用方案也失败了，退出")
            return

    # 过滤：只要主板 + 创业板 + 科创板（去掉 ST、退市、北交所等）
    filtered_codes = []
    for code in codes:
        code = str(code).zfill(6)
        if code.startswith(("0", "3", "6")):
            name = names.get(code, "")
            if "退" not in name:  # 排除退市股
                filtered_codes.append(code)

    print(f"共 {len(filtered_codes)} 只股票（主板+创业板+科创板，排除退市）")

    # 加载进度
    progress = load_progress()
    already_done = len(progress["done"])
    if already_done > 0:
        print(f"已完成 {already_done} 只，从断点继续...")

    # 逐只处理
    start_time = time.time()
    for i, code in enumerate(filtered_codes):
        symbol = normalize_symbol(code)
        if symbol in progress["done"]:
            continue

        name = names.get(code, "")
        count = process_stock(code, progress)

        done_total = len(progress["done"])
        elapsed = time.time() - start_time
        speed = (done_total - already_done) / max(elapsed, 1) * 60

        status = f"OK ({count}条)" if count > 0 else "SKIP"
        print(f"[{done_total}/{len(filtered_codes)}] {symbol} {name} - {status} "
              f"| 累计 {progress['total_records']:,} 条 | {speed:.1f} 只/分钟")

        # 定期保存进度
        if done_total % 50 == 0:
            save_progress(progress)

        time.sleep(SLEEP_BETWEEN)

    # 最终保存
    save_progress(progress)

    # 统计
    elapsed = time.time() - start_time
    stats = {
        "total_stocks": len(filtered_codes),
        "processed": len(progress["done"]),
        "failed": len(progress["failed"]),
        "total_records": progress["total_records"],
        "elapsed_minutes": round(elapsed / 60, 1),
        "output_basic": BASIC_DIR,
        "output_advanced": ADVANCED_DIR,
        "date_range": f"{START_DATE} ~ {END_DATE}",
    }

    stats_file = os.path.join(OUTPUT_BASE, "stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("采集完成！")
    print(f"  股票数: {stats['processed']}")
    print(f"  总记录: {stats['total_records']:,}")
    print(f"  失败数: {stats['failed']}")
    print(f"  耗时: {stats['elapsed_minutes']} 分钟")
    print(f"  基础行情: {BASIC_DIR}")
    print(f"  进阶因子: {ADVANCED_DIR}")

if __name__ == "__main__":
    main()
