#!/usr/bin/env python3
"""
多市场历史行情数据爬取：商品期货 + ETF基金 + 可转债
技术指标计算 + 数据清洗，与 crawl_ashare.py 同架构

输出：
  /opt/quant-llm/data/futures/basic/      期货基础行情
  /opt/quant-llm/data/futures/advanced/   期货进阶因子
  /opt/quant-llm/data/etf/basic/          ETF基础行情
  /opt/quant-llm/data/etf/advanced/       ETF进阶因子
  /opt/quant-llm/data/cbond/basic/        可转债基础行情
  /opt/quant-llm/data/cbond/advanced/     可转债进阶因子

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
from indicators import add_all_technical_indicators

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

def add_technical_indicators(df):
    """给 DataFrame 添加技术指标列（委托给共享指标库）"""
    return add_all_technical_indicators(df)

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
                "amount": float(row.get("amount", 0)),
                # 原有指标
                "rsi_14": round(float(row["rsi_14"]), 1),
                "macd_line": round(float(row["macd_line"]), 2),
                "signal_line": round(float(row["signal_line"]), 2),
                "macd_histogram": round(float(row["macd_histogram"]), 2),
                "volume_ma_5": int(row["volume_ma_5"]) if pd.notna(row["volume_ma_5"]) else 0,
                "close_ma_20": round(float(row["close_ma_20"]), 2),
                # 多周期均线
                "close_ma_5": round(float(row["close_ma_5"]), 2) if pd.notna(row.get("close_ma_5")) else None,
                "close_ma_10": round(float(row["close_ma_10"]), 2) if pd.notna(row.get("close_ma_10")) else None,
                "close_ma_60": round(float(row["close_ma_60"]), 2) if pd.notna(row.get("close_ma_60")) else None,
                "close_ma_120": round(float(row["close_ma_120"]), 2) if pd.notna(row.get("close_ma_120")) else None,
                "ema_12": round(float(row["ema_12"]), 2) if pd.notna(row.get("ema_12")) else None,
                "ema_26": round(float(row["ema_26"]), 2) if pd.notna(row.get("ema_26")) else None,
                # 动量
                "roc_12": round(float(row["roc_12"]), 2) if pd.notna(row.get("roc_12")) else None,
                "williams_r_14": round(float(row["williams_r_14"]), 2) if pd.notna(row.get("williams_r_14")) else None,
                "cci_20": round(float(row["cci_20"]), 2) if pd.notna(row.get("cci_20")) else None,
                # 波动率
                "atr_14": round(float(row["atr_14"]), 4) if pd.notna(row.get("atr_14")) else None,
                "bb_upper": round(float(row["bb_upper"]), 2) if pd.notna(row.get("bb_upper")) else None,
                "bb_lower": round(float(row["bb_lower"]), 2) if pd.notna(row.get("bb_lower")) else None,
                "bb_width": round(float(row["bb_width"]), 4) if pd.notna(row.get("bb_width")) else None,
                "bb_position": round(float(row["bb_position"]), 4) if pd.notna(row.get("bb_position")) else None,
                "hv_20": round(float(row["hv_20"]), 2) if pd.notna(row.get("hv_20")) else None,
                # 量能
                "obv": int(row["obv"]) if pd.notna(row.get("obv")) else None,
                "mfi_14": round(float(row["mfi_14"]), 1) if pd.notna(row.get("mfi_14")) else None,
                "vwap_proxy": round(float(row["vwap_proxy"]), 2) if pd.notna(row.get("vwap_proxy")) else None,
                "vol_change_rate": round(float(row["vol_change_rate"]), 2) if pd.notna(row.get("vol_change_rate")) else None,
                # 趋势
                "adx_14": round(float(row["adx_14"]), 2) if pd.notna(row.get("adx_14")) else None,
                # 派生特征
                "ma_alignment": row.get("ma_alignment", None),
                "obv_trend": row.get("obv_trend", None),
            }
            # 移除 None 值，减小文件体积
            record = {k: v for k, v in record.items() if v is not None}
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

def recalc():
    """从各市场 basic/ 目录读取已有数据，重新计算技术指标写入 advanced/"""
    import glob

    markets = ["futures", "etf", "cbond"]
    start_time = time.time()
    total_success = 0
    total_failed = 0
    total_records = 0

    for market in markets:
        basic_dir = os.path.join(OUTPUT_BASE, market, "basic")
        adv_dir = os.path.join(OUTPUT_BASE, market, "advanced")

        basic_files = sorted(glob.glob(os.path.join(basic_dir, "*.jsonl")))
        if not basic_files:
            print(f"  {market}: basic 目录为空，跳过")
            continue

        os.makedirs(adv_dir, exist_ok=True)
        print(f"\n重算 {market}: {len(basic_files)} 个文件")

        success = 0
        failed = 0
        records = 0

        for i, bfile in enumerate(basic_files):
            fname = os.path.basename(bfile)
            try:
                rows = []
                with open(bfile, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            rows.append(json.loads(line))
                if not rows:
                    failed += 1
                    continue

                df = pd.DataFrame(rows)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                if "amount" in df.columns:
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                else:
                    df["amount"] = 0.0

                df = df.sort_values("date").reset_index(drop=True)

                symbol = rows[0].get("symbol", fname.replace(".jsonl", ""))

                adv_file = os.path.join(adv_dir, fname)
                write_advanced(adv_file, symbol, df)

                success += 1
                records += len(df)
            except Exception as e:
                failed += 1
                print(f"  ERROR {fname}: {e}")

            if (i + 1) % 100 == 0 or i == len(basic_files) - 1:
                print(f"  [{i+1}/{len(basic_files)}] 成功 {success}, 失败 {failed}, "
                      f"累计 {records:,} 条")

        total_success += success
        total_failed += failed
        total_records += records

    elapsed = time.time() - start_time
    print(f"\n重算完成: 成功 {total_success}, 失败 {total_failed}, "
          f"总记录 {total_records:,}, 耗时 {elapsed/60:.1f} 分钟")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recalc":
        recalc()
    else:
        main()
