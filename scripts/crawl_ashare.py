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
from indicators import add_all_technical_indicators

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "training-data", "ashare")
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

def add_technical_indicators(df):
    """给 DataFrame 添加技术指标列（委托给共享指标库）"""
    return add_all_technical_indicators(df)

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

def process_stock(symbol_code, progress, status="normal", name=""):
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
                    "name": name,
                    "status": status,
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
                    "name": name,
                    "status": status,
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
        except Exception:
            print("备用方案也失败了，退出")
            return

    # 过滤：只要主板 + 创业板 + 科创板（保留退市股和ST股，用于风险预警训练）
    filtered_codes = []
    stock_status = {}  # code -> (status, name)
    count_normal = 0
    count_st = 0
    count_delisting = 0
    for code in codes:
        code = str(code).zfill(6)
        if code.startswith(("0", "3", "6")):
            name = names.get(code, "")
            if "退" in name:
                status = "delisting"
                count_delisting += 1
            elif "ST" in name:
                status = "ST"
                count_st += 1
            else:
                status = "normal"
                count_normal += 1
            stock_status[code] = (status, name)
            filtered_codes.append(code)

    print(f"共 {len(filtered_codes)} 只股票（主板+创业板+科创板）")
    print(f"  正常: {count_normal}, ST: {count_st}, 退市: {count_delisting}")

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

        status, name = stock_status.get(code, ("normal", ""))
        count = process_stock(code, progress, status=status, name=name)

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

def _recalc_one_file(args):
    """处理单个 basic 文件 → advanced 文件（供多进程调用）"""
    bfile, adv_file, symbol_meta = args
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    fname = os.path.basename(bfile)
    # 断点续跑：advanced 已存在且包含新指标（cci_20）→ 跳过
    if os.path.exists(adv_file):
        try:
            with open(adv_file, "r") as _f:
                first_line = _f.readline()
            if '"cci_20"' in first_line and '"bb_position"' in first_line:
                return ("skip", fname, 0)
        except Exception:
            pass
    try:
        rows = []
        with open(bfile, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return ("fail", fname, 0)

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
        name = rows[0].get("name", "")
        status = rows[0].get("status", "normal")

        df_adv = add_technical_indicators(df)
        df_adv = df_adv.dropna(subset=["rsi_14", "macd_line", "close_ma_20"])

        # 向量化构建记录列表（避免 iterrows）
        base_meta = {"symbol": symbol, "name": name, "status": status}
        indicator_cols = [
            "date", "open", "high", "low", "close", "volume", "amount",
            "rsi_14", "macd_line", "signal_line", "macd_histogram",
            "volume_ma_5", "close_ma_20", "close_ma_5", "close_ma_10",
            "close_ma_60", "close_ma_120", "ema_12", "ema_26",
            "roc_12", "williams_r_14", "cci_20", "atr_14",
            "bb_upper", "bb_lower", "bb_width", "bb_position",
            "hv_20", "obv", "mfi_14", "vwap_proxy", "vol_change_rate",
            "adx_14", "ma_alignment", "obv_trend",
        ]
        round_map = {
            "open": 2, "high": 2, "low": 2, "close": 2, "rsi_14": 1,
            "macd_line": 2, "signal_line": 2, "macd_histogram": 2,
            "close_ma_20": 2, "close_ma_5": 2, "close_ma_10": 2,
            "close_ma_60": 2, "close_ma_120": 2, "ema_12": 2, "ema_26": 2,
            "roc_12": 2, "williams_r_14": 2, "cci_20": 2,
            "atr_14": 4, "bb_upper": 2, "bb_lower": 2,
            "bb_width": 4, "bb_position": 4, "hv_20": 2,
            "mfi_14": 1, "vwap_proxy": 2, "vol_change_rate": 2, "adx_14": 2,
        }
        int_cols = {"volume", "volume_ma_5", "obv"}

        lines = []
        records = df_adv.to_dict("records")
        for rec in records:
            row_dict = dict(base_meta)
            for col in indicator_cols:
                val = rec.get(col)
                if val is None or (isinstance(val, float) and (pd.isna(val) or val != val)):
                    continue
                if col in int_cols:
                    row_dict[col] = int(val)
                elif col in round_map:
                    row_dict[col] = round(float(val), round_map[col])
                else:
                    row_dict[col] = val
            lines.append(json.dumps(row_dict, ensure_ascii=False))

        with open(adv_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        return ("ok", fname, len(records))
    except Exception as e:
        return ("fail", fname, 0)


def recalc():
    """从 basic/ 目录读取已有数据，重新计算技术指标写入 advanced/（多进程）"""
    import glob
    import warnings
    from multiprocessing import Pool, cpu_count
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    basic_files = sorted(glob.glob(os.path.join(BASIC_DIR, "*.jsonl")))
    if not basic_files:
        print(f"basic 目录为空: {BASIC_DIR}")
        return

    os.makedirs(ADVANCED_DIR, exist_ok=True)

    # 构建任务参数
    tasks = []
    for bfile in basic_files:
        fname = os.path.basename(bfile)
        adv_file = os.path.join(ADVANCED_DIR, fname)
        tasks.append((bfile, adv_file, None))

    n_workers = min(cpu_count(), 16)
    print(f"重算模式：{len(basic_files)} 个 basic 文件, {n_workers} 进程并行", flush=True)
    start_time = time.time()

    success = 0
    skipped = 0
    failed = 0
    total_records = 0

    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_recalc_one_file, tasks, chunksize=8)):
            status, fname, n_rec = result
            if status == "skip":
                skipped += 1
                success += 1
            elif status == "ok":
                success += 1
                total_records += n_rec
            else:
                failed += 1

            if (i + 1) % 200 == 0 or i == len(tasks) - 1:
                elapsed = time.time() - start_time
                print(f"  [{i+1}/{len(tasks)}] 成功 {success}, 跳过 {skipped}, 失败 {failed}, "
                      f"累计 {total_records:,} 条, 耗时 {elapsed:.0f}s", flush=True)

    elapsed = time.time() - start_time
    print(f"\n重算完成: 成功 {success}, 跳过 {skipped}, 失败 {failed}, "
          f"总记录 {total_records:,}, 耗时 {elapsed/60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--recalc":
        recalc()
    else:
        main()
