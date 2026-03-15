#!/usr/bin/env python3
"""
资金流向数据爬取
输出：
  1. training-data/fund_flow/sector_daily/   板块资金流 JSONL（行业/概念/地域，多时间维度）
  2. training-data/fund_flow/stock_daily/    个股资金流排名 JSONL（多时间维度）
  3. training-data/fund_flow/market_daily.jsonl  大盘资金流历史数据

数据源：akshare（东方财富 API，免费，无需 API key）
"""

import os
# 绕过代理，直连东方财富 API
for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    os.environ.pop(k, None)

import akshare as ak
import pandas as pd
import json
import time
import re
from datetime import datetime

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "training-data", "fund_flow")
SECTOR_DIR = os.path.join(OUTPUT_BASE, "sector_daily")
STOCK_DIR = os.path.join(OUTPUT_BASE, "stock_daily")
MARKET_FILE = os.path.join(OUTPUT_BASE, "market_daily.jsonl")
PROGRESS_FILE = os.path.join(OUTPUT_BASE, "progress.json")
SLEEP_BETWEEN = 0.3
TODAY = datetime.now().strftime("%Y-%m-%d")

os.makedirs(SECTOR_DIR, exist_ok=True)
os.makedirs(STOCK_DIR, exist_ok=True)

# ============================================================
# 中文列名 → snake_case 映射
# ============================================================

# 通用映射规则（处理前缀如 "今日", "3日", "5日", "10日"）
_CN_TO_EN = {
    "序号": "rank",
    "代码": "code",
    "名称": "name",
    "最新价": "latest_price",
    "收盘价": "close",
    "涨跌幅": "change_pct",
    "日期": "date",
    "是否净流入": "is_net_inflow",
    "所属板块": "sector",
    # 上证/深证
    "上证-收盘价": "sh_close",
    "上证-涨跌幅": "sh_change_pct",
    "深证-收盘价": "sz_close",
    "深证-涨跌幅": "sz_change_pct",
    # 资金流核心字段（无前缀版本）
    "主力净流入-净额": "main_net_inflow",
    "主力净流入-净占比": "main_net_inflow_pct",
    "超大单净流入-净额": "super_large_net_inflow",
    "超大单净流入-净占比": "super_large_net_inflow_pct",
    "大单净流入-净额": "large_net_inflow",
    "大单净流入-净占比": "large_net_inflow_pct",
    "中单净流入-净额": "medium_net_inflow",
    "中单净流入-净占比": "medium_net_inflow_pct",
    "小单净流入-净额": "small_net_inflow",
    "小单净流入-净占比": "small_net_inflow_pct",
    "主力净流入最大股": "main_inflow_top_stock",
    "主力净流入最大股代码": "main_inflow_top_stock_code",
}

# 带时间前缀的映射（今日/3日/5日/10日）
_TIME_PREFIX_MAP = {
    "今日": "today",
    "3日": "3d",
    "5日": "5d",
    "10日": "10d",
}


def translate_column(col: str) -> str:
    """将中文列名翻译为 snake_case 英文"""
    # 直接匹配
    if col in _CN_TO_EN:
        return _CN_TO_EN[col]

    # 带时间前缀的列名，如 "今日主力净流入-净额" → "today_main_net_inflow"
    for cn_prefix, en_prefix in _TIME_PREFIX_MAP.items():
        if col.startswith(cn_prefix):
            suffix = col[len(cn_prefix):]
            if suffix in _CN_TO_EN:
                return f"{en_prefix}_{_CN_TO_EN[suffix]}"
            # 处理如 "今日涨跌幅"
            if suffix == "涨跌幅":
                return f"{en_prefix}_change_pct"

    # 排行榜类列名，如 "今日排行榜-主力净占比"
    m = re.match(r"(今日|[35]日|10日)排行榜-(.+)", col)
    if m:
        prefix = _TIME_PREFIX_MAP.get(m.group(1), m.group(1))
        rest = m.group(2)
        sub_map = {
            "主力净占比": "main_net_pct",
            "今日排名": "rank",
            "今日涨跌": "change_pct",
            "5日排名": "rank",
            "5日涨跌": "change_pct",
            "10日排名": "rank",
            "10日涨跌": "change_pct",
        }
        en_rest = sub_map.get(rest, rest)
        return f"{prefix}_ranking_{en_rest}"

    # 兜底：去掉特殊字符，用下划线
    col_clean = col.replace("-", "_").replace(" ", "_")
    return col_clean


def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """批量翻译 DataFrame 列名"""
    new_cols = {}
    seen = set()
    for col in df.columns:
        translated = translate_column(col)
        # 避免重复列名
        if translated in seen:
            translated = f"{translated}_2"
        seen.add(translated)
        new_cols[col] = translated
    return df.rename(columns=new_cols)


# ============================================================
# 数值处理
# ============================================================

def round_numeric(val, decimals=2):
    """将数值四舍五入，非数值原样返回"""
    if isinstance(val, float):
        if pd.isna(val):
            return None
        return round(val, decimals)
    return val


def df_to_records(df: pd.DataFrame, extra_fields: dict = None) -> list:
    """将 DataFrame 转为 JSONL 记录列表，数值四舍五入"""
    records = []
    for _, row in df.iterrows():
        record = {}
        if extra_fields:
            record.update(extra_fields)
        for col in df.columns:
            val = row[col]
            if isinstance(val, (pd.Timestamp, datetime)):
                val = val.strftime("%Y-%m-%d")
            elif hasattr(val, "isoformat"):  # datetime.date
                val = str(val)
            elif isinstance(val, float):
                val = round_numeric(val)
            elif isinstance(val, (int,)):
                pass
            else:
                val = str(val) if val is not None and not pd.isna(val) else None
            record[col] = val
        records.append(record)
    return records


def write_jsonl(records: list, filepath: str):
    """写入 JSONL 文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  -> {filepath} ({len(records)} records)")


# ============================================================
# 进度管理
# ============================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"last_run": None, "tasks_done": [], "total_records": 0, "errors": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ============================================================
# 板块资金流
# ============================================================

def crawl_sector_fund_flow(progress):
    """爬取板块资金流排名（行业/概念/地域 x 今日/5日/10日）"""
    print("\n" + "=" * 50)
    print("板块资金流排名")
    print("=" * 50)

    sector_types = ["行业资金流", "概念资金流", "地域资金流"]
    sector_type_en = {"行业资金流": "industry", "概念资金流": "concept", "地域资金流": "region"}
    indicators = ["今日", "5日", "10日"]
    indicator_en = {"今日": "today", "5日": "5d", "10日": "10d"}

    for sector_type in sector_types:
        for indicator in indicators:
            task_key = f"sector_{sector_type_en[sector_type]}_{indicator_en[indicator]}"
            if task_key in progress["tasks_done"]:
                print(f"  [SKIP] {sector_type} {indicator} (already done)")
                continue

            try:
                print(f"  Fetching: {sector_type} {indicator} ...")
                df = ak.stock_sector_fund_flow_rank(
                    indicator=indicator, sector_type=sector_type
                )

                if df is None or df.empty:
                    print(f"  [WARN] {sector_type} {indicator}: empty DataFrame")
                    progress["errors"].append(f"{task_key}: empty")
                    continue

                # 翻译列名
                df = translate_columns(df)

                # 添加元数据
                records = df_to_records(df, extra_fields={
                    "crawl_date": TODAY,
                    "sector_type": sector_type_en[sector_type],
                    "timeframe": indicator_en[indicator],
                })

                filename = f"{TODAY}_{sector_type_en[sector_type]}_{indicator_en[indicator]}.jsonl"
                filepath = os.path.join(SECTOR_DIR, filename)
                write_jsonl(records, filepath)

                progress["tasks_done"].append(task_key)
                progress["total_records"] += len(records)
                save_progress(progress)

            except Exception as e:
                print(f"  [ERROR] {sector_type} {indicator}: {e}")
                progress["errors"].append(f"{task_key}: {str(e)[:200]}")

            time.sleep(SLEEP_BETWEEN)


# ============================================================
# 个股资金流排名
# ============================================================

def crawl_stock_fund_flow(progress):
    """爬取个股资金流排名（今日/3日/5日/10日）"""
    print("\n" + "=" * 50)
    print("个股资金流排名")
    print("=" * 50)

    indicators = ["今日", "3日", "5日", "10日"]
    indicator_en = {"今日": "today", "3日": "3d", "5日": "5d", "10日": "10d"}

    for indicator in indicators:
        task_key = f"stock_rank_{indicator_en[indicator]}"
        if task_key in progress["tasks_done"]:
            print(f"  [SKIP] {indicator} (already done)")
            continue

        try:
            print(f"  Fetching: individual stock fund flow rank {indicator} ...")
            df = ak.stock_individual_fund_flow_rank(indicator=indicator)

            if df is None or df.empty:
                print(f"  [WARN] {indicator}: empty DataFrame")
                progress["errors"].append(f"{task_key}: empty")
                continue

            # 翻译列名
            df = translate_columns(df)

            records = df_to_records(df, extra_fields={
                "crawl_date": TODAY,
                "timeframe": indicator_en[indicator],
            })

            filename = f"{TODAY}_{indicator_en[indicator]}.jsonl"
            filepath = os.path.join(STOCK_DIR, filename)
            write_jsonl(records, filepath)

            progress["tasks_done"].append(task_key)
            progress["total_records"] += len(records)
            save_progress(progress)

        except Exception as e:
            print(f"  [ERROR] stock rank {indicator}: {e}")
            progress["errors"].append(f"{task_key}: {str(e)[:200]}")

        time.sleep(SLEEP_BETWEEN)


# ============================================================
# 大盘资金流
# ============================================================

def crawl_market_fund_flow(progress):
    """爬取大盘资金流历史数据（上证+深证）"""
    print("\n" + "=" * 50)
    print("大盘资金流（历史）")
    print("=" * 50)

    task_key = "market_daily"
    if task_key in progress["tasks_done"]:
        print("  [SKIP] market daily (already done)")
        return

    try:
        print("  Fetching: market fund flow ...")
        df = ak.stock_market_fund_flow()

        if df is None or df.empty:
            print("  [WARN] market fund flow: empty DataFrame")
            progress["errors"].append(f"{task_key}: empty")
            return

        # 翻译列名
        df = translate_columns(df)

        records = df_to_records(df, extra_fields={"crawl_date": TODAY})

        write_jsonl(records, MARKET_FILE)

        progress["tasks_done"].append(task_key)
        progress["total_records"] += len(records)
        save_progress(progress)

    except Exception as e:
        print(f"  [ERROR] market fund flow: {e}")
        progress["errors"].append(f"{task_key}: {str(e)[:200]}")


# ============================================================
# 主流程
# ============================================================

def main():
    start_time = time.time()

    print("=" * 60)
    print("资金流向数据采集")
    print(f"日期: {TODAY}")
    print(f"输出: {OUTPUT_BASE}")
    print("=" * 60)

    # 加载进度（同一天内可断点续跑）
    progress = load_progress()
    if progress["last_run"] != TODAY:
        # 新的一天，重置进度
        progress = {"last_run": TODAY, "tasks_done": [], "total_records": 0, "errors": []}

    already_done = len(progress["tasks_done"])
    if already_done > 0:
        print(f"今日已完成 {already_done} 个任务，从断点继续...")

    # 1. 板块资金流
    crawl_sector_fund_flow(progress)

    # 2. 个股资金流排名
    crawl_stock_fund_flow(progress)

    # 3. 大盘资金流
    crawl_market_fund_flow(progress)

    # 最终统计
    elapsed = time.time() - start_time
    save_progress(progress)

    print("\n" + "=" * 60)
    print("采集完成！")
    print(f"  完成任务: {len(progress['tasks_done'])}")
    print(f"  总记录数: {progress['total_records']:,}")
    print(f"  错误数: {len(progress['errors'])}")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  板块资金流: {SECTOR_DIR}")
    print(f"  个股资金流: {STOCK_DIR}")
    print(f"  大盘资金流: {MARKET_FILE}")

    if progress["errors"]:
        print("\n  错误详情:")
        for err in progress["errors"]:
            print(f"    - {err}")

    print("\n" + "-" * 60)
    print("NOTE: 可将此脚本加入 run.sh 定时执行：")
    print('  python3 "$SCRIPT_DIR/crawl_fund_flow.py"')
    print("-" * 60)


if __name__ == "__main__":
    main()
