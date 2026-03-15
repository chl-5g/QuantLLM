#!/usr/bin/env python3
"""
因子 IC（信息系数）分析

对所有技术因子计算与未来 5/10/20 日收益率的 Rank IC，
输出每个因子的 IC 均值、IC 标准差、IC_IR（信息比率）、正IC占比。

用法:
  python factor_ic_analysis.py                    # 默认分析全部 A 股
  python factor_ic_analysis.py --max-stocks 500   # 限制股票数量加速
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg, PROJECT_ROOT

# 要分析的因子列表（对应 JSONL 中的字段名）
FACTORS = [
    "rsi_14",
    "macd_histogram",
    "macd_line",
    "cci_20",
    "mfi_14",
    "bb_position",
    "williams_r_14",
    "roc_12",
    "adx_14",
    "atr_14",
    "hv_20",
    "obv",
    "vol_change_rate",
    "volume_ma_5",
]

# 衍生因子（需要计算）
DERIVED_FACTORS = [
    "ma20_diff_pct",      # (close - MA20) / MA20
    "volume_ratio",       # volume / volume_ma_5
    "trend_5d",           # 5日收益率
    "trend_20d",          # 20日收益率（反转因子）
]

FORWARD_PERIODS = [5, 10, 20]


def load_jsonl(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("date", ""))
    return rows


def compute_derived(rows, idx):
    """计算衍生因子"""
    row = rows[idx]
    derived = {}

    # MA20 偏离度
    ma20 = row.get("close_ma_20", 0)
    if ma20 and ma20 > 0:
        derived["ma20_diff_pct"] = (row["close"] - ma20) / ma20 * 100
    else:
        derived["ma20_diff_pct"] = None

    # 量比
    vol_ma5 = row.get("volume_ma_5", 0)
    if vol_ma5 and vol_ma5 > 0:
        derived["volume_ratio"] = row["volume"] / vol_ma5
    else:
        derived["volume_ratio"] = None

    # 5日趋势
    if idx >= 5 and rows[idx - 5]["close"] > 0:
        derived["trend_5d"] = (row["close"] - rows[idx - 5]["close"]) / rows[idx - 5]["close"] * 100
    else:
        derived["trend_5d"] = None

    # 20日反转因子
    if idx >= 20 and rows[idx - 20]["close"] > 0:
        derived["trend_20d"] = (row["close"] - rows[idx - 20]["close"]) / rows[idx - 20]["close"] * 100
    else:
        derived["trend_20d"] = None

    return derived


def compute_forward_return(rows, idx, period):
    """计算未来 N 日收益率"""
    if idx + period >= len(rows):
        return None
    future_close = rows[idx + period]["close"]
    current_close = rows[idx]["close"]
    if current_close <= 0:
        return None
    return (future_close - current_close) / current_close * 100


def analyze_stock(filepath, min_data_points=120):
    """分析单只股票，返回 {factor: {period: [ic_values]}}"""
    rows = load_jsonl(filepath)
    if len(rows) < min_data_points + max(FORWARD_PERIODS):
        return None

    all_factors = FACTORS + DERIVED_FACTORS

    # 收集每个截面日期的因子值和未来收益
    # 按月采样避免自相关
    samples_by_month = defaultdict(list)  # month_key -> [(factor_vals, returns)]

    for idx in range(min_data_points, len(rows) - max(FORWARD_PERIODS)):
        row = rows[idx]
        date = row.get("date", "")
        if not date:
            continue

        month_key = date[:7]  # YYYY-MM

        # 提取因子值
        factor_vals = {}
        for f in FACTORS:
            val = row.get(f)
            if val is not None and val != 0:
                factor_vals[f] = float(val)

        # 计算衍生因子
        derived = compute_derived(rows, idx)
        for f in DERIVED_FACTORS:
            if derived.get(f) is not None:
                factor_vals[f] = derived[f]

        # 计算未来收益
        returns = {}
        for period in FORWARD_PERIODS:
            ret = compute_forward_return(rows, idx, period)
            if ret is not None:
                returns[period] = ret

        if factor_vals and returns:
            samples_by_month[month_key].append((factor_vals, returns))

    # 每月取中间一个样本（降低自相关）
    selected = []
    for month_key in sorted(samples_by_month.keys()):
        month_samples = samples_by_month[month_key]
        mid = len(month_samples) // 2
        selected.append(month_samples[mid])

    return selected


def rank_ic(factor_values, return_values):
    """计算 Rank IC（Spearman 相关系数）"""
    if len(factor_values) < 30:
        return None
    corr, pval = stats.spearmanr(factor_values, return_values)
    if np.isnan(corr):
        return None
    return corr


def main():
    parser = argparse.ArgumentParser(description="因子 IC 分析")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="最多分析多少只股票（0=全部）")
    parser.add_argument("--start", type=str, default="2015-01-01",
                        help="分析起始日期")
    parser.add_argument("--end", type=str, default="2025-12-31",
                        help="分析结束日期")
    args = parser.parse_args()

    ashare_dir = os.path.join(PROJECT_ROOT, cfg["data"]["ashare_dir"], "advanced")
    files = sorted(glob.glob(os.path.join(ashare_dir, "*.jsonl")))
    if args.max_stocks > 0:
        files = files[:args.max_stocks]

    print(f"因子 IC 分析")
    print(f"股票数: {len(files)}")
    print(f"因子数: {len(FACTORS) + len(DERIVED_FACTORS)}")
    print(f"预测周期: {FORWARD_PERIODS} 日")
    print()

    all_factors = FACTORS + DERIVED_FACTORS

    # 收集所有截面数据：按日期聚合
    # {date: {symbol: {factor: val, returns: {period: ret}}}}
    date_cross_sections = defaultdict(dict)

    for fi, fp in enumerate(files):
        if fi % 200 == 0:
            print(f"  加载: {fi}/{len(files)}")

        rows = load_jsonl(fp)
        if len(rows) < 140:
            continue

        sym = rows[0].get("symbol", os.path.basename(fp).replace(".jsonl", ""))

        for idx in range(120, len(rows) - max(FORWARD_PERIODS)):
            row = rows[idx]
            date = row.get("date", "")
            if not date or date < args.start or date > args.end:
                continue

            # 每月只取一个截面（月末附近）
            day = int(date[8:10]) if len(date) >= 10 else 0
            if day < 15 or day > 25:
                continue

            factor_vals = {}
            for f in FACTORS:
                val = row.get(f)
                if val is not None:
                    try:
                        factor_vals[f] = float(val)
                    except (ValueError, TypeError):
                        pass

            derived = compute_derived(rows, idx)
            for f in DERIVED_FACTORS:
                if derived.get(f) is not None:
                    factor_vals[f] = derived[f]

            returns = {}
            for period in FORWARD_PERIODS:
                ret = compute_forward_return(rows, idx, period)
                if ret is not None:
                    returns[period] = ret

            if factor_vals and returns:
                date_cross_sections[date][sym] = {
                    "factors": factor_vals,
                    "returns": returns,
                }

    print(f"\n截面日期数: {len(date_cross_sections)}")

    # 计算每个截面日期的 Rank IC
    # {factor: {period: [ic_per_date]}}
    ic_results = {f: {p: [] for p in FORWARD_PERIODS} for f in all_factors}

    sorted_dates = sorted(date_cross_sections.keys())
    for date in sorted_dates:
        section = date_cross_sections[date]
        if len(section) < 30:
            continue

        for factor in all_factors:
            fvals = []
            for period in FORWARD_PERIODS:
                rvals = []
                fv = []
                for sym, data in section.items():
                    if factor in data["factors"] and period in data["returns"]:
                        fv.append(data["factors"][factor])
                        rvals.append(data["returns"][period])

                if len(fv) >= 30:
                    ic = rank_ic(fv, rvals)
                    if ic is not None:
                        ic_results[factor][period].append(ic)

    # 输出结果
    print("\n" + "=" * 90)
    print(f"{'因子':<20} {'周期':>4} {'IC均值':>8} {'IC标准差':>8} {'IC_IR':>8} "
          f"{'正IC%':>6} {'截面数':>6} {'有效性':>6}")
    print("=" * 90)

    summary = {}

    for factor in all_factors:
        factor_summary = {}
        for period in FORWARD_PERIODS:
            ics = ic_results[factor][period]
            if len(ics) < 5:
                continue

            ic_mean = np.mean(ics)
            ic_std = np.std(ics, ddof=1)
            ic_ir = ic_mean / ic_std if ic_std > 1e-8 else 0
            positive_pct = sum(1 for x in ics if x > 0) / len(ics) * 100

            # 有效性判定
            if abs(ic_ir) >= 0.5 and abs(ic_mean) >= 0.03:
                grade = "★★★"
            elif abs(ic_ir) >= 0.3 and abs(ic_mean) >= 0.02:
                grade = "★★"
            elif abs(ic_mean) >= 0.01:
                grade = "★"
            else:
                grade = "—"

            print(f"{factor:<20} {period:>4}d {ic_mean:>+8.4f} {ic_std:>8.4f} "
                  f"{ic_ir:>+8.3f} {positive_pct:>5.1f}% {len(ics):>6} {grade:>6}")

            factor_summary[f"{period}d"] = {
                "ic_mean": round(ic_mean, 5),
                "ic_std": round(ic_std, 5),
                "ic_ir": round(ic_ir, 4),
                "positive_pct": round(positive_pct, 1),
                "n_sections": len(ics),
                "grade": grade,
            }

        if factor_summary:
            summary[factor] = factor_summary
        print()

    # 保存结果
    output_path = os.path.join(PROJECT_ROOT, "output", "factor_ic_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")

    # 打印推荐权重
    print("\n" + "=" * 60)
    print("因子权重建议（基于 10d IC_IR）")
    print("=" * 60)
    ranked = []
    for factor, data in summary.items():
        if "10d" in data:
            ranked.append((factor, data["10d"]["ic_ir"], data["10d"]["ic_mean"]))
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)

    for factor, ic_ir, ic_mean in ranked:
        direction = "正向" if ic_mean > 0 else "反向"
        weight = min(20, max(3, int(abs(ic_ir) * 15)))
        print(f"  {factor:<20} IC_IR={ic_ir:>+.3f}  方向={direction}  建议权重=±{weight}")


if __name__ == "__main__":
    main()
