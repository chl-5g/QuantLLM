#!/usr/bin/env python3
"""
预测性训练数据生成器 — 用实际未来收益做标签。
这是 V3 预测引擎改造的核心：从 rule-based scoring 转向 data-driven prediction。

Type A: 个股收益预测（~15000条）
  输入: [PREDICT_DATA] + 全部技术指标 + 市场环境
  标签: 实际 5/10/20 日收益方向（strong_buy/buy/hold/sell/strong_sell）

Type B: 板块轮动预测（~3000条）
  输入: [SECTOR_ROTATION] + 各板块 ETF 指标
  标签: 实际未来 N 日各板块收益排名

输出: training-data/predictive_signals.jsonl
      training-data/sector_rotation.jsonl
"""

import json
import os
import glob
import random
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASHARE_DIR = os.path.join(PROJECT_ROOT, "training-data", "ashare", "advanced")
ETF_DIR = os.path.join(PROJECT_ROOT, "training-data", "etf", "advanced")
OUTPUT_PREDICT = os.path.join(PROJECT_ROOT, "training-data", "predictive_signals.jsonl")
OUTPUT_SECTOR = os.path.join(PROJECT_ROOT, "training-data", "sector_rotation.jsonl")

SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。本AI仅提供信息参考，不构成投资建议，据此交易风险自负。"

# 采样日期范围
# 从2005年股权分置改革后开始：覆盖完整A股现代史
# 2005-2007 大牛市、2008 金融危机、2009-2014 长期震荡、
# 2015 杠杆牛+股灾、2016-2017 白马慢牛、2018 贸易战、
# 2019-2020 结构牛+新冠、2021-2024 阴跌、2024 反弹
# 留出 2025-07 之后做 walk-forward 测试
SAMPLE_START = "2005-06-01"
SAMPLE_END = "2025-06-30"

# 前瞻窗口（交易日）
FORWARD_WINDOWS = [5, 10, 20]

# 收益分类阈值
def classify_return(ret_pct):
    """将收益率分类为交易方向"""
    if ret_pct > 5:
        return "strong_buy"
    elif ret_pct > 2:
        return "buy"
    elif ret_pct > -2:
        return "hold"
    elif ret_pct > -5:
        return "sell"
    else:
        return "strong_sell"


# 核心板块 ETF 列表
SECTOR_ETFS = {
    "科技": "515000",
    "消费": "159928",
    "医药": "512010",
    "金融": "510050",
    "新能源": "516160",
    "军工": "512660",
    "半导体": "512480",
    "证券": "512880",
    "有色金属": "512400",
    "房地产": "512200",
    "基建": "516970",
    "传媒": "512980",
}


def read_symbol_data(filepath):
    """读取 JSONL 文件"""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return rows


def get_forward_returns(rows, idx):
    """
    计算 idx 位置的前瞻收益率。
    返回 {5: pct, 10: pct, 20: pct} 或 None（数据不足）。
    """
    current_close = rows[idx]["close"]
    if current_close <= 0:
        return None

    result = {}
    for window in FORWARD_WINDOWS:
        future_idx = idx + window
        if future_idx >= len(rows):
            return None
        future_close = rows[future_idx]["close"]
        if future_close <= 0:
            return None
        result[window] = round((future_close - current_close) / current_close * 100, 2)

    return result


def detect_regime_simple(rows, idx):
    """简化版市场环境判定"""
    if idx < 120:
        return "震荡"
    closes_120 = [rows[i]["close"] for i in range(idx - 119, idx + 1)]
    ma120 = sum(closes_120) / len(closes_120)
    current = rows[idx]["close"]
    ma_pct = (current - ma120) / ma120 * 100

    if idx < 140:
        slope = 0
    else:
        closes_prev = [rows[i]["close"] for i in range(idx - 139, idx - 19)]
        ma120_prev = sum(closes_prev) / len(closes_prev)
        slope = (ma120 - ma120_prev) / ma120_prev * 100

    if ma_pct > 3 and slope > 0.5:
        return "牛市"
    elif ma_pct < -3 and slope < -0.5:
        return "熊市"
    return "震荡"


def safe_get(row, key, default=0):
    """安全取值，处理 None 和缺失"""
    val = row.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_get_str(row, key, default="unknown"):
    val = row.get(key)
    return str(val) if val is not None else default


def format_predict_data(row, rows, idx, regime):
    """构造 [PREDICT_DATA] 格式的输入"""
    # 5日趋势
    if idx >= 5:
        close_5ago = rows[idx - 5]["close"]
        trend_5d = (row["close"] - close_5ago) / close_5ago * 100 if close_5ago > 0 else 0
    else:
        trend_5d = 0

    lines = [
        "[PREDICT_DATA]",
        f"symbol: {row.get('symbol', 'unknown')}",
        f"date: {row['date']}",
        f"close: {row['close']}",
        f"change_pct: {safe_get(row, 'change_pct', 0):+.2f}%",
        f"trend_5d: {trend_5d:+.2f}%",
        # 动量
        f"rsi_14: {safe_get(row, 'rsi_14', 50):.1f}",
        f"roc_12: {safe_get(row, 'roc_12'):+.2f}%",
        f"cci_20: {safe_get(row, 'cci_20'):.1f}",
        f"williams_r_14: {safe_get(row, 'williams_r_14', -50):.1f}",
        # MACD
        f"macd_histogram: {safe_get(row, 'macd_histogram'):.4f}",
        f"macd_cross: {'金叉' if safe_get(row, 'macd_line') > safe_get(row, 'signal_line') else '死叉'}",
        # 趋势
        f"adx_14: {safe_get(row, 'adx_14'):.1f}",
        f"ma_alignment: {safe_get_str(row, 'ma_alignment', 'mixed')}",
        f"above_ma20: {'true' if row['close'] > safe_get(row, 'close_ma_20', row['close']) else 'false'}",
        # 波动率
        f"atr_14: {safe_get(row, 'atr_14'):.2f}",
        f"bb_position: {safe_get(row, 'bb_position', 0.5):.2f}",
        f"hv_20: {safe_get(row, 'hv_20'):.1f}%",
        # 量能
        f"mfi_14: {safe_get(row, 'mfi_14', 50):.1f}",
        f"vol_change_rate: {safe_get(row, 'vol_change_rate', 1):.2f}",
        f"obv_trend: {safe_get_str(row, 'obv_trend', 'flat')}",
        # 环境
        f"regime: {regime}",
        "[END]",
        "请预测该股票未来5/10/20日的走势方向。",
    ]
    return "\n".join(lines)


def format_predict_answer(forward_returns, row, regime):
    """构造预测输出（用实际收益标注）"""
    predictions = {}
    key_factors = []
    risk_warnings = []

    rsi = safe_get(row, "rsi_14", 50)
    cci = safe_get(row, "cci_20", 0)
    adx = safe_get(row, "adx_14", 0)
    mfi = safe_get(row, "mfi_14", 50)
    bb_pos = safe_get(row, "bb_position", 0.5)
    ma_align = safe_get_str(row, "ma_alignment", "mixed")
    obv_trend = safe_get_str(row, "obv_trend", "flat")

    for window in FORWARD_WINDOWS:
        ret = forward_returns[window]
        direction = classify_return(ret)
        # 置信度：收益越极端，置信度越高
        confidence = min(0.9, 0.4 + abs(ret) / 20)
        predictions[f"prediction_{window}d"] = {
            "direction": direction,
            "confidence": round(confidence, 2),
        }

    # 基于当前指标生成分析因子
    if rsi < 30:
        key_factors.append(f"RSI={rsi:.0f}超卖，存在反弹动能")
    elif rsi > 70:
        key_factors.append(f"RSI={rsi:.0f}超买，回调压力增大")

    if cci < -100:
        key_factors.append(f"CCI={cci:.0f}深度负值，超卖信号")
    elif cci > 100:
        key_factors.append(f"CCI={cci:.0f}高位，超买信号")

    if adx > 25:
        key_factors.append(f"ADX={adx:.0f}>25，趋势明确")
    elif adx < 15:
        key_factors.append(f"ADX={adx:.0f}<15，无明确趋势")

    if ma_align == "bullish":
        key_factors.append("均线多头排列")
    elif ma_align == "bearish":
        key_factors.append("均线空头排列")

    if obv_trend == "rising":
        key_factors.append("OBV上升，资金持续流入")
    elif obv_trend == "falling":
        key_factors.append("OBV下降，资金持续流出")

    if mfi < 20:
        key_factors.append(f"MFI={mfi:.0f}，资金超卖")
    elif mfi > 80:
        risk_warnings.append(f"MFI={mfi:.0f}，资金面过热")

    if bb_pos < 0.1:
        key_factors.append(f"布林带底部(位置{bb_pos:.2f})，存在均值回归动力")
    elif bb_pos > 0.9:
        risk_warnings.append(f"布林带顶部(位置{bb_pos:.2f})，注意回落风险")

    if regime == "熊市":
        risk_warnings.append("熊市环境，控制仓位")
    elif regime == "牛市":
        key_factors.append("牛市环境，趋势向好")

    if not key_factors:
        key_factors.append("各项指标中性，无显著信号")
    if not risk_warnings:
        risk_warnings.append("当前无显著风险信号")

    output = {**predictions, "key_factors": key_factors, "risk_warning": "；".join(risk_warnings)}

    answer = f"```json\n{json.dumps(output, ensure_ascii=False, indent=2)}\n```\n\n"

    # 添加文字解读
    ret_5d = forward_returns[5]
    ret_20d = forward_returns[20]
    if ret_20d > 5:
        answer += f"**综合研判**：短期({ret_5d:+.1f}%)和中期({ret_20d:+.1f}%)均偏乐观，多项指标共振偏多。"
    elif ret_20d < -5:
        answer += f"**综合研判**：中期({ret_20d:+.1f}%)偏悲观，建议控制仓位或空仓回避。"
    elif ret_5d > 2:
        answer += f"**综合研判**：短期({ret_5d:+.1f}%)有反弹机会，但中期({ret_20d:+.1f}%)方向不明。"
    else:
        answer += f"**综合研判**：短期({ret_5d:+.1f}%)和中期({ret_20d:+.1f}%)信号偏中性，建议观望。"

    return answer


def generate_stock_predictions():
    """Type A: 个股收益预测样本"""
    files = sorted(glob.glob(os.path.join(ASHARE_DIR, "*.jsonl")))
    if not files:
        print("  A股 advanced 数据目录为空，跳过")
        return []

    print(f"  [个股预测] 扫描 {len(files)} 个标的")
    records = []
    stats = defaultdict(int)

    for fi, filepath in enumerate(files):
        symbol = os.path.basename(filepath).replace(".jsonl", "")
        rows = read_symbol_data(filepath)

        if len(rows) < 150:  # 至少需要120天warmup + 20天前瞻
            continue

        # 筛选在采样范围内的候选日期
        candidates = []
        for idx in range(120, len(rows) - 20):
            date = rows[idx]["date"]
            if SAMPLE_START <= date <= SAMPLE_END:
                candidates.append(idx)

        if not candidates:
            continue

        # 每只股票采样 8 个日期（覆盖20年多种行情周期）
        n_samples = min(8, len(candidates))
        sampled = random.sample(candidates, n_samples)

        for idx in sampled:
            row = rows[idx]
            forward_returns = get_forward_returns(rows, idx)
            if forward_returns is None:
                continue

            regime = detect_regime_simple(rows, idx)
            question = format_predict_data(row, rows, idx, regime)
            answer = format_predict_answer(forward_returns, row, regime)

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            records.append(record)

            # 统计标签分布
            for w in FORWARD_WINDOWS:
                direction = classify_return(forward_returns[w])
                stats[f"{w}d_{direction}"] += 1

        if (fi + 1) % 500 == 0:
            print(f"    [{fi+1}/{len(files)}] 已生成 {len(records)} 条")

    print(f"  [个股预测] 共生成 {len(records)} 条")
    print(f"  标签分布:")
    for k in sorted(stats.keys()):
        print(f"    {k}: {stats[k]}")

    return records


def generate_sector_rotation():
    """Type B: 板块轮动预测样本"""
    # 读取所有板块 ETF 数据
    sector_data = {}
    for sector_name, etf_code in SECTOR_ETFS.items():
        filepath = os.path.join(ETF_DIR, f"{etf_code}.jsonl")
        if not os.path.exists(filepath):
            # 尝试其他文件名格式
            candidates = glob.glob(os.path.join(ETF_DIR, f"*{etf_code}*.jsonl"))
            if candidates:
                filepath = candidates[0]
            else:
                print(f"  [板块轮动] 未找到 {sector_name}({etf_code}) 数据，跳过")
                continue
        rows = read_symbol_data(filepath)
        if len(rows) >= 150:
            sector_data[sector_name] = rows

    if len(sector_data) < 4:
        print(f"  [板块轮动] 有效板块不足4个（{len(sector_data)}），跳过")
        return []

    print(f"  [板块轮动] 加载 {len(sector_data)} 个板块ETF")

    # 建立日期索引（找所有板块共同的交易日）
    date_sets = []
    for name, rows in sector_data.items():
        dates = set(r["date"] for r in rows)
        date_sets.append(dates)
    common_dates = sorted(set.intersection(*date_sets))
    print(f"  [板块轮动] 共同交易日 {len(common_dates)} 天")

    # 为每个板块建立日期到行索引的映射
    sector_date_idx = {}
    for name, rows in sector_data.items():
        date_map = {}
        for i, r in enumerate(rows):
            date_map[r["date"]] = i
        sector_date_idx[name] = date_map

    records = []
    # 采样日期
    valid_dates = [d for d in common_dates if SAMPLE_START <= d <= SAMPLE_END]
    if len(valid_dates) < 30:
        print(f"  [板块轮动] 有效日期不足30天，跳过")
        return []

    # 每隔5个交易日采样一次
    sampled_dates = valid_dates[::5]
    print(f"  [板块轮动] 采样 {len(sampled_dates)} 个日期")

    for date in sampled_dates:
        # 检查所有板块在此日期后是否有足够的前瞻数据
        sector_snapshots = {}
        sector_forward = {}
        all_valid = True

        for name, rows in sector_data.items():
            idx = sector_date_idx[name].get(date)
            if idx is None or idx < 20 or idx + 20 >= len(rows):
                all_valid = False
                break

            row = rows[idx]
            # 5日前瞻收益
            ret_5d = (rows[idx + 5]["close"] - row["close"]) / row["close"] * 100 if idx + 5 < len(rows) else None
            ret_10d = (rows[idx + 10]["close"] - row["close"]) / row["close"] * 100 if idx + 10 < len(rows) else None

            if ret_5d is None or ret_10d is None:
                all_valid = False
                break

            # 5日历史收益
            hist_5d = (row["close"] - rows[idx - 5]["close"]) / rows[idx - 5]["close"] * 100

            sector_snapshots[name] = {
                "rsi": safe_get(row, "rsi_14", 50),
                "adx": safe_get(row, "adx_14", 0),
                "mfi": safe_get(row, "mfi_14", 50),
                "roc_12": safe_get(row, "roc_12", 0),
                "bb_position": safe_get(row, "bb_position", 0.5),
                "vol_change_rate": safe_get(row, "vol_change_rate", 1),
                "ma_alignment": safe_get_str(row, "ma_alignment", "mixed"),
                "hist_5d_return": round(hist_5d, 2),
            }
            sector_forward[name] = {
                "ret_5d": round(ret_5d, 2),
                "ret_10d": round(ret_10d, 2),
            }

        if not all_valid:
            continue

        # 构造输入
        lines = [f"[SECTOR_ROTATION]", f"date: {date}"]

        # 按历史5日收益排序展示
        sorted_sectors = sorted(sector_snapshots.keys(),
                                key=lambda s: sector_snapshots[s]["hist_5d_return"],
                                reverse=True)

        for name in sorted_sectors:
            snap = sector_snapshots[name]
            etf_code = SECTOR_ETFS[name]
            lines.append(
                f"- {name}ETF({etf_code}): rsi={snap['rsi']:.0f}, "
                f"adx={snap['adx']:.0f}, mfi={snap['mfi']:.0f}, "
                f"roc_12={snap['roc_12']:+.1f}%, "
                f"bb_pos={snap['bb_position']:.2f}, "
                f"vol_ratio={snap['vol_change_rate']:.2f}, "
                f"ma_align={snap['ma_alignment']}, "
                f"5d_return={snap['hist_5d_return']:+.2f}%"
            )

        # 用第一个板块的数据判断市场整体环境
        first_sector = list(sector_data.keys())[0]
        first_rows = sector_data[first_sector]
        first_idx = sector_date_idx[first_sector][date]
        regime = detect_regime_simple(first_rows, first_idx)
        lines.append(f"market_regime: {regime}")
        lines.append("[END]")
        lines.append("请分析当前板块轮动方向，推荐配置。")

        question = "\n".join(lines)

        # 构造输出（用实际前瞻收益标注）
        # 按 5日收益排名
        ranking_5d = sorted(sector_forward.keys(),
                            key=lambda s: sector_forward[s]["ret_5d"],
                            reverse=True)
        ranking_10d = sorted(sector_forward.keys(),
                             key=lambda s: sector_forward[s]["ret_10d"],
                             reverse=True)

        # 推荐前3强+后3弱
        top3 = ranking_5d[:3]
        bottom3 = ranking_5d[-3:]

        output = {
            "ranking_5d": [
                {"sector": s, "expected_return": f"{sector_forward[s]['ret_5d']:+.2f}%"}
                for s in ranking_5d
            ],
            "ranking_10d": [
                {"sector": s, "expected_return": f"{sector_forward[s]['ret_10d']:+.2f}%"}
                for s in ranking_10d
            ],
            "recommendation": {
                "overweight": top3,
                "underweight": bottom3,
            },
            "regime": regime,
        }

        # 生成分析文字
        analysis = f"```json\n{json.dumps(output, ensure_ascii=False, indent=2)}\n```\n\n"
        analysis += f"**板块轮动分析**（{date}）：\n\n"
        analysis += f"**强势板块**：{'、'.join(top3)}，建议超配。\n"
        for s in top3:
            snap = sector_snapshots[s]
            fwd = sector_forward[s]
            analysis += f"- {s}：5日预期{fwd['ret_5d']:+.1f}%，RSI={snap['rsi']:.0f}，ADX={snap['adx']:.0f}\n"
        analysis += f"\n**弱势板块**：{'、'.join(bottom3)}，建议低配或回避。\n"
        for s in bottom3:
            fwd = sector_forward[s]
            analysis += f"- {s}：5日预期{fwd['ret_5d']:+.1f}%\n"

        if regime == "熊市":
            analysis += f"\n**风险提示**：当前市场处于熊市环境，所有板块配置应降低仓位，优先防御性板块。"
        elif regime == "牛市":
            analysis += f"\n**环境利好**：牛市环境下可适度提高进攻性板块权重。"

        record = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": analysis},
            ]
        }
        records.append(record)

    print(f"  [板块轮动] 共生成 {len(records)} 条")
    return records


def main():
    print("=" * 60)
    print("预测性训练数据生成（实际收益标签）")
    print("=" * 60)

    # Type A: 个股预测
    print("\n--- Type A: 个股收益预测 ---")
    stock_records = generate_stock_predictions()

    # Type B: 板块轮动
    print("\n--- Type B: 板块轮动预测 ---")
    sector_records = generate_sector_rotation()

    # 写入
    if stock_records:
        random.shuffle(stock_records)
        with open(OUTPUT_PREDICT, "w", encoding="utf-8") as f:
            for r in stock_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n个股预测数据: {len(stock_records)} 条 → {OUTPUT_PREDICT}")

    if sector_records:
        with open(OUTPUT_SECTOR, "w", encoding="utf-8") as f:
            for r in sector_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"板块轮动数据: {len(sector_records)} 条 → {OUTPUT_SECTOR}")

    total = len(stock_records) + len(sector_records)
    print(f"\n{'=' * 60}")
    print(f"预测性训练数据生成完成！总计 {total} 条")

    if stock_records:
        lens = sorted(sum(len(m["content"]) for m in r["messages"]) for r in stock_records)
        print(f"个股预测 - 对话长度 中位数: {lens[len(lens)//2]}, P90: {lens[len(lens)*9//10]}, 最长: {lens[-1]}")

    if sector_records:
        lens = sorted(sum(len(m["content"]) for m in r["messages"]) for r in sector_records)
        print(f"板块轮动 - 对话长度 中位数: {lens[len(lens)//2]}, P90: {lens[len(lens)*9//10]}, 最长: {lens[-1]}")


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    # 断点续跑：输出文件已存在且非空时跳过（--force 强制重生成）
    if not force:
        skip_stock = os.path.exists(OUTPUT_PREDICT) and os.path.getsize(OUTPUT_PREDICT) > 0
        skip_sector = os.path.exists(OUTPUT_SECTOR) and os.path.getsize(OUTPUT_SECTOR) > 0
        if skip_stock and skip_sector:
            stock_lines = sum(1 for _ in open(OUTPUT_PREDICT))
            sector_lines = sum(1 for _ in open(OUTPUT_SECTOR))
            print(f"输出文件已存在: predictive_signals({stock_lines}条), sector_rotation({sector_lines}条)")
            print("跳过生成。如需重新生成，使用 --force 参数")
            sys.exit(0)
    main()
