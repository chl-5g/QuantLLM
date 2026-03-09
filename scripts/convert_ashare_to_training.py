#!/usr/bin/env python3
"""
将爬取的A股行情数据转化为LLM微调训练数据（ChatML JSONL格式）
输入：/tmp/training-data/ashare/advanced/*.jsonl（带技术指标的行情）
输出：/tmp/training-data/ashare_train.jsonl

生成策略：
1. 技术面分析问答 — 给出指标，请求分析
2. 趋势判断问答 — 给出连续N日数据，判断趋势
3. 买卖信号识别 — 根据指标组合识别信号
4. 风险提示问答 — 识别异常形态并提示风险
"""

import json
import os
import glob
import random
import numpy as np
from collections import defaultdict

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(PROJECT_ROOT, "ashare", "advanced")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "ashare_train.jsonl")
SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长技术分析、因子分析、趋势判断和风险管理。"

# 每只股票最多生成的问答对数
MAX_PER_STOCK = 3
# 总数上限
TOTAL_LIMIT = 5000

# ============================================================
# 辅助函数
# ============================================================

def describe_rsi(rsi):
    if rsi >= 80:
        return "严重超买", "极高"
    elif rsi >= 70:
        return "超买", "偏高"
    elif rsi >= 55:
        return "偏强", "中性偏强"
    elif rsi >= 45:
        return "中性", "中性"
    elif rsi >= 30:
        return "偏弱", "中性偏弱"
    elif rsi >= 20:
        return "超卖", "偏低"
    else:
        return "严重超卖", "极低"

def describe_macd(hist, prev_hist=None):
    if hist > 0:
        if prev_hist is not None and hist > prev_hist:
            return "MACD柱线为正且放大，多头动能增强"
        elif prev_hist is not None and hist < prev_hist:
            return "MACD柱线为正但缩小，多头动能减弱"
        return "MACD柱线为正，处于多头区间"
    else:
        if prev_hist is not None and hist < prev_hist:
            return "MACD柱线为负且放大，空头动能增强"
        elif prev_hist is not None and hist > prev_hist:
            return "MACD柱线为负但缩小，空头动能减弱"
        return "MACD柱线为负，处于空头区间"

def describe_ma_position(close, ma20):
    diff_pct = (close - ma20) / ma20 * 100
    if diff_pct > 10:
        return f"股价远高于20日均线（偏离{diff_pct:.1f}%），短期乖离较大"
    elif diff_pct > 3:
        return f"股价在20日均线上方运行（偏离{diff_pct:.1f}%），中期趋势偏多"
    elif diff_pct > -3:
        return f"股价贴近20日均线（偏离{diff_pct:.1f}%），处于方向选择阶段"
    elif diff_pct > -10:
        return f"股价在20日均线下方运行（偏离{diff_pct:.1f}%），中期趋势偏空"
    else:
        return f"股价远低于20日均线（偏离{diff_pct:.1f}%），短期超跌"

def describe_volume(volume, volume_ma5):
    if volume_ma5 == 0:
        return "成交量数据异常"
    ratio = volume / volume_ma5
    if ratio > 2.5:
        return f"成交量为5日均量的{ratio:.1f}倍，属于巨量放量"
    elif ratio > 1.5:
        return f"成交量为5日均量的{ratio:.1f}倍，明显放量"
    elif ratio > 0.8:
        return "成交量接近5日均量水平，量能正常"
    elif ratio > 0.5:
        return f"成交量仅为5日均量的{ratio:.1f}倍，明显缩量"
    else:
        return f"成交量仅为5日均量的{ratio:.1f}倍，极度缩量"

def calc_change_pct(close, open_price):
    if open_price == 0:
        return 0
    return round((close - open_price) / open_price * 100, 2)

# ============================================================
# 问答生成模板
# ============================================================

def gen_technical_analysis(symbol, row, prev_row=None):
    """模板1：单日技术面综合分析"""
    rsi_desc, rsi_level = describe_rsi(row["rsi_14"])
    prev_hist = prev_row["macd_histogram"] if prev_row else None
    macd_desc = describe_macd(row["macd_histogram"], prev_hist)
    ma_desc = describe_ma_position(row["close"], row["close_ma_20"])
    vol_desc = describe_volume(row["volume"], row["volume_ma_5"]) if row["volume_ma_5"] > 0 else ""
    change = calc_change_pct(row["close"], row["open"])

    question = (
        f"{symbol} 在 {row['date']} 的技术指标如下：\n"
        f"- 收盘价 {row['close']}，当日涨跌 {change:+.2f}%\n"
        f"- RSI(14) = {row['rsi_14']}\n"
        f"- MACD柱线 = {row['macd_histogram']}，信号线 = {row['signal_line']}\n"
        f"- 20日均线 = {row['close_ma_20']}\n"
        f"- 成交量 {row['volume']:,}，5日均量 {row['volume_ma_5']:,}\n"
        f"请对该股当前技术面进行综合分析。"
    )

    answer_parts = [
        f"从技术面来看，{symbol} 在 {row['date']} 呈现以下特征：",
        f"",
        f"**RSI指标**：RSI(14) 为 {row['rsi_14']}，处于{rsi_desc}区域。" + (
            "短期需警惕回调风险，不宜追高。" if row["rsi_14"] >= 70
            else "短期存在超跌反弹的可能。" if row["rsi_14"] <= 30
            else "动能处于正常范围。"
        ),
        f"",
        f"**MACD指标**：{macd_desc}。MACD线为 {row['macd_line']}，信号线为 {row['signal_line']}。" + (
            "DIF在DEA上方，短期偏多。" if row["macd_line"] > row["signal_line"]
            else "DIF在DEA下方，短期偏空。"
        ),
        f"",
        f"**均线位置**：{ma_desc}。" + (
            "均线系统支撑有效。" if row["close"] > row["close_ma_20"]
            else "均线形成压力位。"
        ),
    ]
    if vol_desc:
        answer_parts.extend([
            f"",
            f"**量能分析**：{vol_desc}。" + (
                "放量配合上涨为健康走势。" if change > 0 and row["volume"] > row["volume_ma_5"]
                else "放量下跌需警惕主力出货。" if change < 0 and row["volume"] > row["volume_ma_5"] * 1.5
                else "缩量运行说明市场观望情绪浓厚。" if row["volume"] < row["volume_ma_5"] * 0.7
                else ""
            ),
        ])
    answer_parts.extend([
        f"",
        f"**综合判断**：" + (
            "多项指标共振偏多，但RSI偏高需注意回调风险，建议逢低布局而非追涨。"
            if row["rsi_14"] >= 60 and row["macd_histogram"] > 0 and row["close"] > row["close_ma_20"]
            else "指标出现底部特征，RSI超卖叠加MACD底背离可能性，可关注企稳信号。"
            if row["rsi_14"] <= 35 and row["macd_histogram"] < 0
            else "指标信号不一致，建议观望为主，等待方向明确后再操作。"
            if (row["macd_histogram"] > 0) != (row["close"] > row["close_ma_20"])
            else "技术面整体偏多，量价配合良好，可持股待涨。"
            if row["macd_histogram"] > 0 and row["close"] > row["close_ma_20"]
            else "技术面整体偏空，建议轻仓或空仓观望。"
        ),
        f"",
        f"*以上分析仅基于技术指标，实际操作需结合基本面和市场环境综合判断。*"
    ])

    return question, "\n".join(answer_parts)


def gen_trend_analysis(symbol, rows):
    """模板2：连续N日趋势判断（需要5-10日数据）"""
    if len(rows) < 5:
        return None, None

    window = rows[-5:]
    closes = [r["close"] for r in window]
    volumes = [r["volume"] for r in window]
    rsis = [r["rsi_14"] for r in window]

    # 涨跌统计
    up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
    total_change = (closes[-1] - closes[0]) / closes[0] * 100

    data_lines = []
    for r in window:
        chg = calc_change_pct(r["close"], r["open"])
        data_lines.append(f"  {r['date']}: 开{r['open']} 高{r['high']} 低{r['low']} 收{r['close']} 量{r['volume']:,} RSI={r['rsi_14']}")

    question = (
        f"以下是 {symbol} 最近5个交易日的行情数据：\n"
        + "\n".join(data_lines)
        + f"\n请判断该股短期趋势方向，并给出操作建议。"
    )

    # 判断趋势
    if total_change > 5 and up_days >= 4:
        trend = "强势上涨"
        detail = f"5日累计涨幅 {total_change:.1f}%，{up_days}阳{5-up_days}阴，属于强势上攻走势。"
        advice = "短期动能充沛，但连续上涨后需关注获利回吐压力。若伴随放量可继续持有，缩量上涨则需提高警惕。"
    elif total_change > 2:
        trend = "温和上涨"
        detail = f"5日累计涨幅 {total_change:.1f}%，走势偏强。"
        advice = "趋势向好但力度一般，适合轻仓跟随。关注能否突破近期阻力位放量上攻。"
    elif total_change < -5 and up_days <= 1:
        trend = "持续下跌"
        detail = f"5日累计跌幅 {total_change:.1f}%，仅{up_days}天收阳，空方占据绝对优势。"
        advice = "下跌趋势明确，不宜抄底。等待RSI进入超卖区域且出现放量长下影线等止跌信号后再考虑介入。"
    elif total_change < -2:
        trend = "偏弱震荡"
        detail = f"5日累计跌幅 {total_change:.1f}%，走势偏弱。"
        advice = "弱势格局，建议减仓或观望。若RSI接近30关注是否有超跌反弹机会。"
    else:
        trend = "横盘整理"
        detail = f"5日累计变动 {total_change:+.1f}%，振幅有限，处于方向选择阶段。"
        advice = "横盘整理阶段不宜重仓，等待放量突破方向明确后再跟随。关注20日均线的支撑/压力作用。"

    vol_trend = "放量" if volumes[-1] > volumes[0] * 1.3 else "缩量" if volumes[-1] < volumes[0] * 0.7 else "量能平稳"

    answer = (
        f"**趋势判断：{trend}**\n\n"
        f"{detail}\n\n"
        f"**量能特征**：近5日{vol_trend}，" + (
            "量价配合良好。" if (total_change > 0 and volumes[-1] > volumes[0])
            else "量价背离需警惕。" if (total_change > 3 and volumes[-1] < volumes[0] * 0.7)
            else "量能变化不显著。"
        ) + f"\n\n"
        f"**RSI走势**：从 {rsis[0]} 变动到 {rsis[-1]}，" + (
            "动能持续增强。" if rsis[-1] > rsis[0] + 5
            else "动能有所衰减。" if rsis[-1] < rsis[0] - 5
            else "动能变化不大。"
        ) + f"\n\n"
        f"**操作建议**：{advice}\n\n"
        f"*注：以上分析基于短期技术面，中长期操作需结合基本面研究。*"
    )

    return question, answer


def gen_signal_detection(symbol, row, prev_rows):
    """模板3：买卖信号识别"""
    if len(prev_rows) < 3:
        return None, None

    signals = []
    signal_type = None

    # 金叉/死叉检测
    if len(prev_rows) >= 2:
        prev_macd = prev_rows[-1]["macd_line"] - prev_rows[-1]["signal_line"]
        curr_macd = row["macd_line"] - row["signal_line"]
        if prev_macd < 0 and curr_macd > 0:
            signals.append("MACD金叉（DIF上穿DEA）")
            signal_type = "buy"
        elif prev_macd > 0 and curr_macd < 0:
            signals.append("MACD死叉（DIF下穿DEA）")
            signal_type = "sell"

    # RSI信号
    if row["rsi_14"] <= 25:
        signals.append(f"RSI进入深度超卖区域（{row['rsi_14']}）")
        signal_type = signal_type or "buy"
    elif row["rsi_14"] >= 78:
        signals.append(f"RSI进入深度超买区域（{row['rsi_14']}）")
        signal_type = signal_type or "sell"

    # 均线突破
    prev_above = prev_rows[-1]["close"] > prev_rows[-1]["close_ma_20"]
    curr_above = row["close"] > row["close_ma_20"]
    if not prev_above and curr_above:
        signals.append("股价向上突破20日均线")
        signal_type = signal_type or "buy"
    elif prev_above and not curr_above:
        signals.append("股价跌破20日均线")
        signal_type = signal_type or "sell"

    # 放量异动
    if row["volume_ma_5"] > 0 and row["volume"] > row["volume_ma_5"] * 2:
        change = calc_change_pct(row["close"], row["open"])
        if change > 3:
            signals.append(f"放量大涨（量比{row['volume']/row['volume_ma_5']:.1f}，涨幅{change:.1f}%）")
        elif change < -3:
            signals.append(f"放量大跌（量比{row['volume']/row['volume_ma_5']:.1f}，跌幅{change:.1f}%）")

    if not signals:
        return None, None

    question = (
        f"{symbol} 在 {row['date']} 出现以下技术信号：\n"
        + "\n".join(f"- {s}" for s in signals)
        + f"\n当日收盘价 {row['close']}，RSI(14)={row['rsi_14']}，MACD柱线={row['macd_histogram']}。"
        + f"\n请分析这些信号的含义及操作建议。"
    )

    if signal_type == "buy":
        answer = (
            f"**信号解读：偏多信号**\n\n"
            f"{symbol} 在 {row['date']} 出现了以下值得关注的买入信号：\n\n"
            + "\n".join(f"- {s}" for s in signals)
            + f"\n\n**信号强度分析**：\n"
            f"- 当前RSI为{row['rsi_14']}，" + (
                "处于超卖区域，反弹概率较大。" if row["rsi_14"] < 30
                else "处于中性偏弱区域，有上行空间。" if row["rsi_14"] < 50
                else "已处于偏强区域，追涨空间有限。"
            ) + "\n"
            f"- MACD柱线为{row['macd_histogram']}，" + (
                "多头动能正在积蓄。" if row["macd_histogram"] > 0
                else "空头动能尚未完全释放，信号需要确认。"
            ) + "\n\n"
            f"**操作建议**：\n"
            f"- 多个信号共振时可信度更高，单一信号建议轻仓试探\n"
            f"- 可在信号出现后等待1-2日确认，避免假突破\n"
            f"- 设置合理止损位（建议参考近期低点或20日均线下方2-3%）\n"
            f"- 分批建仓，控制单次买入仓位不超过总资金的20%\n\n"
            f"*风险提示：技术信号存在失败概率，需严格执行止损纪律。*"
        )
    else:
        answer = (
            f"**信号解读：偏空信号**\n\n"
            f"{symbol} 在 {row['date']} 出现了以下风险信号：\n\n"
            + "\n".join(f"- {s}" for s in signals)
            + f"\n\n**风险评估**：\n"
            f"- 当前RSI为{row['rsi_14']}，" + (
                "处于超买区域，回调风险较大。" if row["rsi_14"] > 70
                else "尚在中性区间，但趋势转弱。" if row["rsi_14"] > 45
                else "已进入偏弱区域，可能进一步走弱。"
            ) + "\n"
            f"- MACD柱线为{row['macd_histogram']}，" + (
                "虽然仍在多头区间，但动能明显衰减。" if row["macd_histogram"] > 0
                else "空头动能正在加速，下跌可能延续。"
            ) + "\n\n"
            f"**操作建议**：\n"
            f"- 持仓者建议减仓或设置移动止损保护利润\n"
            f"- 空仓者不宜在此时抄底，等待企稳信号\n"
            f"- 关注下方支撑位（20日均线 {row['close_ma_20']}）的支撑效果\n"
            f"- 如果跌破关键支撑位伴随放量，需果断离场\n\n"
            f"*风险提示：市场情绪转变时技术支撑可能失效，严格风控。*"
        )

    return question, answer


def gen_risk_alert(symbol, row, prev_rows):
    """模板4：风险提示"""
    if len(prev_rows) < 3:
        return None, None

    risks = []

    # 高位放量滞涨
    change = calc_change_pct(row["close"], row["open"])
    if (row["rsi_14"] > 70 and row["volume_ma_5"] > 0 and
        row["volume"] > row["volume_ma_5"] * 1.8 and abs(change) < 1):
        risks.append("高位放量滞涨")

    # 连续缩量新高（量价背离）
    closes = [r["close"] for r in prev_rows[-3:]] + [row["close"]]
    volumes = [r["volume"] for r in prev_rows[-3:]] + [row["volume"]]
    if len(closes) >= 4:
        if all(closes[i] > closes[i-1] for i in range(1, len(closes))):
            if all(volumes[i] < volumes[i-1] for i in range(1, len(volumes))):
                risks.append("连续上涨但成交量递减（量价背离）")

    # RSI顶背离（价格新高RSI不新高）
    if len(prev_rows) >= 5:
        recent_rsis = [r["rsi_14"] for r in prev_rows[-5:]] + [row["rsi_14"]]
        recent_closes = [r["close"] for r in prev_rows[-5:]] + [row["close"]]
        if recent_closes[-1] == max(recent_closes) and recent_rsis[-1] < max(recent_rsis[:-1]):
            risks.append("RSI顶背离（价格创新高但RSI未创新高）")

    # 跌停或接近跌停
    if change < -9:
        risks.append(f"当日跌幅达{change:.1f}%，接近跌停")

    if not risks:
        return None, None

    question = (
        f"{symbol} 在 {row['date']} 出现以下异常信号，请进行风险分析：\n"
        f"- 收盘价: {row['close']}，涨跌: {change:+.1f}%\n"
        f"- RSI(14): {row['rsi_14']}\n"
        f"- 成交量: {row['volume']:,}（5日均量: {row['volume_ma_5']:,}）\n"
        f"- MACD柱线: {row['macd_histogram']}\n"
        f"异常特征：\n"
        + "\n".join(f"- {r}" for r in risks)
    )

    answer = (
        f"**风险警示：{symbol} 需要高度关注**\n\n"
        f"在 {row['date']} 检测到以下风险信号：\n\n"
        + "\n".join(f"**{i+1}. {r}**" for i, r in enumerate(risks))
        + "\n\n**详细分析**：\n\n"
    )

    for risk in risks:
        if "量价背离" in risk:
            answer += (
                "量价背离是重要的顶部预警信号。价格持续上涨但成交量递减，说明上涨动能不足，"
                "买方力量逐渐枯竭。历史统计表明，量价背离后出现回调的概率在60-70%以上。\n\n"
            )
        elif "高位放量滞涨" in risk:
            answer += (
                "在RSI高位出现放量但价格涨幅有限，可能暗示主力在高位出货。"
                "大量资金流入但价格无法有效拉升，说明卖方压力沉重。\n\n"
            )
        elif "顶背离" in risk:
            answer += (
                "RSI顶背离是经典的趋势反转预警。价格创新高但RSI动能指标未能同步，"
                "说明上涨动力在减弱。配合其他指标确认后，可能预示一波较大幅度的回调。\n\n"
            )
        elif "跌停" in risk or "跌幅" in risk:
            answer += (
                "单日大幅下跌通常伴随恐慌情绪，可能由突发利空驱动。"
                "需关注后续是否有持续杀跌，以及成交量变化。\n\n"
            )

    answer += (
        "**应对建议**：\n"
        "1. 已持仓：立即检查仓位，设置或收紧止损线\n"
        "2. 拟买入：暂缓操作，等待风险信号消除\n"
        "3. 密切关注后续2-3个交易日的量价变化，确认是否为趋势反转\n"
        "4. 如有条件，关注该股的基本面是否出现变化（财报、政策、行业动态）\n\n"
        "*风险提示：单一技术信号不构成投资建议，需结合多维度分析综合判断。*"
    )

    return question, answer


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("A股行情数据 → 训练数据转换")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jsonl")))
    print(f"找到 {len(files)} 个股票数据文件")

    if not files:
        print("没有数据文件，请先运行 crawl_ashare.py")
        return

    all_records = []
    stats = defaultdict(int)

    for fi, filepath in enumerate(files):
        symbol = os.path.basename(filepath).replace(".jsonl", "")

        # 读取该股票所有数据
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except:
                    continue

        if len(rows) < 30:
            stats["skipped_short"] += 1
            continue

        stock_records = []

        # 选取有代表性的日期点生成问答
        # 策略：从数据后半段选取（更近期的数据更有参考价值）
        half = len(rows) // 2
        candidate_indices = list(range(half, len(rows)))
        random.shuffle(candidate_indices)

        for idx in candidate_indices:
            if len(stock_records) >= MAX_PER_STOCK:
                break

            row = rows[idx]
            prev_rows = rows[max(0, idx-5):idx]
            prev_row = rows[idx-1] if idx > 0 else None

            # 随机选择一种模板
            template_choice = random.random()

            if template_choice < 0.35:
                q, a = gen_technical_analysis(symbol, row, prev_row)
                ttype = "technical_analysis"
            elif template_choice < 0.55:
                q, a = gen_trend_analysis(symbol, rows[max(0, idx-4):idx+1])
                ttype = "trend_analysis"
            elif template_choice < 0.80:
                q, a = gen_signal_detection(symbol, row, prev_rows)
                ttype = "signal_detection"
            else:
                q, a = gen_risk_alert(symbol, row, prev_rows)
                ttype = "risk_alert"

            if q and a:
                record = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]
                }
                stock_records.append(record)
                stats[ttype] += 1

        all_records.extend(stock_records)

        if (fi + 1) % 200 == 0:
            print(f"  处理进度: {fi+1}/{len(files)}，已生成 {len(all_records)} 条")

    # 如果超过上限，随机采样
    if len(all_records) > TOTAL_LIMIT:
        random.shuffle(all_records)
        all_records = all_records[:TOTAL_LIMIT]
        print(f"数据量超过上限，随机采样 {TOTAL_LIMIT} 条")

    # 打乱顺序
    random.shuffle(all_records)

    # 写出
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"转换完成！")
    print(f"  总条数: {len(all_records)}")
    print(f"  类型分布:")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")
    print(f"  输出文件: {OUTPUT_FILE}")

    # 统计对话长度
    lens = []
    for r in all_records:
        total_len = sum(len(m["content"]) for m in r["messages"])
        lens.append(total_len)
    if lens:
        lens.sort()
        print(f"  对话长度 - 中位数: {lens[len(lens)//2]}, P90: {lens[len(lens)*9//10]}, 最长: {lens[-1]}")


if __name__ == "__main__":
    main()
