#!/usr/bin/env python3
"""
多市场行情数据 → LLM训练数据（ChatML JSONL）
支持：A股、商品期货、ETF基金、可转债

每个品种有专属问答模板，生成领域特定的技术分析训练对。
"""

import json
import os
import glob
import random
import numpy as np
from collections import defaultdict

random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "training-data", "all_market_train.jsonl")
SYSTEM_PROMPT = "你是一个专业的量化交易专家，擅长技术分析、因子分析、趋势判断和风险管理。"

# 各品种数据目录和配额
MARKET_CONFIG = {
    "ashare": {
        "dir": os.path.join(PROJECT_ROOT, "training-data", "ashare", "advanced"),
        "label": "A股",
        "max_per_symbol": 3,
        "max_total": 5000,
    },
    "futures": {
        "dir": os.path.join(PROJECT_ROOT, "training-data", "futures", "advanced"),
        "label": "商品期货",
        "max_per_symbol": 5,  # 品种少，每个多生成点
        "max_total": 2000,
    },
    "etf": {
        "dir": os.path.join(PROJECT_ROOT, "training-data", "etf", "advanced"),
        "label": "ETF基金",
        "max_per_symbol": 3,
        "max_total": 2000,
    },
    "cbond": {
        "dir": os.path.join(PROJECT_ROOT, "training-data", "cbond", "advanced"),
        "label": "可转债",
        "max_per_symbol": 4,
        "max_total": 1500,
    },
}

# ============================================================
# 通用辅助函数
# ============================================================

def describe_rsi(rsi):
    if rsi >= 80: return "严重超买", "极高"
    elif rsi >= 70: return "超买", "偏高"
    elif rsi >= 55: return "偏强", "中性偏强"
    elif rsi >= 45: return "中性", "中性"
    elif rsi >= 30: return "偏弱", "中性偏弱"
    elif rsi >= 20: return "超卖", "偏低"
    else: return "严重超卖", "极低"

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

def calc_change_pct(close, open_price):
    if open_price == 0: return 0
    return round((close - open_price) / open_price * 100, 2)

def read_symbol_data(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                continue
    return rows

# ============================================================
# 通用技术分析模板（适用所有品种）
# ============================================================

def gen_technical_analysis(symbol, market_label, row, prev_row=None):
    rsi_desc, _ = describe_rsi(row["rsi_14"])
    prev_hist = prev_row["macd_histogram"] if prev_row else None
    macd_desc = describe_macd(row["macd_histogram"], prev_hist)
    change = calc_change_pct(row["close"], row["open"])

    ma_diff_pct = (row["close"] - row["close_ma_20"]) / row["close_ma_20"] * 100 if row["close_ma_20"] != 0 else 0

    question = (
        f"【{market_label}】{symbol} 在 {row['date']} 的技术指标如下：\n"
        f"- 收盘价 {row['close']}，当日涨跌 {change:+.2f}%\n"
        f"- RSI(14) = {row['rsi_14']}\n"
        f"- MACD柱线 = {row['macd_histogram']}，信号线 = {row['signal_line']}\n"
        f"- 20日均线 = {row['close_ma_20']}\n"
        f"- 成交量 {row['volume']:,}，5日均量 {row['volume_ma_5']:,}\n"
        f"请对该{market_label}标的当前技术面进行综合分析。"
    )

    above_ma = row["close"] > row["close_ma_20"]
    bullish_macd = row["macd_histogram"] > 0

    answer = f"从技术面来看，{symbol} 在 {row['date']} 呈现以下特征：\n\n"
    answer += f"**RSI指标**：RSI(14) 为 {row['rsi_14']}，处于{rsi_desc}区域。"
    if row["rsi_14"] >= 70:
        answer += "短期需警惕回调风险，不宜追高。\n\n"
    elif row["rsi_14"] <= 30:
        answer += "短期存在超跌反弹的可能。\n\n"
    else:
        answer += "动能处于正常范围。\n\n"

    answer += f"**MACD指标**：{macd_desc}。"
    answer += f"{'DIF在DEA上方，短期偏多' if row['macd_line'] > row['signal_line'] else 'DIF在DEA下方，短期偏空'}。\n\n"

    answer += f"**均线位置**：股价{'在' if above_ma else '在'}20日均线{'上方' if above_ma else '下方'}运行（偏离{ma_diff_pct:.1f}%），"
    answer += f"{'均线系统支撑有效' if above_ma else '均线形成压力位'}。\n\n"

    if row["volume_ma_5"] > 0:
        vol_ratio = row["volume"] / row["volume_ma_5"]
        answer += f"**量能分析**：成交量为5日均量的{vol_ratio:.1f}倍，"
        if vol_ratio > 1.5:
            answer += f"{'放量配合上涨为健康走势' if change > 0 else '放量下跌需警惕'}。\n\n"
        elif vol_ratio < 0.7:
            answer += "缩量运行说明市场观望情绪浓厚。\n\n"
        else:
            answer += "量能正常。\n\n"

    # 综合判断
    answer += "**综合判断**："
    if row["rsi_14"] >= 60 and bullish_macd and above_ma:
        answer += "多项指标共振偏多，但需注意RSI偏高的回调风险，建议逢低布局而非追涨。"
    elif row["rsi_14"] <= 35 and not bullish_macd:
        answer += "指标出现底部特征，可关注企稳信号。"
    elif bullish_macd != above_ma:
        answer += "指标信号不一致，建议观望为主，等待方向明确后再操作。"
    elif bullish_macd and above_ma:
        answer += "技术面整体偏多，可持仓待涨。"
    else:
        answer += "技术面整体偏空，建议轻仓或观望。"

    answer += "\n\n*以上分析仅基于技术指标，实际操作需结合基本面和市场环境综合判断。*"
    return question, answer

def gen_trend_analysis(symbol, market_label, rows):
    if len(rows) < 5:
        return None, None
    window = rows[-5:]
    closes = [r["close"] for r in window]
    volumes = [r["volume"] for r in window]
    total_change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0
    up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])

    data_lines = []
    for r in window:
        data_lines.append(f"  {r['date']}: 开{r['open']} 高{r['high']} 低{r['low']} 收{r['close']} 量{r['volume']:,}")

    question = (
        f"【{market_label}】以下是{market_label}标的 {symbol} 最近5个交易日的行情数据：\n"
        + "\n".join(data_lines)
        + f"\n请判断短期趋势方向，并给出操作建议。"
    )

    if total_change > 5 and up_days >= 4:
        trend, detail = "强势上涨", f"5日累计涨幅 {total_change:.1f}%，{up_days}阳{5-up_days-1}阴"
        advice = "短期动能充沛，但连续上涨后需关注获利回吐压力。"
    elif total_change > 2:
        trend, detail = "温和上涨", f"5日累计涨幅 {total_change:.1f}%"
        advice = "趋势向好但力度一般，适合轻仓跟随。"
    elif total_change < -5 and up_days <= 1:
        trend, detail = "持续下跌", f"5日累计跌幅 {total_change:.1f}%"
        advice = "下跌趋势明确，不宜抄底，等待止跌信号。"
    elif total_change < -2:
        trend, detail = "偏弱震荡", f"5日累计跌幅 {total_change:.1f}%"
        advice = "弱势格局，建议减仓或观望。"
    else:
        trend, detail = "横盘整理", f"5日累计变动 {total_change:+.1f}%"
        advice = "横盘整理阶段不宜重仓，等待放量突破方向明确后再跟随。"

    vol_trend = "放量" if volumes[-1] > volumes[0] * 1.3 else "缩量" if volumes[-1] < volumes[0] * 0.7 else "量能平稳"

    answer = (
        f"**趋势判断：{trend}**\n\n{detail}。\n\n"
        f"**量能特征**：近5日{vol_trend}。\n\n"
        f"**操作建议**：{advice}\n\n"
        f"*注：以上分析基于短期技术面，中长期操作需结合基本面研究。*"
    )
    return question, answer

# ============================================================
# 期货专属模板
# ============================================================

def gen_futures_basis_analysis(symbol, rows):
    """期货：持仓量+基差分析（模拟）"""
    if len(rows) < 10:
        return None, None

    recent = rows[-5:]
    closes = [r["close"] for r in recent]
    vols = [r["volume"] for r in recent]
    change = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] != 0 else 0

    # 量仓分析
    vol_change = (vols[-1] - vols[0]) / vols[0] * 100 if vols[0] != 0 else 0

    question = (
        f"【商品期货】商品期货 {symbol} 近5个交易日情况：\n"
        f"- 价格从 {closes[0]} 变动到 {closes[-1]}（{change:+.1f}%）\n"
        f"- 成交量变动：{vol_change:+.1f}%\n"
        f"- 最新RSI(14) = {recent[-1]['rsi_14']}\n"
        f"请从量价关系角度分析该品种的多空力量对比。"
    )

    if change > 0 and vol_change > 20:
        pattern = "增仓上行"
        analysis = "价格上涨伴随成交量放大，说明多头主动进场意愿强烈，上涨趋势得到资金确认。这是典型的多头增仓格局，短期看涨。"
        advice = "可顺势做多，止损设在近5日低点下方。"
    elif change > 0 and vol_change < -20:
        pattern = "减仓上行"
        analysis = "价格上涨但成交量萎缩，说明空头在平仓离场推动价格上行，而非新多入场。这种上涨持续性较差。"
        advice = "谨慎追多，上涨可能是空头回补造成的，关注量能是否能跟上。"
    elif change < 0 and vol_change > 20:
        pattern = "增仓下行"
        analysis = "价格下跌伴随成交量放大，说明空头主动进场做空，下跌动能充足。这是典型的空头增仓格局，短期看跌。"
        advice = "建议空仓观望或顺势做空，止损设在近期高点上方。"
    elif change < 0 and vol_change < -20:
        pattern = "减仓下行"
        analysis = "价格下跌但成交量萎缩，说明多头在平仓离场，但空头也没有加大力度。下跌动能在减弱，可能接近底部。"
        advice = "不宜追空，可开始关注企稳信号。"
    else:
        pattern = "多空均衡"
        analysis = "量价变化不显著，多空力量处于相对均衡状态，市场在等待新的驱动因素。"
        advice = "观望为主，等待突破方向确立后再操作。"

    answer = (
        f"**量价分析：{pattern}**\n\n"
        f"{analysis}\n\n"
        f"**RSI状态**：当前RSI为{recent[-1]['rsi_14']}，"
        + ("处于超买区域，即使趋势偏多也需警惕回调。" if recent[-1]["rsi_14"] > 70
           else "处于超卖区域，空头力量可能接近释放完毕。" if recent[-1]["rsi_14"] < 30
           else "处于正常范围。")
        + f"\n\n**操作建议**：{advice}\n\n"
        f"*期货带有杠杆，请严格控制仓位和止损。*"
    )
    return question, answer

def gen_futures_seasonality(symbol, rows):
    """期货：月度统计分析"""
    if len(rows) < 60:
        return None, None

    # 按月统计涨跌
    from collections import Counter
    monthly = defaultdict(list)
    for r in rows:
        month = r["date"][:7]
        monthly[month].append(r["close"])

    monthly_returns = {}
    sorted_months = sorted(monthly.keys())
    for i in range(1, len(sorted_months)):
        m = sorted_months[i]
        prev_m = sorted_months[i-1]
        ret = (monthly[m][-1] - monthly[prev_m][-1]) / monthly[prev_m][-1] * 100
        month_num = int(m.split("-")[1])
        if month_num not in monthly_returns:
            monthly_returns[month_num] = []
        monthly_returns[month_num].append(ret)

    if len(monthly_returns) < 6:
        return None, None

    # 找出表现最好和最差的月份
    avg_by_month = {m: np.mean(rets) for m, rets in monthly_returns.items() if len(rets) >= 2}
    if not avg_by_month:
        return None, None

    best_month = max(avg_by_month, key=avg_by_month.get)
    worst_month = min(avg_by_month, key=avg_by_month.get)

    question = (
        f"【商品期货】请分析商品期货 {symbol} 的历史月度表现规律。"
        f"该品种有 {len(rows)} 个交易日的历史数据。"
    )

    answer = (
        f"基于 {symbol} 的历史数据统计分析：\n\n"
        f"**月度表现规律**：\n"
    )
    for m in sorted(avg_by_month.keys()):
        avg = avg_by_month[m]
        count = len(monthly_returns[m])
        win_rate = sum(1 for r in monthly_returns[m] if r > 0) / count * 100
        answer += f"- {m}月：平均收益 {avg:+.2f}%，上涨概率 {win_rate:.0f}%（{count}年样本）\n"

    answer += (
        f"\n**季节性特征**：\n"
        f"- 历史表现最强月份：{best_month}月（平均 {avg_by_month[best_month]:+.2f}%）\n"
        f"- 历史表现最弱月份：{worst_month}月（平均 {avg_by_month[worst_month]:+.2f}%）\n\n"
        f"**使用建议**：\n"
        f"- 季节性规律可作为交易的辅助参考，但不应作为唯一依据\n"
        f"- 需结合当年供需基本面、宏观环境和技术面综合判断\n"
        f"- 样本量较少的月份统计意义有限，需谨慎解读\n\n"
        f"*历史表现不代表未来收益，季节性规律可能因市场结构变化而失效。*"
    )
    return question, answer

# ============================================================
# ETF 专属模板
# ============================================================

def gen_etf_tracking_analysis(symbol, rows):
    """ETF：波动率和走势分析"""
    if len(rows) < 20:
        return None, None

    recent = rows[-20:]
    closes = [r["close"] for r in recent]
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    volatility = np.std(returns) * np.sqrt(252) * 100
    total_return = (closes[-1] - closes[0]) / closes[0] * 100
    max_close = max(closes)
    drawdown = (closes[-1] - max_close) / max_close * 100

    question = (
        f"【ETF基金】ETF {symbol} 近20个交易日表现如下：\n"
        f"- 区间收益率: {total_return:+.2f}%\n"
        f"- 区间最大回撤: {drawdown:.2f}%\n"
        f"- 年化波动率: {volatility:.1f}%\n"
        f"- 最新RSI(14) = {recent[-1]['rsi_14']}\n"
        f"请分析该ETF的近期表现和配置价值。"
    )

    if volatility < 15:
        vol_desc = "低波动"
        vol_advice = "适合稳健型投资者，可作为底仓配置"
    elif volatility < 30:
        vol_desc = "中等波动"
        vol_advice = "波动适中，适合均衡型配置"
    else:
        vol_desc = "高波动"
        vol_advice = "波动较大，适合有经验的投资者，需控制仓位"

    answer = (
        f"**{symbol} 近期表现分析**\n\n"
        f"**收益与风险**：\n"
        f"- 近20日收益率 {total_return:+.2f}%，{'表现积极' if total_return > 0 else '表现疲软'}\n"
        f"- 最大回撤 {drawdown:.2f}%，{'风控表现良好' if abs(drawdown) < 5 else '需关注回撤控制'}\n"
        f"- 年化波动率 {volatility:.1f}%，属于{vol_desc}品种\n\n"
        f"**技术面**：\n"
        f"- RSI(14) = {recent[-1]['rsi_14']}，{describe_rsi(recent[-1]['rsi_14'])[0]}\n"
        f"- MACD柱线 = {recent[-1]['macd_histogram']}，{'偏多' if recent[-1]['macd_histogram'] > 0 else '偏空'}\n\n"
        f"**配置建议**：\n"
        f"- {vol_advice}\n"
        f"- {'当前处于相对低位，可考虑分批建仓' if total_return < -3 and recent[-1]['rsi_14'] < 40 else '当前位置适中，可常规定投' if abs(total_return) < 3 else '近期涨幅较大，建议等回调后再布局'}\n"
        f"- ETF适合长期持有和定投策略，短期波动不应影响长期配置计划\n\n"
        f"*ETF投资需关注跟踪标的的基本面变化和市场风格切换。*"
    )
    return question, answer

# ============================================================
# 可转债专属模板
# ============================================================

def gen_cbond_analysis(symbol, rows):
    """可转债：价格区间分析"""
    if len(rows) < 10:
        return None, None

    recent = rows[-10:]
    closes = [r["close"] for r in recent]
    latest = recent[-1]
    price = latest["close"]

    question = (
        f"【可转债】可转债 {symbol} 当前价格 {price}，RSI(14)={latest['rsi_14']}，"
        f"MACD柱线={latest['macd_histogram']}。\n"
        f"近10日价格区间: {min(closes):.2f} ~ {max(closes):.2f}\n"
        f"请分析该转债的投资价值和风险。"
    )

    if price < 100:
        zone = "折价区间"
        value_desc = (
            "转债价格低于面值100元，处于折价状态。此时债性保护较强，"
            "到期至少可获得面值兑付（假设不违约）。低价转债具有'下有保底'的特征，"
            "是可转债投资中相对安全的区域。"
        )
        advice = "低价转债适合防守型策略，可重点关注信用资质和到期收益率。若正股出现反转，还可能享受额外的转股收益。"
    elif price < 115:
        zone = "平价附近"
        value_desc = (
            "转债价格在面值附近，债性和股性相对均衡。此价格区间风险收益比较好，"
            "下跌空间有限而上涨潜力取决于正股表现。"
        )
        advice = "平价附近的转债性价比较高，可结合正股基本面选择优质标的进行配置。"
    elif price < 130:
        zone = "偏股区间"
        value_desc = (
            "转债价格已明显高于面值，股性增强，走势与正股关联度提高。"
            "转债的下行保护减弱，但仍有一定的债底支撑。"
        )
        advice = "需重点分析正股走势，转债价格主要跟随正股波动。可配合技术分析把握波段机会。"
    else:
        zone = "高价区间"
        value_desc = (
            f"转债价格已达 {price}，远高于面值，几乎完全跟随正股波动。"
            "此时债性保护已很弱，风险收益特征接近股票。需警惕强赎风险（发行人有权以面值+利息强制赎回）。"
        )
        advice = "高价转债风险较大，不建议重仓。若持有应密切关注强赎公告和正股走势，设置严格止损。"

    answer = (
        f"**{symbol} 投资分析**\n\n"
        f"**价格定位：{zone}**（当前价 {price}）\n\n"
        f"{value_desc}\n\n"
        f"**技术面**：\n"
        f"- RSI(14) = {latest['rsi_14']}，{describe_rsi(latest['rsi_14'])[0]}\n"
        f"- MACD {'偏多' if latest['macd_histogram'] > 0 else '偏空'}，动能"
        + ("正在增强" if latest["macd_histogram"] > 0 else "趋于减弱") + "\n"
        f"- 近10日振幅: {(max(closes) - min(closes)) / min(closes) * 100:.1f}%\n\n"
        f"**投资建议**：{advice}\n\n"
        f"*可转债投资需关注：正股走势、转股溢价率、剩余期限、强赎条款和信用评级。*"
    )
    return question, answer

def gen_cbond_strategy(symbol, rows):
    """可转债：双低策略分析"""
    if len(rows) < 20:
        return None, None

    latest = rows[-1]
    price = latest["close"]

    # 模拟转股溢价率（基于价格波动推测，实际需要正股数据）
    recent_vol = np.std([r["close"] for r in rows[-20:]]) / np.mean([r["close"] for r in rows[-20:]]) * 100

    question = (
        f"【可转债】请介绍可转债的'双低策略'，并以 {symbol}（当前价 {price}）为例分析是否符合双低条件。"
    )

    answer = (
        f"**可转债双低策略详解**\n\n"
        f"双低策略是可转债投资中最经典的量化策略之一，核心思路是寻找'价格低+溢价率低'的转债。\n\n"
        f"**策略原理**：\n"
        f"- **低价格**：转债价格越低，债底保护越强，下跌空间越小\n"
        f"- **低溢价率**：转股溢价率越低，转债与正股的联动性越强，正股上涨时转债跟涨能力越好\n"
        f"- **双低值** = 转债价格 + 转股溢价率×100，双低值越低越好\n\n"
        f"**筛选标准**（常用）：\n"
        f"- 转债价格 < 115元\n"
        f"- 转股溢价率 < 30%\n"
        f"- 双低值 < 150\n"
        f"- 排除：剩余期限<1年、评级AA-以下、正股ST\n\n"
        f"**{symbol} 初步分析**：\n"
        f"- 当前价格 {price}，{'符合低价条件（<115）' if price < 115 else '价格偏高，不符合低价条件'}\n"
        f"- 近期波动率 {recent_vol:.1f}%，{'波动适中' if recent_vol < 5 else '波动偏大'}\n"
        f"- 需要结合正股价格计算实际转股溢价率后才能完整判断\n\n"
        f"**策略执行要点**：\n"
        f"1. 每月或每两周调仓一次，按双低值排名选取前20-30只等权持有\n"
        f"2. 严格分散，单只转债仓位不超过5%\n"
        f"3. 回避有信用风险的标的（关注正股基本面和评级变动）\n"
        f"4. 历史年化收益率约15-25%，最大回撤通常在10%以内\n\n"
        f"*双低策略的超额收益来源于市场对低价转债的定价偏差，随着策略普及可能逐步衰减。*"
    )
    return question, answer

# ============================================================
# 结构化交易信号模板（JSON 输出格式）
# ============================================================

def gen_trading_signal(symbol, market_label, row, prev_row=None):
    """生成结构化交易信号（教模型输出 JSON 格式决策）"""
    rsi = row["rsi_14"]
    macd_hist = row["macd_histogram"]
    above_ma = row["close"] > row["close_ma_20"]
    change = calc_change_pct(row["close"], row["open"])
    vol_ratio = row["volume"] / row["volume_ma_5"] if row["volume_ma_5"] > 0 else 1.0

    question = (
        f"【{market_label}】请根据以下行情数据，给出 {symbol} 在 {row['date']} 的交易信号。"
        f"要求以 JSON 格式输出，包含 action、reasons、confidence 字段。\n"
        f"- 收盘价: {row['close']}，涨跌: {change:+.2f}%\n"
        f"- RSI(14) = {rsi}\n"
        f"- MACD柱线 = {macd_hist}\n"
        f"- 20日均线 = {row['close_ma_20']}\n"
        f"- 成交量/5日均量 = {vol_ratio:.2f}\n"
    )

    # 决策逻辑
    reasons = []
    bullish_count = 0
    bearish_count = 0

    if rsi >= 70:
        reasons.append(f"RSI={rsi}，处于超买区域，有回调风险")
        bearish_count += 1
    elif rsi <= 30:
        reasons.append(f"RSI={rsi}，处于超卖区域，存在反弹机会")
        bullish_count += 1
    elif rsi >= 50:
        reasons.append(f"RSI={rsi}，动能偏强")
        bullish_count += 0.5
    else:
        reasons.append(f"RSI={rsi}，动能偏弱")
        bearish_count += 0.5

    if macd_hist > 0:
        reasons.append("MACD柱线为正，多头区间")
        bullish_count += 1
    else:
        reasons.append("MACD柱线为负，空头区间")
        bearish_count += 1

    if above_ma:
        reasons.append("价格在20日均线上方，趋势偏多")
        bullish_count += 1
    else:
        reasons.append("价格在20日均线下方，趋势偏空")
        bearish_count += 1

    if vol_ratio > 1.5:
        reasons.append(f"成交量放大({vol_ratio:.1f}倍均量)，市场活跃")
    elif vol_ratio < 0.6:
        reasons.append(f"成交量萎缩({vol_ratio:.1f}倍均量)，观望情绪浓")

    # 综合判断
    if bullish_count >= 2.5 and rsi < 75:
        action = "buy"
        confidence = min(0.85, 0.5 + bullish_count * 0.1)
        reasons.append("多项指标共振偏多，建议买入")
    elif bearish_count >= 2.5 and rsi > 25:
        action = "sell"
        confidence = min(0.85, 0.5 + bearish_count * 0.1)
        reasons.append("多项指标共振偏空，建议卖出")
    else:
        action = "hold"
        confidence = 0.5
        reasons.append("多空信号不一致，建议持仓观望")

    signal = {
        "action": action,
        "symbol": symbol,
        "date": row["date"],
        "reasons": reasons,
        "confidence": round(confidence, 2),
        "risk_note": "以上信号仅基于技术指标，需结合基本面和仓位管理综合决策"
    }

    answer = (
        f"根据{symbol}在{row['date']}的技术指标综合分析，交易信号如下：\n\n"
        f"```json\n{json.dumps(signal, ensure_ascii=False, indent=2)}\n```\n\n"
        f"**信号解读**：\n"
    )
    for r in reasons:
        answer += f"- {r}\n"
    answer += f"\n*置信度 {confidence:.0%}，{'信号较强' if confidence >= 0.7 else '信号一般，建议结合其他因素确认'}。*"

    return question, answer


# ============================================================
# 交易评分模板（0-100 分制，固定输入格式）
# ============================================================

def _detect_regime(all_rows, current_idx):
    """
    从个股历史数据推导牛/熊/震荡环境。
    用 close vs 120日均线 + 均线方向 近似判断。
    返回: ("牛市"|"熊市"|"震荡", ma120, ma120_slope_pct)
    """
    ma_window = 120
    if current_idx < ma_window:
        return "震荡", None, 0

    closes_120 = [all_rows[i]["close"] for i in range(current_idx - ma_window + 1, current_idx + 1)]
    ma120 = sum(closes_120) / len(closes_120)

    # 均线方向：用 20 日前的 MA120 对比
    if current_idx >= ma_window + 20:
        closes_120_prev = [all_rows[i]["close"] for i in range(current_idx - ma_window - 19, current_idx - 19)]
        ma120_prev = sum(closes_120_prev) / len(closes_120_prev)
        ma120_slope = (ma120 - ma120_prev) / ma120_prev * 100
    else:
        ma120_slope = 0

    current_close = all_rows[current_idx]["close"]

    if current_close > ma120 and ma120_slope > 0.5:
        return "牛市", round(ma120, 2), round(ma120_slope, 2)
    elif current_close < ma120 and ma120_slope < -0.5:
        return "熊市", round(ma120, 2), round(ma120_slope, 2)
    else:
        return "震荡", round(ma120, 2), round(ma120_slope, 2)


def gen_trading_score(symbol, market_label, row, rows_context, prev_row=None,
                      all_rows=None, current_idx=None):
    """
    生成交易评分训练样本（0-100分制）。
    固定输入格式：[MARKET_DATA] [MARKET_REGIME] [POSITION] [PORTFOLIO] 分段标记。
    输出 JSON：score + reasons + risk_factors。
    牛市倾向激进买入（加分），熊市倾向保守（减分）。
    """
    rsi = row["rsi_14"]
    macd_hist = row["macd_histogram"]
    macd_line = row["macd_line"]
    signal_line = row["signal_line"]
    above_ma = row["close"] > row["close_ma_20"]
    change = calc_change_pct(row["close"], row["open"])
    vol_ratio = row["volume"] / row["volume_ma_5"] if row["volume_ma_5"] > 0 else 1.0
    ma_diff_pct = (row["close"] - row["close_ma_20"]) / row["close_ma_20"] * 100 if row["close_ma_20"] != 0 else 0

    # ---- 市场环境判断 ----
    if all_rows is not None and current_idx is not None:
        regime, ma120, ma120_slope = _detect_regime(all_rows, current_idx)
    else:
        regime, ma120, ma120_slope = "震荡", None, 0

    regime_block = f"[MARKET_REGIME]\nregime: {regime}\n"
    if ma120 is not None:
        regime_block += f"ma120: {ma120}\nma120_slope: {ma120_slope:+.2f}%\n"

    # 模拟持仓状态（随机生成，让模型学会处理有/无持仓的场景）
    import hashlib
    seed_val = int(hashlib.md5(f"{symbol}{row['date']}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_val)

    holding = rng.random() < 0.4  # 40% 概率有持仓
    if holding:
        cost_offset = rng.uniform(-0.08, 0.08)
        cost = round(row["close"] * (1 - cost_offset), 2)
        pnl_pct = round((row["close"] - cost) / cost * 100, 2)
        days_held = rng.randint(1, 30)
        position_block = (
            f"[POSITION]\n"
            f"holding: true\n"
            f"cost: {cost}\n"
            f"pnl_pct: {pnl_pct:+.2f}%\n"
            f"days_held: {days_held}\n"
        )
    else:
        pnl_pct = 0
        days_held = 0
        position_block = (
            f"[POSITION]\n"
            f"holding: false\n"
        )

    cash_pct = round(rng.uniform(0.15, 0.85), 2)
    total_assets = rng.choice([50000, 100000, 200000, 500000])

    # 近5日趋势
    if len(rows_context) >= 5:
        recent_closes = [r["close"] for r in rows_context[-5:]]
        trend_5d = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100 if recent_closes[0] != 0 else 0
    else:
        trend_5d = change

    # ---- 固定格式输入 ----
    question = (
        f"[MARKET_DATA]\n"
        f"symbol: {symbol}\n"
        f"market: {market_label}\n"
        f"date: {row['date']}\n"
        f"close: {row['close']}\n"
        f"change_pct: {change:+.2f}%\n"
        f"trend_5d: {trend_5d:+.2f}%\n"
        f"rsi_14: {rsi}\n"
        f"macd_histogram: {macd_hist}\n"
        f"macd_cross: {'金叉' if macd_line > signal_line else '死叉'}\n"
        f"above_ma20: {'true' if above_ma else 'false'}\n"
        f"ma20_diff_pct: {ma_diff_pct:+.1f}%\n"
        f"volume_ratio: {vol_ratio:.2f}\n"
        f"{regime_block}"
        f"{position_block}"
        f"[PORTFOLIO]\n"
        f"total_assets: {total_assets}\n"
        f"cash_pct: {cash_pct}\n"
        f"[END]\n"
        f"请输出交易评分(0-100)和理由。"
    )

    # ---- 评分逻辑（覆盖全分段） ----
    score = 50  # 基准分
    reasons = []
    risk_factors = []

    # RSI 贡献 (±20)
    if rsi >= 80:
        score -= 20
        reasons.append(f"RSI={rsi}，严重超买")
        risk_factors.append("超买回调风险极高")
    elif rsi >= 70:
        score -= 12
        reasons.append(f"RSI={rsi}，超买区域")
        risk_factors.append("短期有回调压力")
    elif rsi >= 55:
        score += 5
        reasons.append(f"RSI={rsi}，动能偏强")
    elif rsi >= 45:
        pass  # 中性不加分
    elif rsi >= 30:
        score -= 5
        reasons.append(f"RSI={rsi}，动能偏弱")
    elif rsi >= 20:
        score += 12
        reasons.append(f"RSI={rsi}，超卖区域，存在反弹机会")
    else:
        score += 18
        reasons.append(f"RSI={rsi}，严重超卖，反弹概率较大")

    # MACD 贡献 (±12)
    if macd_hist > 0 and (prev_row and macd_hist > prev_row["macd_histogram"]):
        score += 12
        reasons.append("MACD柱线为正且放大，多头动能增强")
    elif macd_hist > 0:
        score += 6
        reasons.append("MACD柱线为正，处于多头区间")
    elif macd_hist < 0 and (prev_row and macd_hist < prev_row["macd_histogram"]):
        score -= 12
        reasons.append("MACD柱线为负且放大，空头动能增强")
        risk_factors.append("MACD空头加速")
    else:
        score -= 6
        reasons.append("MACD柱线为负，处于空头区间")

    # MACD 金叉/死叉 (±5)
    if macd_line > signal_line:
        score += 5
        reasons.append("MACD金叉，短期偏多")
    else:
        score -= 5
        reasons.append("MACD死叉，短期偏空")

    # 均线位置 (±8)
    if above_ma and ma_diff_pct > 3:
        score += 8
        reasons.append(f"价格在20日均线上方{ma_diff_pct:.1f}%，趋势强势")
    elif above_ma:
        score += 4
        reasons.append("价格在20日均线上方，趋势偏多")
    elif ma_diff_pct < -3:
        score -= 8
        reasons.append(f"价格在20日均线下方{abs(ma_diff_pct):.1f}%，趋势弱势")
        risk_factors.append("远离均线支撑")
    else:
        score -= 4
        reasons.append("价格在20日均线下方，趋势偏空")

    # 量能 (±5)
    if vol_ratio > 2.0 and change > 0:
        score += 5
        reasons.append(f"放量上涨({vol_ratio:.1f}倍均量)，资金积极进场")
    elif vol_ratio > 2.0 and change < 0:
        score -= 5
        reasons.append(f"放量下跌({vol_ratio:.1f}倍均量)，抛压沉重")
        risk_factors.append("放量下跌")
    elif vol_ratio < 0.5:
        risk_factors.append(f"严重缩量({vol_ratio:.1f}倍均量)，流动性不足")

    # 5日趋势 (±5)
    if trend_5d > 5:
        score += 5
        reasons.append(f"近5日涨幅{trend_5d:.1f}%，短期强势")
    elif trend_5d < -5:
        score -= 5
        reasons.append(f"近5日跌幅{trend_5d:.1f}%，短期弱势")
        risk_factors.append("连续下跌趋势")

    # 持仓盈亏影响 (±5)
    if holding:
        if pnl_pct > 10:
            risk_factors.append(f"持仓浮盈{pnl_pct:.1f}%，注意保护利润")
        elif pnl_pct < -5:
            score -= 5
            risk_factors.append(f"持仓浮亏{pnl_pct:.1f}%，关注止损")

    # ---- 牛熊环境调整 ----
    # 牛市：激进买入（整体加分），熊市：保守买入（整体减分）
    if regime == "牛市":
        regime_adj = 8
        score += regime_adj
        reasons.append(f"当前处于牛市环境（MA120向上{ma120_slope:+.1f}%），策略偏激进")
    elif regime == "熊市":
        regime_adj = -8
        score += regime_adj
        reasons.append(f"当前处于熊市环境（MA120向下{ma120_slope:+.1f}%），策略偏保守")
        risk_factors.append("熊市环境下需严格止损，控制仓位")
    else:
        reasons.append("当前处于震荡市，策略中性")

    # 钳制到 [0, 100]
    score = max(0, min(100, score))

    # 补充通用风险因子
    if cash_pct < 0.2:
        risk_factors.append("现金比例偏低，加仓空间有限")

    if not risk_factors:
        risk_factors.append("当前无显著风险信号")

    # ---- 输出 ----
    output_json = {
        "score": score,
        "reasons": reasons,
        "risk_factors": risk_factors,
    }

    answer = (
        f"```json\n{json.dumps(output_json, ensure_ascii=False, indent=2)}\n```\n\n"
        f"**评分解读**：交易评分 {score}/100，"
    )

    if score >= 70:
        answer += "多项指标共振偏多，买入信号较强。"
    elif score >= 55:
        answer += "偏多但信号不够强烈，可轻仓试探或持仓待涨。"
    elif score >= 45:
        answer += "多空均衡，建议观望等待方向明确。"
    elif score >= 30:
        answer += "偏空信号，建议减仓或观望。"
    else:
        answer += "强烈看空信号，建议卖出或空仓回避。"

    if risk_factors and risk_factors[0] != "当前无显著风险信号":
        answer += "\n\n**风险提示**：" + "；".join(risk_factors) + "。"

    answer += "\n\n*评分仅基于技术指标，实际决策需结合基本面、市场环境和仓位管理。*"

    return question, answer


# ============================================================
# 主流程
# ============================================================

def process_market(market_key, config):
    """处理单个市场的数据"""
    data_dir = config["dir"]
    label = config["label"]
    max_per = config["max_per_symbol"]
    max_total = config["max_total"]

    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        print(f"  [{label}] 无数据文件，跳过")
        return []

    print(f"  [{label}] {len(files)} 个标的")
    records = []
    stats = defaultdict(int)

    for filepath in files:
        if len(records) >= max_total:
            break

        symbol = os.path.basename(filepath).replace(".jsonl", "")
        rows = read_symbol_data(filepath)
        if len(rows) < 30:
            continue

        stock_records = []
        half = len(rows) // 2
        candidates = list(range(half, len(rows)))
        random.shuffle(candidates)

        for idx in candidates:
            if len(stock_records) >= max_per:
                break

            row = rows[idx]
            prev_rows = rows[max(0, idx-5):idx]
            prev_row = rows[idx-1] if idx > 0 else None

            q, a = None, None
            ttype = None
            r = random.random()

            # 各模板分配比例：技术分析 / 趋势 / 信号JSON / 评分0-100 / 品种专属
            rows_ctx = rows[max(0, idx-5):idx+1]

            if market_key == "futures":
                if r < 0.20:
                    q, a = gen_technical_analysis(symbol, label, row, prev_row)
                    ttype = "technical"
                elif r < 0.32:
                    q, a = gen_trend_analysis(symbol, label, rows[max(0, idx-4):idx+1])
                    ttype = "trend"
                elif r < 0.42:
                    q, a = gen_trading_signal(symbol, label, row, prev_row)
                    ttype = "signal"
                elif r < 0.55:
                    q, a = gen_trading_score(symbol, label, row, rows_ctx, prev_row, all_rows=rows, current_idx=idx)
                    ttype = "score"
                elif r < 0.78:
                    q, a = gen_futures_basis_analysis(symbol, rows[max(0, idx-10):idx+1])
                    ttype = "futures_basis"
                else:
                    q, a = gen_futures_seasonality(symbol, rows[:idx+1])
                    ttype = "futures_season"
            elif market_key == "etf":
                if r < 0.22:
                    q, a = gen_technical_analysis(symbol, label, row, prev_row)
                    ttype = "technical"
                elif r < 0.35:
                    q, a = gen_trend_analysis(symbol, label, rows[max(0, idx-4):idx+1])
                    ttype = "trend"
                elif r < 0.47:
                    q, a = gen_trading_signal(symbol, label, row, prev_row)
                    ttype = "signal"
                elif r < 0.62:
                    q, a = gen_trading_score(symbol, label, row, rows_ctx, prev_row, all_rows=rows, current_idx=idx)
                    ttype = "score"
                else:
                    q, a = gen_etf_tracking_analysis(symbol, rows[max(0, idx-19):idx+1])
                    ttype = "etf_tracking"
            elif market_key == "cbond":
                if r < 0.18:
                    q, a = gen_technical_analysis(symbol, label, row, prev_row)
                    ttype = "technical"
                elif r < 0.30:
                    q, a = gen_trend_analysis(symbol, label, rows[max(0, idx-4):idx+1])
                    ttype = "trend"
                elif r < 0.40:
                    q, a = gen_trading_signal(symbol, label, row, prev_row)
                    ttype = "signal"
                elif r < 0.52:
                    q, a = gen_trading_score(symbol, label, row, rows_ctx, prev_row, all_rows=rows, current_idx=idx)
                    ttype = "score"
                elif r < 0.78:
                    q, a = gen_cbond_analysis(symbol, rows[max(0, idx-9):idx+1])
                    ttype = "cbond_analysis"
                else:
                    q, a = gen_cbond_strategy(symbol, rows[max(0, idx-19):idx+1])
                    ttype = "cbond_strategy"
            else:  # ashare
                if r < 0.22:
                    q, a = gen_technical_analysis(symbol, label, row, prev_row)
                    ttype = "technical"
                elif r < 0.38:
                    q, a = gen_trend_analysis(symbol, label, rows[max(0, idx-4):idx+1])
                    ttype = "trend"
                elif r < 0.55:
                    q, a = gen_trading_signal(symbol, label, row, prev_row)
                    ttype = "signal"
                else:
                    q, a = gen_trading_score(symbol, label, row, rows_ctx, prev_row, all_rows=rows, current_idx=idx)
                    ttype = "score"

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

        records.extend(stock_records)

    if len(records) > max_total:
        random.shuffle(records)
        records = records[:max_total]

    print(f"  [{label}] 生成 {len(records)} 条训练数据")
    for k, v in sorted(stats.items()):
        print(f"    {k}: {v}")

    return records


def main():
    print("=" * 60)
    print("多市场行情 → 训练数据转换")
    print("=" * 60)

    all_records = []

    for market_key, config in MARKET_CONFIG.items():
        records = process_market(market_key, config)
        all_records.extend(records)

    random.shuffle(all_records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 60}")
    print(f"全部转换完成！")
    print(f"  总条数: {len(all_records)}")
    print(f"  输出: {OUTPUT_FILE}")

    if all_records:
        lens = sorted(sum(len(m["content"]) for m in r["messages"]) for r in all_records)
        print(f"  对话长度 - 中位数: {lens[len(lens)//2]}, P90: {lens[len(lens)*9//10]}, 最长: {lens[-1]}")


if __name__ == "__main__":
    main()
