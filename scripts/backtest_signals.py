#!/usr/bin/env python3
"""
回测系统 -- 验证模型信号的历史表现

两套策略：
  1. 个股策略（每日选股，T+1，止损）
  2. 行业ETF轮动（每周再平衡）

对标基准：沪深300ETF (510300)

用法:
  python backtest_signals.py                   # 使用 config.yaml 中 backtest 配置
  python backtest_signals.py --start 2025-06-01 --end 2025-12-31
"""

import os
import sys
import json
import csv
import glob
import math
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

# ---- 项目配置 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg, PROJECT_ROOT

# ---- 默认回测参数 ----
BT_DEFAULTS = {
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "initial_capital": 100_000,
    "benchmark": "510300",
    "rebalance_freq": "weekly",
    "max_positions": 10,
}

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

RISK_FREE_RATE = 0.02  # 年化无风险利率


# ============================================================
# 数据加载
# ============================================================

def load_jsonl(filepath):
    """加载 JSONL 文件，返回行列表（按日期排序）"""
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


def load_all_stocks(data_dir, date_range):
    """
    加载所有个股数据，返回 {symbol: [rows]} 以及
    {date: {symbol: row}} 的倒排索引（仅保留日期范围内的行）
    """
    start, end = date_range
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    symbol_data = {}      # symbol -> all rows (for regime detection)
    date_index = defaultdict(dict)  # date -> {symbol: row}

    for fp in files:
        rows = load_jsonl(fp)
        if not rows:
            continue
        sym = rows[0].get("symbol", os.path.basename(fp).replace(".jsonl", ""))
        symbol_data[sym] = rows
        for r in rows:
            d = r.get("date", "")
            if start <= d <= end:
                date_index[d][sym] = r

    return symbol_data, date_index


def load_etf(data_dir, code):
    """加载单只 ETF 的全部数据"""
    fp = os.path.join(data_dir, f"ETF.{code}.jsonl")
    if not os.path.exists(fp):
        # 尝试不带 ETF. 前缀
        fp2 = os.path.join(data_dir, f"{code}.jsonl")
        if os.path.exists(fp2):
            fp = fp2
        else:
            return []
    return load_jsonl(fp)


def load_all_etfs(data_dir, codes, date_range):
    """加载多只 ETF"""
    start, end = date_range
    etf_data = {}       # code -> all rows
    date_index = defaultdict(dict)

    for code in codes:
        rows = load_etf(data_dir, code)
        if not rows:
            continue
        etf_data[code] = rows
        for r in rows:
            d = r.get("date", "")
            if start <= d <= end:
                date_index[d][code] = r
    return etf_data, date_index


# ============================================================
# 市场环境判定 (移植自 convert_all_to_training.py _detect_regime)
# ============================================================

def detect_regime(all_rows, current_idx):
    """
    多指标融合判定牛/熊/震荡。
    返回 (regime_str, ma120, ma120_slope, detail_dict)
    """
    ma_window = 120
    if current_idx < ma_window:
        return "震荡", None, 0, {"regime_score": 0}

    regime_score = 0
    detail = {}

    # 1. MA120 位置
    closes_120 = [all_rows[i]["close"] for i in range(current_idx - ma_window + 1, current_idx + 1)]
    ma120 = sum(closes_120) / len(closes_120)
    current_close = all_rows[current_idx]["close"]
    ma120_pct = (current_close - ma120) / ma120 * 100

    if ma120_pct > 5:
        regime_score += 1
    elif ma120_pct > 0:
        regime_score += 0.5
    elif ma120_pct < -5:
        regime_score -= 1
    elif ma120_pct < 0:
        regime_score -= 0.5

    detail["ma120"] = round(ma120, 2)

    # 2. MA120 斜率
    if current_idx >= ma_window + 20:
        closes_120_prev = [all_rows[i]["close"] for i in range(current_idx - ma_window - 19, current_idx - 19)]
        ma120_prev = sum(closes_120_prev) / len(closes_120_prev)
        ma120_slope = (ma120 - ma120_prev) / ma120_prev * 100
    else:
        ma120_slope = 0

    if ma120_slope > 1.0:
        regime_score += 1
    elif ma120_slope > 0.3:
        regime_score += 0.5
    elif ma120_slope < -1.0:
        regime_score -= 1
    elif ma120_slope < -0.3:
        regime_score -= 0.5

    detail["ma120_slope"] = round(ma120_slope, 2)

    # 3. 量能趋势
    if current_idx >= 80:
        vol_recent = np.mean([all_rows[i]["volume"] for i in range(current_idx - 19, current_idx + 1)])
        vol_long = np.mean([all_rows[i]["volume"] for i in range(current_idx - 79, current_idx - 19)])
        vol_ratio = vol_recent / vol_long if vol_long > 0 else 1.0

        if vol_ratio > 1.3:
            regime_score += 1
        elif vol_ratio > 1.1:
            regime_score += 0.5
        elif vol_ratio < 0.7:
            regime_score -= 0.5
        detail["vol_trend"] = round(vol_ratio, 2)
    else:
        detail["vol_trend"] = None

    # 4. 波动率状态
    if current_idx >= 80:
        returns_recent = []
        for i in range(current_idx - 19, current_idx + 1):
            if i > 0 and all_rows[i - 1]["close"] > 0:
                returns_recent.append(
                    (all_rows[i]["close"] - all_rows[i - 1]["close"]) / all_rows[i - 1]["close"]
                )
        returns_long = []
        for i in range(current_idx - 79, current_idx - 19):
            if i > 0 and all_rows[i - 1]["close"] > 0:
                returns_long.append(
                    (all_rows[i]["close"] - all_rows[i - 1]["close"]) / all_rows[i - 1]["close"]
                )
        if returns_recent and returns_long:
            vol_recent_std = np.std(returns_recent)
            vol_long_std = np.std(returns_long)
            vol_ratio_std = vol_recent_std / vol_long_std if vol_long_std > 0 else 1.0

            if vol_ratio_std > 1.8:
                regime_score -= 1
            elif vol_ratio_std > 1.3:
                regime_score -= 0.5
            elif vol_ratio_std < 0.7:
                regime_score += 0.5
            detail["volatility_ratio"] = round(vol_ratio_std, 2)
        else:
            detail["volatility_ratio"] = None
    else:
        detail["volatility_ratio"] = None

    # 5. 60日价格动量
    if current_idx >= 60:
        close_60ago = all_rows[current_idx - 60]["close"]
        momentum_60d = (current_close - close_60ago) / close_60ago * 100 if close_60ago > 0 else 0

        if momentum_60d > 15:
            regime_score += 1
        elif momentum_60d > 5:
            regime_score += 0.5
        elif momentum_60d < -15:
            regime_score -= 1
        elif momentum_60d < -5:
            regime_score -= 0.5

        detail["momentum_60d"] = round(momentum_60d, 2)
    else:
        detail["momentum_60d"] = None

    detail["regime_score"] = round(regime_score, 1)

    if regime_score >= 2.0:
        regime = "牛市"
    elif regime_score <= -2.0:
        regime = "熊市"
    else:
        regime = "震荡"

    return regime, round(ma120, 2), round(ma120_slope, 2), detail


def find_row_index(all_rows, date_str):
    """二分查找日期对应的索引，返回 -1 如果不存在"""
    lo, hi = 0, len(all_rows) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        d = all_rows[mid].get("date", "")
        if d == date_str:
            return mid
        elif d < date_str:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


# ============================================================
# 评分代理 (移植自 gen_trading_score 完整逻辑)
# ============================================================

def compute_score(row, all_rows, current_idx, prev_row=None, is_etf=False,
                   regime_info=None):
    """
    基于 IC 分析的反转型评分系统 v2。

    核心发现（factor_ic_analysis.py 验证）：
    - A 股所有技术因子 IC 为负 → 反转效应主导
    - 因子值高 → 未来收益低（应减分）
    - 因子分组：动量反转组、波动率组、量能组，每组设贡献上限

    regime_info: 预计算的 (regime, ma120, ma120_slope, detail)，避免重复计算
    返回 score (0-100), regime
    """
    rsi = row.get("rsi_14", 50) or 50
    macd_hist = row.get("macd_histogram", 0) or 0
    macd_line = row.get("macd_line", 0) or 0
    close_ma_20 = row.get("close_ma_20", row["close"]) or row["close"]
    change = (row["close"] - row["open"]) / row["open"] * 100 if row["open"] != 0 else 0
    vol_ma_5 = row.get("volume_ma_5", 0) or 0
    vol_ratio = row["volume"] / vol_ma_5 if vol_ma_5 > 0 else 1.0
    ma_diff_pct = (row["close"] - close_ma_20) / close_ma_20 * 100 if close_ma_20 != 0 else 0

    # 市场环境（优先用外部传入的，避免重复计算）
    if regime_info:
        regime, ma120, ma120_slope, regime_detail = regime_info
    else:
        regime, ma120, ma120_slope, regime_detail = detect_regime(all_rows, current_idx)
    rs = regime_detail.get("regime_score", 0)

    # 近期趋势
    trend_5d = 0
    trend_20d = 0
    if current_idx >= 5:
        c5 = all_rows[current_idx - 5]["close"]
        trend_5d = (row["close"] - c5) / c5 * 100 if c5 > 0 else 0
    if current_idx >= 20:
        c20 = all_rows[current_idx - 20]["close"]
        trend_20d = (row["close"] - c20) / c20 * 100 if c20 > 0 else 0

    # 换手率、总股本
    turnover_rate = row.get("turnover_rate", 0) or 0
    total_shares = row.get("total_shares", 0) or 0

    # 新增指标
    roc_12 = row.get("roc_12", 0) or 0
    cci_20 = row.get("cci_20", 0) or 0
    bb_position = row.get("bb_position", 0.5) or 0.5
    mfi_14 = row.get("mfi_14", 50) or 50
    hv_20 = row.get("hv_20", 0) or 0
    atr_14 = row.get("atr_14", 0) or 0
    ma_alignment = row.get("ma_alignment", "mixed") or "mixed"

    # ============================================================
    # 分组评分（反转逻辑：因子越高越看空）
    # ============================================================
    score = 50  # 基准

    # ---- 组1: 动量反转组（上限 ±20）----
    # IC_IR: rsi=-0.40, roc_12=-0.39, trend_20d=-0.46, macd_line=-0.45
    momentum_score = 0

    # RSI 反转（IC_IR=-0.40，权重 ±8）— 极化阈值
    if rsi >= 80:
        momentum_score -= 8  # 严重超买 → 看空
    elif rsi >= 65:
        momentum_score -= 3
    elif rsi <= 20:
        momentum_score += 8  # 严重超卖 → 看多（反转买入）
    elif rsi <= 35:
        momentum_score += 3

    # 20日反转因子（IC_IR=-0.46，权重 ±7，最强反转信号）
    if trend_20d > 20:
        momentum_score -= 7  # 涨太多 → 看空
    elif trend_20d > 8:
        momentum_score -= 3
    elif trend_20d < -20:
        momentum_score += 7  # 跌太多 → 看多
    elif trend_20d < -8:
        momentum_score += 3

    # ROC 反转（IC_IR=-0.39，权重 ±5）
    if roc_12 > 20:
        momentum_score -= 5
    elif roc_12 > 8:
        momentum_score -= 2
    elif roc_12 < -20:
        momentum_score += 5
    elif roc_12 < -8:
        momentum_score += 2

    # 组上限 ±20
    momentum_score = max(-20, min(20, momentum_score))
    score += momentum_score

    # ---- 组2: 趋势确认组（上限 ±12）----
    # IC_IR: ma20_diff=-0.38, macd_hist=-0.22, cci=-0.30
    trend_score = 0

    # MA20 偏离（IC_IR=-0.38，权重 ±5）— 反转
    if ma_diff_pct > 8:
        trend_score -= 5  # 远离均线上方 → 回归压力
    elif ma_diff_pct > 3:
        trend_score -= 2
    elif ma_diff_pct < -8:
        trend_score += 5  # 远离均线下方 → 反弹
    elif ma_diff_pct < -3:
        trend_score += 2

    # CCI 极值（IC_IR=-0.30，权重 ±4）— 反转
    if cci_20 > 150:
        trend_score -= 4
    elif cci_20 > 100:
        trend_score -= 2
    elif cci_20 < -150:
        trend_score += 4
    elif cci_20 < -100:
        trend_score += 2

    # MACD 柱线（IC_IR=-0.22，权重 ±3）— 弱反转
    if macd_hist > 0.5:
        trend_score -= 2
    elif macd_hist < -0.5:
        trend_score += 2

    # MFI 资金流（IC_IR=-0.31，权重 ±4）— 反转
    if mfi_14 > 75:
        trend_score -= 3
    elif mfi_14 < 25:
        trend_score += 3

    # 组上限 ±12
    trend_score = max(-12, min(12, trend_score))
    score += trend_score

    # ---- 组3: 波动率/风险组（上限 ±10）----
    # IC_IR: hv=-0.42, atr=-0.43（高波动→未来收益差）
    vol_score = 0

    # HV 波动率（IC_IR=-0.42，权重 ±5）
    if hv_20 > 60:
        vol_score -= 5  # 高波动 → 风险大，减分
    elif hv_20 > 40:
        vol_score -= 2
    elif hv_20 < 15:
        vol_score += 3  # 低波动 → 可能酝酿突破

    # BB 位置（IC_IR=-0.31，权重 ±4）— 反转
    if bb_position > 0.9:
        vol_score -= 4  # 布林上轨 → 看空
    elif bb_position < 0.1:
        vol_score += 4  # 布林下轨 → 看多

    # 组上限 ±10
    vol_score = max(-10, min(10, vol_score))
    score += vol_score

    # ---- 组4: 量能组（上限 ±8）----
    # IC_IR: volume_ma_5=-0.67（最强！成交量越大未来越跌）
    volume_score = 0

    # 量比反转（IC_IR=-0.21）
    if vol_ratio > 3.0:
        volume_score -= 4  # 巨量 → 见顶信号
    elif vol_ratio > 2.0:
        volume_score -= 2
    elif vol_ratio < 0.4:
        volume_score += 2  # 极度缩量 → 筑底

    # 5日趋势（IC_IR=-0.18，弱信号）
    if trend_5d > 8:
        volume_score -= 3
    elif trend_5d < -8:
        volume_score += 3

    # 组上限 ±8
    volume_score = max(-8, min(8, volume_score))
    score += volume_score

    # ---- 市场环境调整（±8）----
    if regime == "牛市":
        regime_adj = 8 if rs >= 3.5 else 5
        score += regime_adj
    elif regime == "熊市":
        regime_adj = -8 if rs <= -3.5 else -5
        score += regime_adj

    # ---- 组5: 困境反转组（上限 ±15）----
    # 参考国盛困境反转策略：价格跌到历史低位 + 量能萎缩 + 触底反弹信号
    distress_score = 0

    if not is_etf and current_idx >= 250:
        # 价格分位数：当前价格在过去250天中的位置 (0-1)
        lookback_250 = [all_rows[i]["close"] for i in range(current_idx - 250, current_idx)]
        price_pctile = sum(1 for p in lookback_250 if p < row["close"]) / 250

        if price_pctile <= 0.05:
            distress_score += 8   # 过去一年最低5% → 极度困境，强反转信号
        elif price_pctile <= 0.15:
            distress_score += 5   # 底部15%
        elif price_pctile <= 0.30:
            distress_score += 2   # 偏低
        elif price_pctile >= 0.90:
            distress_score -= 6   # 历史高位 → 大概率回落
        elif price_pctile >= 0.75:
            distress_score -= 3

        # 量能萎缩度：近5日均量 vs 过去120日均量
        if vol_ma_5 > 0 and current_idx >= 120:
            vol_120 = sum(all_rows[i]["volume"] for i in range(current_idx - 120, current_idx)) / 120
            if vol_120 > 0:
                vol_shrink = vol_ma_5 / vol_120
                if vol_shrink < 0.3:
                    distress_score += 5   # 量能极度萎缩 → 恐慌释放完毕
                elif vol_shrink < 0.5:
                    distress_score += 3
                elif vol_shrink > 2.5:
                    distress_score -= 4   # 天量 → 见顶信号
                elif vol_shrink > 1.8:
                    distress_score -= 2

        # 从底部回升确认（近5天从低点反弹3%以上 = 反转启动信号）
        if current_idx >= 10:
            recent_low = min(all_rows[i]["low"] for i in range(current_idx - 10, current_idx))
            if recent_low > 0:
                bounce_pct = (row["close"] - recent_low) / recent_low * 100
                if bounce_pct >= 5 and price_pctile <= 0.20:
                    distress_score += 4   # 低位反弹5%+ → 强确认

    # 组上限 ±15
    distress_score = max(-15, min(15, distress_score))
    score += distress_score

    # ---- 底部确认加分（多重信号共振）----
    bottom_signals = 0
    if rsi < 25:
        bottom_signals += 1
    if ma_diff_pct < -12:
        bottom_signals += 1
    if current_idx >= 20:
        lookback = min(current_idx, 120)
        min_close = min(all_rows[i]["close"] for i in range(current_idx - lookback, current_idx))
        near_low_pct = (row["close"] - min_close) / min_close if min_close > 0 else 1
        if near_low_pct < 0.10:
            bottom_signals += 1
    if bottom_signals >= 2:
        score += 10  # 多重底部确认

    # ---- 选股筛选（仅个股，非牛市） ----
    if not is_etf and regime != "牛市":
        if turnover_rate > 0 and turnover_rate < 0.03:
            score -= 20
        if total_shares >= 2_000_000_000:
            score -= 15

    # ---- 趋势过滤：个股价格在自身MA120下方且非超跌时扣分 ----
    stock_ma120 = sum(all_rows[i]["close"] for i in range(current_idx - 120, current_idx)) / 120 \
        if current_idx >= 120 else 0
    if not is_etf and current_idx >= 120 and stock_ma120 > 0:
        price_vs_ma120 = (row["close"] - stock_ma120) / stock_ma120 * 100
        if price_vs_ma120 < -30:
            pass  # 极度超跌，反转信号强，不扣分
        elif price_vs_ma120 < -15:
            score -= 5  # 深度下跌趋势，轻微扣分
        elif price_vs_ma120 < 0:
            score -= 8  # MA120下方，趋势向下，扣分

    return max(0, min(100, score)), regime


# ============================================================
# Portfolio 模拟器
# ============================================================

class Position:
    __slots__ = ("symbol", "shares", "cost", "buy_date", "buy_price")

    def __init__(self, symbol, shares, cost, buy_date, buy_price):
        self.symbol = symbol
        self.shares = shares
        self.cost = cost          # 总成本（含手续费）
        self.buy_date = buy_date
        self.buy_price = buy_price


class Portfolio:
    """组合模拟器，支持 T+1、佣金、印花税、滑点、止损"""

    def __init__(self, initial_capital, bt_cfg):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.positions = {}        # symbol -> Position
        self.trades = []           # 交易记录
        self.daily_equity = []     # (date, equity)
        self.daily_trades_count = defaultdict(int)  # date -> trade count

        # 费率
        self.commission_pct = bt_cfg.get("commission_pct", 0.00025)
        self.stamp_duty_pct = bt_cfg.get("stamp_duty_pct", 0.001)
        self.slippage_pct = bt_cfg.get("slippage_pct", 0.0003)

        # 风控
        self.max_position_pct = bt_cfg.get("max_position_pct", 0.10)
        self.max_total_pct = bt_cfg.get("max_total_pct", 0.80)
        self.stop_loss_pct = bt_cfg.get("stop_loss_pct", -0.05)
        self.portfolio_stop_loss_pct = bt_cfg.get("portfolio_stop_loss_pct", -0.30)
        self.max_daily_trades = bt_cfg.get("max_daily_trades", 5)
        self.max_positions = bt_cfg.get("max_positions", 10)

        # 状态
        self.halted = False  # 组合止损后暂停交易
        self.halt_date = None  # 组合止损触发日期
        self.cooldown_days = bt_cfg.get("cooldown_days", 60)  # 冷却期天数
        self.min_hold_days = bt_cfg.get("min_hold_days", 5)  # 持仓保护期

    def equity(self, prices):
        """计算总权益 = 现金 + 持仓市值"""
        total = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.buy_price)
            total += pos.shares * price
        return total

    def _can_trade(self, date):
        if self.halted:
            return False
        if self.daily_trades_count[date] >= self.max_daily_trades:
            return False
        return True

    def buy(self, symbol, price, date, prices):
        """买入，自动计算可买股数（按最大仓位比例）"""
        if not self._can_trade(date):
            return False
        if symbol in self.positions:
            return False  # 不加仓
        if len(self.positions) >= self.max_positions:
            return False

        eq = self.equity(prices)
        # 检查总仓位
        total_pos_value = sum(
            pos.shares * prices.get(s, pos.buy_price) for s, pos in self.positions.items()
        )
        if total_pos_value / eq >= self.max_total_pct:
            return False

        # 买入金额 = min(单票上限, 剩余可用仓位)
        max_amount = eq * self.max_position_pct
        remaining_room = eq * self.max_total_pct - total_pos_value
        amount = min(max_amount, remaining_room, self.cash * 0.99)  # 留 1% 应急
        if amount < 100:
            return False

        # 滑点（买入价抬高）
        exec_price = price * (1 + self.slippage_pct)
        if exec_price <= 0:
            return False
        # A 股最小买入 100 股
        shares = int(amount / exec_price / 100) * 100
        if shares < 100:
            return False

        cost = shares * exec_price
        commission = max(cost * self.commission_pct, 5)  # 最低 5 元
        total_cost = cost + commission

        if total_cost > self.cash:
            shares -= 100
            if shares < 100:
                return False
            cost = shares * exec_price
            commission = max(cost * self.commission_pct, 5)
            total_cost = cost + commission
            if total_cost > self.cash:
                return False

        self.cash -= total_cost
        self.positions[symbol] = Position(symbol, shares, total_cost, date, exec_price)
        self.daily_trades_count[date] += 1
        self.trades.append({
            "date": date, "symbol": symbol, "action": "BUY",
            "price": round(exec_price, 4), "shares": shares,
            "cost": round(total_cost, 2), "pnl": 0,
        })
        return True

    def sell(self, symbol, price, date, reason="signal"):
        """卖出全部持仓"""
        if symbol not in self.positions:
            return False
        if not self._can_trade(date):
            return False

        pos = self.positions[symbol]
        # T+1 检查
        if pos.buy_date == date:
            return False

        exec_price = price * (1 - self.slippage_pct)
        proceeds = pos.shares * exec_price
        commission = max(proceeds * self.commission_pct, 5)
        stamp_duty = proceeds * self.stamp_duty_pct
        net_proceeds = proceeds - commission - stamp_duty

        pnl = net_proceeds - pos.cost
        self.cash += net_proceeds
        del self.positions[symbol]
        self.daily_trades_count[date] += 1

        self.trades.append({
            "date": date, "symbol": symbol, "action": f"SELL({reason})",
            "price": round(exec_price, 4), "shares": pos.shares,
            "cost": round(net_proceeds, 2), "pnl": round(pnl, 2),
        })
        return True

    def check_cooldown(self, date):
        """检查冷却期是否结束，恢复交易"""
        if self.halted and self.halt_date:
            from datetime import datetime
            halt_dt = datetime.strptime(self.halt_date, "%Y-%m-%d")
            curr_dt = datetime.strptime(date, "%Y-%m-%d")
            if (curr_dt - halt_dt).days >= self.cooldown_days:
                self.halted = False
                self.halt_date = None

    def check_stop_loss(self, date, prices):
        """检查止损：单票止损 + 组合止损"""
        eq = self.equity(prices)
        # 组合止损（触发后进入冷却期，不永久停止）
        if not self.halted and \
           (eq - self.initial_capital) / self.initial_capital <= self.portfolio_stop_loss_pct:
            self.halted = True
            self.halt_date = date
            # 清仓
            for sym in list(self.positions.keys()):
                price = prices.get(sym, self.positions[sym].buy_price)
                self.sell(sym, price, date, reason="portfolio_stop")
            return

        # 单票止损（含持仓保护期）
        min_hold = self.min_hold_days
        for sym in list(self.positions.keys()):
            pos = self.positions[sym]
            if pos.buy_date == date:
                continue  # T+1
            # 持仓保护期内不止损（除非跌幅超过保护期止损线）
            from datetime import datetime
            hold_days = (datetime.strptime(date, "%Y-%m-%d") -
                         datetime.strptime(pos.buy_date, "%Y-%m-%d")).days
            price = prices.get(sym, pos.buy_price)
            pnl_pct = (price - pos.buy_price) / pos.buy_price
            if hold_days < min_hold:
                # 保护期内只有跌幅超过30%才强制止损（极端情况）
                if pnl_pct <= -0.30:
                    self.sell(sym, price, date, reason="stop_loss")
            elif pnl_pct <= self.stop_loss_pct:
                self.sell(sym, price, date, reason="stop_loss")

    def snapshot(self, date, prices):
        """记录当日权益"""
        eq = self.equity(prices)
        self.daily_equity.append((date, eq))
        return eq


# ============================================================
# 绩效指标
# ============================================================

def compute_metrics(equity_curve, risk_free=RISK_FREE_RATE):
    """
    从 [(date, equity), ...] 计算绩效指标。
    返回 dict
    """
    if len(equity_curve) < 2:
        return {"error": "数据不足"}

    dates = [e[0] for e in equity_curve]
    values = np.array([e[1] for e in equity_curve], dtype=float)

    total_return = (values[-1] / values[0]) - 1
    n_days = (datetime.strptime(dates[-1], "%Y-%m-%d") -
              datetime.strptime(dates[0], "%Y-%m-%d")).days
    if n_days <= 0:
        n_days = 1
    ann_factor = 365 / n_days
    ann_return = (1 + total_return) ** ann_factor - 1

    # 日收益率
    daily_returns = np.diff(values) / values[:-1]
    daily_rf = (1 + risk_free) ** (1 / 252) - 1
    excess_daily = daily_returns - daily_rf

    std_daily = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-8
    sharpe = np.mean(excess_daily) / std_daily * np.sqrt(252) if std_daily > 1e-8 else 0

    # 最大回撤
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # 最大回撤持续天数
    max_dd_duration = 0
    current_dd_start = None
    for i in range(len(values)):
        if values[i] < running_max[i]:
            if current_dd_start is None:
                current_dd_start = i
        else:
            if current_dd_start is not None:
                dur = (datetime.strptime(dates[i], "%Y-%m-%d") -
                       datetime.strptime(dates[current_dd_start], "%Y-%m-%d")).days
                max_dd_duration = max(max_dd_duration, dur)
                current_dd_start = None
    if current_dd_start is not None:
        dur = (datetime.strptime(dates[-1], "%Y-%m-%d") -
               datetime.strptime(dates[current_dd_start], "%Y-%m-%d")).days
        max_dd_duration = max(max_dd_duration, dur)

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration_days": max_dd_duration,
        "calmar_ratio": round(calmar, 3),
    }


def compute_trade_metrics(trades):
    """从交易记录计算胜率、盈亏比等"""
    sells = [t for t in trades if t["action"].startswith("SELL")]
    if not sells:
        return {
            "total_trades": len(trades),
            "win_rate_pct": 0,
            "profit_factor": 0,
        }

    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "total_trades": len(trades),
        "sell_trades": len(sells),
        "win_rate_pct": round(len(wins) / len(sells) * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(-gross_loss, 2),
    }


def compute_vs_benchmark(strategy_equity, benchmark_equity, risk_free=RISK_FREE_RATE):
    """计算超额收益和信息比率"""
    s_ret = (strategy_equity[-1][1] / strategy_equity[0][1]) - 1
    b_ret = (benchmark_equity[-1][1] / benchmark_equity[0][1]) - 1
    excess = s_ret - b_ret

    # 信息比率：超额收益的日均值 / 超额收益的标准差 * sqrt(252)
    s_dates = {e[0]: e[1] for e in strategy_equity}
    b_dates = {e[0]: e[1] for e in benchmark_equity}
    common = sorted(set(s_dates) & set(b_dates))

    if len(common) < 2:
        return {"excess_return_pct": round(excess * 100, 2), "information_ratio": 0}

    s_vals = np.array([s_dates[d] for d in common], dtype=float)
    b_vals = np.array([b_dates[d] for d in common], dtype=float)
    s_daily = np.diff(s_vals) / s_vals[:-1]
    b_daily = np.diff(b_vals) / b_vals[:-1]
    excess_daily = s_daily - b_daily
    te = np.std(excess_daily, ddof=1)
    ir = np.mean(excess_daily) / te * np.sqrt(252) if te > 1e-8 else 0

    return {
        "excess_return_pct": round(excess * 100, 2),
        "information_ratio": round(ir, 3),
    }


# ============================================================
# 策略1: 个股选股 (每日)
# ============================================================

def run_stock_strategy(symbol_data, date_index, bt_cfg, regime_source=None):
    """每日选股策略。regime_source: 基准数据用于判定市场环境（如沪深300 ETF）"""
    print("\n[策略1] 个股选股 (每日)")
    initial = bt_cfg["initial_capital"]
    risk_cfg = cfg.get("risk_control", {})
    thresholds = cfg.get("score_thresholds", {})

    portfolio_cfg = {
        "commission_pct": risk_cfg.get("commission_pct", 0.00025),
        "stamp_duty_pct": 0.001,
        "slippage_pct": risk_cfg.get("slippage_pct", 0.0003),
        "max_position_pct": risk_cfg.get("max_position_pct", 0.10),
        "max_total_pct": risk_cfg.get("max_total_position_pct", 0.80),
        "stop_loss_pct": risk_cfg.get("stop_loss_pct", -0.20),  # 反转策略需要更大容忍度
        "portfolio_stop_loss_pct": risk_cfg.get("portfolio_stop_loss_pct", -0.30),
        "cooldown_days": 60,  # 组合止损后冷却60天再恢复
        "min_hold_days": 5,   # 持仓保护期
        "max_daily_trades": risk_cfg.get("max_daily_trades", 5),
        "max_positions": bt_cfg.get("max_positions", 10),
    }

    pf = Portfolio(initial, portfolio_cfg)
    sorted_dates = sorted(date_index.keys())
    total_dates = len(sorted_dates)

    # 预构建 symbol -> row 索引映射
    sym_idx_cache = {}  # symbol -> {date: index_in_all_rows}
    for sym, rows in symbol_data.items():
        idx_map = {}
        for i, r in enumerate(rows):
            idx_map[r["date"]] = i
        sym_idx_cache[sym] = idx_map

    # 预构建 regime_source 的日期索引（用沪深300判定市场环境）
    regime_idx_cache = {}
    if regime_source:
        for i, r in enumerate(regime_source):
            regime_idx_cache[r["date"]] = i

    for di, date in enumerate(sorted_dates):
        if di % 50 == 0:
            print(f"  进度: {di}/{total_dates} ({date})")

        available = date_index[date]
        prices = {sym: r["close"] for sym, r in available.items()}

        # 冷却期检查：到期后恢复交易
        if pf.halted:
            pf.check_cooldown(date)
        if pf.halted:
            pf.snapshot(date, prices)
            continue

        # 止损检查
        pf.check_stop_loss(date, prices)
        if pf.halted:
            pf.snapshot(date, prices)
            continue

        # 获取当前 regime（优先用沪深300基准数据判定）
        current_regime = "震荡"
        regime_info = None  # 预计算的 regime 信息，传给 compute_score 避免重复计算
        if regime_source and date in regime_idx_cache:
            idx = regime_idx_cache[date]
            if idx >= 120:
                regime_info = detect_regime(regime_source, idx)
                current_regime = regime_info[0]
        else:
            # fallback: 用第一个有足够数据的标的
            for sym in list(available.keys())[:5]:
                if sym in sym_idx_cache and date in sym_idx_cache[sym]:
                    idx = sym_idx_cache[sym][date]
                    rows = symbol_data[sym]
                    if idx >= 120:
                        regime_info = detect_regime(rows, idx)
                        current_regime = regime_info[0]
                        break

        # 确定阈值 + 熊市仓位控制
        if current_regime == "牛市":
            buy_th = thresholds.get("bull_buy", 68)
            sell_th = thresholds.get("bull_sell", 35)
            max_positions_now = pf.max_positions  # 满仓
        elif current_regime == "熊市":
            buy_th = thresholds.get("bear_buy", 88)
            sell_th = thresholds.get("bear_sell", 30)
            max_positions_now = max(2, pf.max_positions // 4)  # 熊市最多25%仓位
        else:
            buy_th = thresholds.get("buy", 78)
            sell_th = thresholds.get("sell", 28)
            max_positions_now = max(3, pf.max_positions * 2 // 3)  # 震荡市66%仓位

        # ---- 卖出检查 ----
        for sym in list(pf.positions.keys()):
            if sym not in available:
                continue
            if sym not in sym_idx_cache or date not in sym_idx_cache[sym]:
                continue
            rows = symbol_data[sym]
            idx = sym_idx_cache[sym][date]
            prev = rows[idx - 1] if idx > 0 else None
            score, _ = compute_score(available[sym], rows, idx, prev, is_etf=False,
                                     regime_info=regime_info)
            if score <= sell_th:
                pf.sell(sym, available[sym]["close"], date, reason="signal")

        # ---- 熊市主动减仓：持仓超过仓位上限时卖出最低分的 ----
        if len(pf.positions) > max_positions_now:
            # 对所有持仓重新评分，卖出最低分的
            pos_scores = []
            for sym in list(pf.positions.keys()):
                if sym in available and sym in sym_idx_cache and date in sym_idx_cache[sym]:
                    rows = symbol_data[sym]
                    idx = sym_idx_cache[sym][date]
                    prev = rows[idx - 1] if idx > 0 else None
                    s, _ = compute_score(available[sym], rows, idx, prev,
                                         is_etf=False, regime_info=regime_info)
                    pos_scores.append((s, sym))
            pos_scores.sort()  # 升序，最差的在前
            excess = len(pf.positions) - max_positions_now
            for s, sym in pos_scores[:excess]:
                if sym in available:
                    pf.sell(sym, available[sym]["close"], date, reason="regime_reduce")

        # ---- 买入：评分所有可用标的，取 top N ----
        if len(pf.positions) < max_positions_now:
            candidates = []
            for sym, r in available.items():
                if sym in pf.positions:
                    continue
                if sym not in sym_idx_cache or date not in sym_idx_cache[sym]:
                    continue
                rows = symbol_data[sym]
                idx = sym_idx_cache[sym][date]
                if idx < 20:
                    continue  # 数据太少
                prev = rows[idx - 1] if idx > 0 else None
                score, _ = compute_score(r, rows, idx, prev, is_etf=False,
                                         regime_info=regime_info)
                if score >= buy_th:
                    candidates.append((score, sym))

            # 按评分降序，取 top N
            candidates.sort(reverse=True)
            slots = max_positions_now - len(pf.positions)
            for score, sym in candidates[:slots]:
                if not pf._can_trade(date):
                    break
                pf.buy(sym, available[sym]["close"], date, prices)

        pf.snapshot(date, prices)

    print(f"  完成，共 {total_dates} 个交易日")
    return pf


# ============================================================
# 策略2: 行业ETF轮动 (每周)
# ============================================================

def run_etf_strategy(etf_data, etf_date_index, bt_cfg, regime_source=None):
    """行业ETF轮动策略。regime_source: 基准数据用于判定市场环境"""
    print("\n[策略2] 行业ETF轮动 (每周)")
    initial = bt_cfg["initial_capital"]
    risk_cfg = cfg.get("risk_control", {})

    portfolio_cfg = {
        "commission_pct": risk_cfg.get("commission_pct", 0.00025),
        "stamp_duty_pct": 0.001,
        "slippage_pct": risk_cfg.get("slippage_pct", 0.0003),
        "max_position_pct": 0.30,          # ETF 轮动，仓位放宽
        "max_total_pct": 0.95,
        "stop_loss_pct": -0.08,
        "portfolio_stop_loss_pct": -0.30,
        "max_daily_trades": 10,
        "max_positions": 4,
    }

    pf = Portfolio(initial, portfolio_cfg)
    sorted_dates = sorted(etf_date_index.keys())
    total_dates = len(sorted_dates)

    # 构建索引
    etf_idx_cache = {}
    for code, rows in etf_data.items():
        idx_map = {}
        for i, r in enumerate(rows):
            idx_map[r["date"]] = i
        etf_idx_cache[code] = idx_map

    last_rebalance = None
    etf_codes = list(SECTOR_ETFS.values())

    for di, date in enumerate(sorted_dates):
        if di % 50 == 0:
            print(f"  进度: {di}/{total_dates} ({date})")

        available = etf_date_index[date]
        prices = {code: r["close"] for code, r in available.items()}

        if pf.halted:
            pf.snapshot(date, prices)
            continue

        pf.check_stop_loss(date, prices)
        if pf.halted:
            pf.snapshot(date, prices)
            continue

        # 判断是否需要再平衡（每周一，或距上次超过 7 天）
        dt = datetime.strptime(date, "%Y-%m-%d")
        need_rebalance = False
        if last_rebalance is None:
            need_rebalance = True
        else:
            days_since = (dt - datetime.strptime(last_rebalance, "%Y-%m-%d")).days
            if days_since >= 5 and dt.weekday() == 0:  # 周一
                need_rebalance = True
            elif days_since >= 7:
                need_rebalance = True

        if need_rebalance:
            # 评分所有行业 ETF
            scored = []
            for code in etf_codes:
                if code not in available:
                    continue
                if code not in etf_idx_cache or date not in etf_idx_cache[code]:
                    continue
                rows = etf_data[code]
                idx = etf_idx_cache[code][date]
                if idx < 20:
                    continue
                prev = rows[idx - 1] if idx > 0 else None
                score, _ = compute_score(available[code], rows, idx, prev, is_etf=True)
                scored.append((score, code))

            scored.sort(reverse=True)
            # 选 top 3-4（评分 >= 50 才入选）
            target_codes = set()
            for score, code in scored[:4]:
                if score >= 50:
                    target_codes.add(code)

            # 卖出不在目标中的持仓
            for sym in list(pf.positions.keys()):
                if sym not in target_codes:
                    price = prices.get(sym, pf.positions[sym].buy_price)
                    pf.sell(sym, price, date, reason="rotation")

            # 买入目标中未持有的
            for code in target_codes:
                if code not in pf.positions and code in available:
                    pf.buy(code, available[code]["close"], date, prices)

            last_rebalance = date

        pf.snapshot(date, prices)

    print(f"  完成，共 {total_dates} 个交易日")
    return pf


# ============================================================
# 基准
# ============================================================

def compute_benchmark_equity(benchmark_data, date_range, initial_capital):
    """计算基准的每日净值曲线（全仓持有）"""
    start, end = date_range
    rows_in_range = [r for r in benchmark_data if start <= r["date"] <= end]
    if not rows_in_range:
        return []

    base_price = rows_in_range[0]["close"]
    return [(r["date"], initial_capital * r["close"] / base_price) for r in rows_in_range]


# ============================================================
# 输出
# ============================================================

def save_results(results, output_dir):
    """保存回测结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 汇总 JSON
    json_path = os.path.join(output_dir, "backtest_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果 -> {json_path}")

    # 2. 权益曲线 CSV
    eq_path = os.path.join(output_dir, "backtest_equity.csv")
    # 合并三条曲线到同一日期轴
    all_dates = set()
    for curve in [results["stock_equity"], results["etf_equity"], results["benchmark_equity"]]:
        for d, _ in curve:
            all_dates.add(d)
    all_dates = sorted(all_dates)

    stock_map = dict(results["stock_equity"])
    etf_map = dict(results["etf_equity"])
    bm_map = dict(results["benchmark_equity"])

    with open(eq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "strategy1_equity", "strategy2_equity", "benchmark_equity"])
        for d in all_dates:
            w.writerow([
                d,
                round(stock_map.get(d, 0), 2),
                round(etf_map.get(d, 0), 2),
                round(bm_map.get(d, 0), 2),
            ])
    print(f"权益曲线 -> {eq_path}")

    # 3. 交易记录 CSV
    trades_path = os.path.join(output_dir, "backtest_trades.csv")
    all_trades = results.get("all_trades", [])
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "date", "symbol", "action", "price", "shares", "cost", "pnl"])
        w.writeheader()
        for t in all_trades:
            w.writerow(t)
    print(f"交易记录 -> {trades_path}")


def print_comparison(results):
    """打印对比表"""
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"{'指标':<30} {'个股策略':>12} {'ETF轮动':>12} {'沪深300':>12}")
    print(sep)

    metrics_keys = [
        ("total_return_pct", "总收益率(%)"),
        ("annualized_return_pct", "年化收益率(%)"),
        ("sharpe_ratio", "夏普比率"),
        ("max_drawdown_pct", "最大回撤(%)"),
        ("max_drawdown_duration_days", "最大回撤天数"),
        ("calmar_ratio", "卡尔马比率"),
    ]

    s1 = results["summary"]["strategy1"]
    s2 = results["summary"]["strategy2"]
    bm = results["summary"]["benchmark"]

    for key, label in metrics_keys:
        v1 = s1.get("metrics", {}).get(key, "N/A")
        v2 = s2.get("metrics", {}).get(key, "N/A")
        vb = bm.get("metrics", {}).get(key, "N/A")
        print(f"{label:<30} {str(v1):>12} {str(v2):>12} {str(vb):>12}")

    # 交易统计
    trade_keys = [
        ("total_trades", "总交易笔数"),
        ("sell_trades", "卖出笔数"),
        ("win_rate_pct", "胜率(%)"),
        ("profit_factor", "盈亏比"),
    ]
    print(sep)
    for key, label in trade_keys:
        v1 = s1.get("trade_metrics", {}).get(key, "N/A")
        v2 = s2.get("trade_metrics", {}).get(key, "N/A")
        print(f"{label:<30} {str(v1):>12} {str(v2):>12} {'':>12}")

    # 超额收益
    print(sep)
    for key, label in [
        ("excess_return_pct", "超额收益(%)"),
        ("information_ratio", "信息比率"),
    ]:
        v1 = s1.get("vs_benchmark", {}).get(key, "N/A")
        v2 = s2.get("vs_benchmark", {}).get(key, "N/A")
        print(f"{label:<30} {str(v1):>12} {str(v2):>12} {'':>12}")

    print(sep)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="QuantLLM 回测系统")
    parser.add_argument("--start", type=str, help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, help="初始资金")
    parser.add_argument("--skip-stock", action="store_true", help="跳过个股策略（加速调试）")
    args = parser.parse_args()

    # 合并配置
    bt_section = cfg.get("backtest", {})
    bt_cfg = {**BT_DEFAULTS, **bt_section}
    if args.start:
        bt_cfg["start_date"] = args.start
    if args.end:
        bt_cfg["end_date"] = args.end
    if args.capital:
        bt_cfg["initial_capital"] = args.capital

    date_range = (bt_cfg["start_date"], bt_cfg["end_date"])
    initial_capital = bt_cfg["initial_capital"]
    benchmark_code = bt_cfg["benchmark"]

    print("=" * 60)
    print("QuantLLM 回测系统")
    print("=" * 60)
    print(f"回测区间: {date_range[0]} ~ {date_range[1]}")
    print(f"初始资金: {initial_capital:,.0f}")
    print(f"基准: CSI300 ETF ({benchmark_code})")

    # ---- 数据目录 ----
    ashare_dir = os.path.join(PROJECT_ROOT, cfg["data"]["ashare_dir"], "advanced")
    etf_dir = os.path.join(PROJECT_ROOT, cfg["data"]["etf_dir"], "advanced")

    # ---- 加载基准 ----
    print("\n加载基准数据...")
    benchmark_data = load_etf(etf_dir, benchmark_code)
    if not benchmark_data:
        print(f"错误: 找不到基准 ETF {benchmark_code} 的数据", file=sys.stderr)
        sys.exit(1)
    benchmark_equity = compute_benchmark_equity(benchmark_data, date_range, initial_capital)
    print(f"  基准数据: {len(benchmark_data)} 条, 区间内 {len(benchmark_equity)} 个交易日")

    # ---- 策略1: 个股 ----
    stock_pf = None
    if not args.skip_stock:
        print("\n加载个股数据...")
        symbol_data, stock_date_index = load_all_stocks(ashare_dir, date_range)
        print(f"  个股数: {len(symbol_data)}, 交易日: {len(stock_date_index)}")
        stock_pf = run_stock_strategy(symbol_data, stock_date_index, bt_cfg,
                                       regime_source=benchmark_data)
    else:
        print("\n[跳过] 个股策略")

    # ---- 策略2: ETF 轮动 ----
    print("\n加载ETF数据...")
    all_etf_codes = list(SECTOR_ETFS.values()) + [benchmark_code]
    etf_data, etf_date_index = load_all_etfs(etf_dir, all_etf_codes, date_range)
    print(f"  ETF数: {len(etf_data)}, 交易日: {len(etf_date_index)}")
    etf_pf = run_etf_strategy(etf_data, etf_date_index, bt_cfg,
                              regime_source=benchmark_data)

    # ---- 计算指标 ----
    print("\n计算绩效指标...")

    bm_metrics = compute_metrics(benchmark_equity)

    results = {
        "summary": {
            "config": {
                "start_date": bt_cfg["start_date"],
                "end_date": bt_cfg["end_date"],
                "initial_capital": initial_capital,
                "benchmark": benchmark_code,
            },
            "benchmark": {"metrics": bm_metrics},
        },
        "benchmark_equity": benchmark_equity,
        "all_trades": [],
    }

    # 策略1 指标
    if stock_pf:
        s1_metrics = compute_metrics(stock_pf.daily_equity)
        s1_trade = compute_trade_metrics(stock_pf.trades)
        s1_vs_bm = compute_vs_benchmark(stock_pf.daily_equity, benchmark_equity)
        results["summary"]["strategy1"] = {
            "name": "个股选股(每日)",
            "metrics": s1_metrics,
            "trade_metrics": s1_trade,
            "vs_benchmark": s1_vs_bm,
        }
        results["stock_equity"] = stock_pf.daily_equity
        for t in stock_pf.trades:
            results["all_trades"].append({"strategy": "stock", **t})
    else:
        results["summary"]["strategy1"] = {
            "name": "个股选股(每日)",
            "metrics": {"total_return_pct": 0},
            "trade_metrics": {},
            "vs_benchmark": {},
        }
        results["stock_equity"] = []

    # 策略2 指标
    s2_metrics = compute_metrics(etf_pf.daily_equity)
    s2_trade = compute_trade_metrics(etf_pf.trades)
    s2_vs_bm = compute_vs_benchmark(etf_pf.daily_equity, benchmark_equity)
    results["summary"]["strategy2"] = {
        "name": "行业ETF轮动(每周)",
        "metrics": s2_metrics,
        "trade_metrics": s2_trade,
        "vs_benchmark": s2_vs_bm,
    }
    results["etf_equity"] = etf_pf.daily_equity
    for t in etf_pf.trades:
        results["all_trades"].append({"strategy": "etf", **t})

    # ---- 输出 ----
    output_dir = os.path.join(PROJECT_ROOT, "output")
    save_results(results, output_dir)
    print_comparison(results)

    print(f"\n回测完成。")


if __name__ == "__main__":
    main()
