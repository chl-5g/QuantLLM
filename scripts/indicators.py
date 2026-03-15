#!/usr/bin/env python3
"""
技术指标共享库 — 纯 pandas/numpy 实现，不依赖 ta-lib。
crawl_ashare.py / crawl_multi_market.py / generate_predictive_data.py 共用。
"""

import pandas as pd
import numpy as np


# ============================================================
# 基础指标（原有，从 crawl_ashare.py 迁移）
# ============================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI — Relative Strength Index"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26,
              signal: int = 9):
    """MACD — 返回 (macd_line, signal_line, histogram)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_ma(series: pd.Series, period: int) -> pd.Series:
    """SMA — 简单移动平均"""
    return series.rolling(window=period, min_periods=period).mean()


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA — 指数移动平均"""
    return series.ewm(span=span, adjust=False).mean()


# ============================================================
# 动量指标
# ============================================================

def calc_roc(series: pd.Series, period: int = 12) -> pd.Series:
    """ROC — Rate of Change (%)"""
    shifted = series.shift(period)
    return ((series - shifted) / shifted.replace(0, np.nan)) * 100


def calc_williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 14) -> pd.Series:
    """Williams %R — 范围 [-100, 0]"""
    highest = high.rolling(window=period, min_periods=period).max()
    lowest = low.rolling(window=period, min_periods=period).min()
    denom = highest - lowest
    return ((highest - close) / denom.replace(0, np.nan)) * -100


def calc_cci(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 20) -> pd.Series:
    """CCI — Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))


# ============================================================
# 波动率指标
# ============================================================

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """ATR — Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def calc_bollinger_bands(close: pd.Series, period: int = 20,
                         num_std: float = 2.0):
    """Bollinger Bands — 返回 (upper, middle, lower, width, position)"""
    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0, np.nan)
    # position: 0=lower band, 1=upper band
    band_range = (upper - lower).replace(0, np.nan)
    position = (close - lower) / band_range
    return upper, middle, lower, width, position


def calc_historical_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """HV — 历史波动率（年化 %）"""
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=period, min_periods=period).std() * np.sqrt(252) * 100


# ============================================================
# 量能指标
# ============================================================

def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV — On Balance Volume"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def calc_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series, period: int = 14) -> pd.Series:
    """MFI — Money Flow Index (量价 RSI)"""
    tp = (high + low + close) / 3
    mf = tp * volume
    tp_diff = tp.diff()
    pos_mf = mf.where(tp_diff > 0, 0.0)
    neg_mf = mf.where(tp_diff < 0, 0.0)
    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi


def calc_vwap_proxy(high: pd.Series, low: pd.Series, close: pd.Series,
                    volume: pd.Series, period: int = 20) -> pd.Series:
    """VWAP 近似 — 滚动周期内的成交量加权均价"""
    tp = (high + low + close) / 3
    tp_vol = tp * volume
    return (tp_vol.rolling(window=period, min_periods=period).sum()
            / volume.rolling(window=period, min_periods=period).sum().replace(0, np.nan))


def calc_volume_change_rate(volume: pd.Series, period: int = 5) -> pd.Series:
    """量变化率 — 当前成交量 / N日均量"""
    vol_ma = volume.rolling(window=period, min_periods=period).mean()
    return volume / vol_ma.replace(0, np.nan)


# ============================================================
# 趋势指标
# ============================================================

def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """ADX — Average Directional Index"""
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calc_atr(high, low, close, period)
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / atr_safe

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()
    return adx


# ============================================================
# 综合入口
# ============================================================

def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    给 DataFrame 一次性添加全部技术指标。
    要求 df 包含列：open, high, low, close, volume（数值型，按日期升序排列）。
    """
    df = df.copy()

    # --- 原有指标 ---
    df["rsi_14"] = calc_rsi(df["close"], 14)
    macd_line, signal_line, histogram = calc_macd(df["close"])
    df["macd_line"] = macd_line
    df["signal_line"] = signal_line
    df["macd_histogram"] = histogram
    df["volume_ma_5"] = calc_ma(df["volume"], 5)
    df["close_ma_20"] = calc_ma(df["close"], 20)

    # --- 多周期均线 ---
    df["close_ma_5"] = calc_ma(df["close"], 5)
    df["close_ma_10"] = calc_ma(df["close"], 10)
    df["close_ma_60"] = calc_ma(df["close"], 60)
    df["close_ma_120"] = calc_ma(df["close"], 120)
    df["ema_12"] = calc_ema(df["close"], 12)
    df["ema_26"] = calc_ema(df["close"], 26)

    # --- 动量 ---
    df["roc_12"] = calc_roc(df["close"], 12)
    df["williams_r_14"] = calc_williams_r(df["high"], df["low"], df["close"], 14)
    df["cci_20"] = calc_cci(df["high"], df["low"], df["close"], 20)

    # --- 波动率 ---
    df["atr_14"] = calc_atr(df["high"], df["low"], df["close"], 14)
    bb_upper, bb_mid, bb_lower, bb_width, bb_pos = calc_bollinger_bands(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_width"] = bb_width
    df["bb_position"] = bb_pos
    df["hv_20"] = calc_historical_volatility(df["close"], 20)

    # --- 量能 ---
    df["obv"] = calc_obv(df["close"], df["volume"])
    df["mfi_14"] = calc_mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["vwap_proxy"] = calc_vwap_proxy(df["high"], df["low"], df["close"], df["volume"])
    df["vol_change_rate"] = calc_volume_change_rate(df["volume"], 5)

    # --- 趋势 ---
    df["adx_14"] = calc_adx(df["high"], df["low"], df["close"], 14)

    # --- 派生特征 ---
    # MA 排列：MA5 > MA10 > MA20 > MA60 = bullish，反之 bearish
    ma_cols = ["close_ma_5", "close_ma_10", "close_ma_20", "close_ma_60"]
    df["ma_alignment"] = _calc_ma_alignment(df, ma_cols)

    # OBV 趋势（10日线性回归斜率方向）
    df["obv_trend"] = _calc_obv_trend(df["obv"], 10)

    return df


def _calc_ma_alignment(df: pd.DataFrame, ma_cols: list) -> pd.Series:
    """判断均线排列：bullish / bearish / mixed"""
    result = pd.Series("mixed", index=df.index)
    valid = df[ma_cols].notna().all(axis=1)

    bullish = valid.copy()
    bearish = valid.copy()
    for i in range(len(ma_cols) - 1):
        bullish &= df[ma_cols[i]] > df[ma_cols[i + 1]]
        bearish &= df[ma_cols[i]] < df[ma_cols[i + 1]]

    result[bullish] = "bullish"
    result[bearish] = "bearish"
    return result


def _calc_obv_trend(obv: pd.Series, period: int = 10) -> pd.Series:
    """OBV 趋势方向：rising / falling / flat"""
    obv_ma_short = obv.rolling(window=period, min_periods=period).mean()
    obv_ma_long = obv.rolling(window=period * 2, min_periods=period * 2).mean()

    result = pd.Series("flat", index=obv.index)
    valid = obv_ma_short.notna() & obv_ma_long.notna()
    diff_pct = ((obv_ma_short - obv_ma_long) / obv_ma_long.abs().replace(0, np.nan))

    result[valid & (diff_pct > 0.02)] = "rising"
    result[valid & (diff_pct < -0.02)] = "falling"
    return result
