#!/usr/bin/env python3
"""构建个股因子文件（PE/PB/ROE/北向资金）供实盘评分使用。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import akshare as ak
import pandas as pd

from _config import PROJECT_ROOT


def _norm_symbol(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if len(s) == 6 and s.isdigit():
        return ("sh" if s[0] in {"6", "9"} else "sz") + s
    if "." in s:
        code, mk = s.split(".", 1)
        if len(code) == 6 and code.isdigit():
            return (mk.lower() + code) if mk.lower() in {"sh", "sz", "bj"} else s
    if s[:2] in {"sh", "sz", "bj"} and len(s) == 8 and s[2:].isdigit():
        return s
    return s


def _load_universe(advanced_dir: Path, limit: int) -> List[str]:
    syms = []
    for fp in sorted(advanced_dir.glob("*.jsonl")):
        syms.append(_norm_symbol(fp.stem))
        if len(syms) >= limit:
            break
    return syms


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = [str(c) for c in df.columns]
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for x in cols:
            if c in x:
                return x
    return ""


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(str(v).replace(",", ""))
    except Exception:
        return default


def _fetch_spot_factor_map() -> Dict[str, dict]:
    """一次性获取 PE/PB。"""
    out: Dict[str, dict] = {}
    try:
        df = ak.stock_zh_a_spot_em()
    except Exception:
        return out
    if df is None or df.empty:
        return out
    code_col = _pick_col(df, ["代码"])
    pe_col = _pick_col(df, ["市盈率-动态", "市盈率", "PE"])
    pb_col = _pick_col(df, ["市净率", "PB"])
    if not code_col:
        return out
    for _, r in df.iterrows():
        code = str(r.get(code_col, "")).strip()
        sym = _norm_symbol(code)
        if not sym:
            continue
        out[sym] = {
            "pe_ttm": _safe_float(r.get(pe_col, 0), 0.0) if pe_col else 0.0,
            "pb": _safe_float(r.get(pb_col, 0), 0.0) if pb_col else 0.0,
        }
    return out


def _fetch_roe(symbol6: str) -> float:
    """按个股获取 ROE（较慢，建议限量）。"""
    try:
        df = ak.stock_financial_analysis_indicator(symbol=symbol6)
    except Exception:
        return 0.0
    if df is None or df.empty:
        return 0.0
    roe_col = _pick_col(df, ["净资产收益率", "ROE"])
    if not roe_col:
        return 0.0
    for i in range(len(df) - 1, -1, -1):
        v = _safe_float(df.iloc[i].get(roe_col, 0), 0.0)
        if v != 0:
            return v
    return 0.0


def _fetch_northbound_5d_map() -> Dict[str, float]:
    """获取北向5日净流入（可用时）。"""
    out: Dict[str, float] = {}
    candidates = [
        {"market": "北向", "indicator": "5日排行"},
        {"market": "北向", "indicator": "今日排行"},
    ]
    df = None
    for kw in candidates:
        try:
            df = ak.stock_hsgt_hold_stock_em(**kw)
            if df is not None and not df.empty:
                break
        except Exception:
            continue
    if df is None or df.empty:
        return out
    code_col = _pick_col(df, ["代码", "股票代码"])
    val_col = _pick_col(df, ["5日增持估计市值", "5日增持市值", "增持估计市值", "今日增持估计市值", "今日增持市值"])
    if not code_col or not val_col:
        return out
    for _, r in df.iterrows():
        sym = _norm_symbol(str(r.get(code_col, "")))
        if not sym:
            continue
        out[sym] = _safe_float(r.get(val_col, 0), 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stock factor file for live ranking")
    parser.add_argument("--max-symbols", type=int, default=500, help="最大处理股票数")
    parser.add_argument("--roe-limit", type=int, default=120, help="按财报接口计算 ROE 的股票上限")
    parser.add_argument("--out", default="training-data/factors/stock_factors_latest.json", help="输出路径（相对项目根）")
    args = parser.parse_args()

    adv_dir = Path(PROJECT_ROOT) / "training-data" / "ashare" / "advanced"
    out_fp = Path(PROJECT_ROOT) / args.out
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    universe = _load_universe(adv_dir, args.max_symbols)
    spot_map = _fetch_spot_factor_map()
    north_map = _fetch_northbound_5d_map()

    rows: Dict[str, dict] = {}
    for sym in universe:
        rows[sym] = {
            "symbol": sym,
            "pe_ttm": 0.0,
            "pb": 0.0,
            "roe": 0.0,
            "northbound_net_5d": 0.0,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if sym in spot_map:
            rows[sym]["pe_ttm"] = round(_safe_float(spot_map[sym].get("pe_ttm", 0), 0.0), 6)
            rows[sym]["pb"] = round(_safe_float(spot_map[sym].get("pb", 0), 0.0), 6)
        if sym in north_map:
            rows[sym]["northbound_net_5d"] = round(_safe_float(north_map[sym], 0.0), 2)

    for i, sym in enumerate(universe[: max(0, args.roe_limit)]):
        code6 = sym[-6:] if len(sym) >= 6 else ""
        if len(code6) == 6 and code6.isdigit():
            rows[sym]["roe"] = round(_fetch_roe(code6), 6)

    payload = {
        "_meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "universe_size": len(universe),
            "roe_limit": args.roe_limit,
        },
        "factors": rows,
    }
    out_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pe_hit = sum(1 for x in rows.values() if float(x.get("pe_ttm", 0) or 0) != 0)
    pb_hit = sum(1 for x in rows.values() if float(x.get("pb", 0) or 0) != 0)
    roe_hit = sum(1 for x in rows.values() if float(x.get("roe", 0) or 0) != 0)
    nb_hit = sum(1 for x in rows.values() if float(x.get("northbound_net_5d", 0) or 0) != 0)
    print(f"[FACTORS] out={out_fp}")
    print(f"[FACTORS] universe={len(universe)} pe_hit={pe_hit} pb_hit={pb_hit} roe_hit={roe_hit} north_hit={nb_hit}")


if __name__ == "__main__":
    main()
