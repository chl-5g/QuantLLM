#!/usr/bin/env python3
"""两层架构实盘入口：规则初筛 -> Qwen14B精筛 -> 计划/执行。"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from _config import PROJECT_ROOT, cfg
from backtest_signals import compute_score, detect_regime
from qwen_skills import call_skill_stock_rank
from trade_execution import execute_plan, sanitize_plan


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _regime_score_to_100(regime_info: tuple) -> float:
    """
    detect_regime 的原始 score 约在 [-5, +5] 附近，映射到 [0, 100] 连续分值。
    """
    detail = regime_info[3] if len(regime_info) > 3 and isinstance(regime_info[3], dict) else {}
    raw = float(detail.get("regime_score", 0) or 0)
    return round(_clamp((raw + 5.0) * 10.0, 0.0, 100.0), 2)


def _target_exposure_by_regime_score(score100: float) -> float:
    """
    P1 连续化仓位：
    0-30: <=20%
    30-50: 30-50%
    50-70: 50-70%
    70-100: 80-95%
    """
    s = _clamp(float(score100), 0.0, 100.0)
    if s <= 30:
        # 0->0.05, 30->0.20
        return round(0.05 + (s / 30.0) * 0.15, 4)
    if s <= 50:
        # 30->0.30, 50->0.50
        return round(0.30 + ((s - 30.0) / 20.0) * 0.20, 4)
    if s <= 70:
        # 50->0.50, 70->0.70
        return round(0.50 + ((s - 50.0) / 20.0) * 0.20, 4)
    # 70->0.80, 100->0.95
    return round(0.80 + ((s - 70.0) / 30.0) * 0.15, 4)


def _max_positions_by_regime_score(score100: float, max_positions: int) -> int:
    s = _clamp(float(score100), 0.0, 100.0)
    upper = max(1, int(max_positions))
    if s <= 30:
        return max(1, min(upper, 2))
    if s <= 50:
        return max(1, min(upper, 3))
    if s <= 70:
        return max(1, min(upper, 5))
    return max(1, min(upper, upper))


def _allow_only_strong_buy(score100: float, regime: str) -> bool:
    # 连续化优先，同时兼容旧 regime 字段
    return score100 <= 30.0 or ("熊" in str(regime))


def _normalize_action(a: str) -> str:
    s = str(a or "").strip().lower()
    return s if s in {"strong_buy", "buy", "hold", "sell", "strong_sell"} else "hold"


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_stock_factor_map() -> Dict[str, dict]:
    """
    可选的个股因子覆盖文件：
    - 默认: training-data/factors/stock_factors_latest.json
    - 格式A: {"sz000001": {"pe_ttm": 12, "pb": 1.6, "roe": 14.2, "northbound_net_5d": 1.2e8}, ...}
    - 格式B: [{"symbol":"000001.SZ", ...}, ...]
    """
    trade_cfg = cfg.get("trade_live", {})
    rel = str(trade_cfg.get("stock_factor_file", "training-data/factors/stock_factors_latest.json")).strip()
    if not rel:
        return {}
    fp = Path(PROJECT_ROOT) / rel
    if not fp.exists():
        return {}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[str, dict] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            sym = _normalize_symbol(str(k))
            if sym and isinstance(v, dict):
                out[sym] = dict(v)
        return out
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            sym = _normalize_symbol(str(row.get("symbol", "")))
            if sym:
                out[sym] = dict(row)
    return out


def _default_plan_file() -> Path:
    out_dir = Path(PROJECT_ROOT) / cfg.get("trade_live", {}).get("log_dir", "output/trade_logs")
    plan_dir = out_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    return plan_dir / f"signal_plan_{datetime.now().strftime('%Y%m%d')}.json"


def _default_weekly_file() -> Path:
    out_dir = Path(PROJECT_ROOT) / cfg.get("trade_live", {}).get("log_dir", "output/trade_logs")
    plan_dir = out_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    iso = datetime.now().isocalendar()
    return plan_dir / f"weekly_buylist_{iso.year}W{iso.week:02d}.json"


def _save_signal_plan(plan_file: Path, payload: dict) -> None:
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    plan_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_signal_plan(plan_file: Path) -> dict:
    data = json.loads(plan_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("plan_file_format_invalid")
    return data


def _build_weekly_buylist(rankings: List[dict], candidates: List[dict], min_n: int, max_n: int) -> List[dict]:
    """
    每周输出 3-10 支“可买入”标的：
    - 优先使用 LLM 精排里 action=buy/strong_buy
    - 若不足 min_n，用规则候选高分补齐
    """
    lower = max(1, int(min_n))
    upper = max(lower, int(max_n))
    picked = []
    seen = set()
    rank_map = {str(r.get("symbol", "")): r for r in rankings}

    for r in rankings:
        action = str(r.get("action", "hold"))
        if action not in {"buy", "strong_buy"}:
            continue
        sym = _normalize_symbol(str(r.get("symbol", "")))
        if not sym or sym in seen:
            continue
        picked.append(
            {
                "symbol": sym,
                "score": float(r.get("score", 0) or 0),
                "action": action,
                "reason": str(r.get("reason", "")),
                "source": "llm_ranking",
            }
        )
        seen.add(sym)
        if len(picked) >= upper:
            break

    if len(picked) < lower:
        for c in candidates:
            sym = _normalize_symbol(str(c.get("symbol", "")))
            if not sym or sym in seen:
                continue
            rr = rank_map.get(str(c.get("symbol", "")), {})
            picked.append(
                {
                    "symbol": sym,
                    "score": float(c.get("score", 0) or 0),
                    "action": str(rr.get("action", "buy_candidate")),
                    "reason": "weekly_fill_from_rule_prefilter",
                    "source": "rule_prefilter",
                }
            )
            seen.add(sym)
            if len(picked) >= lower:
                break

    picked.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    return picked[:upper]


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return rows


def _normalize_symbol(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return s
    sl = s.lower()
    # 兼容 akshare/行情文件常见格式：000001.SZ / 600519.SH / 430001.BJ
    m = re.match(r"^(\d{6})\.(sh|sz|bj)$", sl)
    if m:
        code, market = m.group(1), m.group(2)
        return f"{market}{code}"
    # 已经是执行层格式
    if re.match(r"^(sh|sz|bj)\d{6}$", sl):
        return sl
    if len(sl) == 6 and sl[0] in {"6", "9"}:
        return f"sh{sl}"
    if len(sl) == 6 and sl.isdigit():
        return f"sz{sl}"
    return sl


def _build_candidates(max_universe: int, prefilter_top_k: int) -> Tuple[str, List[dict], dict]:
    data_dir = Path(PROJECT_ROOT) / cfg["data"]["ashare_dir"] / "advanced"
    files = sorted([f for f in data_dir.iterdir() if f.suffix == ".jsonl"])[:max_universe]

    candidates: List[dict] = []
    factor_map = _load_stock_factor_map()
    factor_hit = 0
    regime = "震荡"
    regime_info = ("震荡", None, 0, {"regime_score": 0})

    for f in files:
        rows = _read_jsonl(f)
        if len(rows) < 130:
            continue
        idx = len(rows) - 1
        row = rows[idx]
        prev = rows[idx - 1]

        if len(candidates) == 0:
            regime_info = detect_regime(rows, idx)
            regime = regime_info[0]

        score, _ = compute_score(row, rows, idx, prev, regime_info=regime_info)
        symbol = _normalize_symbol(row.get("symbol") or f.stem)
        frow = factor_map.get(symbol, {})
        if frow:
            factor_hit += 1
        pe_ttm = _to_float(frow.get("pe_ttm", frow.get("pe", 0)), 0.0)
        pb = _to_float(frow.get("pb", 0), 0.0)
        roe = _to_float(frow.get("roe", frow.get("roe_ttm", 0)), 0.0)
        northbound_net_5d = _to_float(
            frow.get("northbound_net_5d", frow.get("northbound_flow_5d", 0)),
            0.0,
        )

        # 基本面因子叠加：过滤价值陷阱 + 奖励基本面稳健标的
        if frow:
            if (pe_ttm <= 0) or (pb <= 0) or (roe <= 0):
                score -= 20
            elif (0 < pe_ttm <= 45) and (0 < pb <= 6.5) and (roe >= 8):
                score += 5
            # 北向资金因子（有数据才生效）：净流入加分，净流出减分
            if northbound_net_5d > 0:
                score += 3
            elif northbound_net_5d < 0:
                score -= 3

        close_20 = rows[idx - 20]["close"] if idx >= 20 else row["close"]
        trend_20d = round((row["close"] - close_20) / close_20 * 100, 2) if close_20 else 0.0
        vol_ma5 = row.get("volume_ma_5", 1) or 1

        candidates.append(
            {
                "symbol": symbol,
                "score": round(score, 2),
                "rsi_14": round(row.get("rsi_14", 50) or 50, 1),
                "trend_20d": trend_20d,
                "bb_position": round(row.get("bb_position", 0.5) or 0.5, 3),
                "hv_20": round(row.get("hv_20", 0) or 0, 1),
                "vol_ratio": round((row.get("volume") or 0) / vol_ma5, 2),
                "turnover_rate": round(row.get("turnover_rate", 0) or 0, 3),
                "pe_ttm": round(pe_ttm, 3),
                "pb": round(pb, 3),
                "roe": round(roe, 3),
                "northbound_net_5d": round(northbound_net_5d, 2),
            }
        )

    regime_score_100 = _regime_score_to_100(regime_info)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return regime, candidates[:prefilter_top_k], {
        "universe_files": len(files),
        "valid_candidates": len(candidates),
        "regime_score_100": regime_score_100,
        "regime_detail": regime_info[3] if len(regime_info) > 3 else {},
        "factor_coverage": {"hit": factor_hit, "total": len(candidates)},
    }


def _build_execution_plan(
    rankings: List[dict],
    candidates: List[dict],
    max_positions: int,
    regime: str,
    regime_score_100: float,
) -> List[dict]:
    trade_cfg = cfg.get("trade_live", {})
    target_exposure = _target_exposure_by_regime_score(regime_score_100)
    regime_cap = _max_positions_by_regime_score(regime_score_100, max_positions=max_positions)
    picks = rankings[: max(1, min(max_positions, regime_cap))]

    # 按 score 分配仓位：高分拿更多仓位；hold 目标仓位为0。
    only_strong_buy = _allow_only_strong_buy(regime_score_100, regime)
    c_map = {_normalize_symbol(str(c.get("symbol", ""))): c for c in candidates}
    rebound_trend_min = float(trade_cfg.get("asym_rebound_confirm_trend_20d", 1.0))
    rebound_rsi_th = float(trade_cfg.get("asym_rebound_rsi_threshold", 42.0))

    filtered_picks = []
    for r in picks:
        rr = dict(r)
        rr["action"] = _normalize_action(rr.get("action", "hold"))
        sym = _normalize_symbol(rr.get("symbol", ""))
        c = c_map.get(sym, {})
        trend_20d = float(c.get("trend_20d", 0.0) or 0.0)
        rsi_14 = float(c.get("rsi_14", 50.0) or 50.0)
        # 非对称买入：低位不抢，要求出现回升确认后再买
        if rr["action"] in {"buy", "strong_buy"}:
            cold_zone = (regime_score_100 <= 50.0) or (rsi_14 <= rebound_rsi_th)
            if cold_zone and trend_20d < rebound_trend_min:
                rr["action"] = "hold"
                prev_reason = str(rr.get("reason", "")).strip()
                rr["reason"] = (prev_reason + " | " if prev_reason else "") + "wait_rebound_confirm"
        filtered_picks.append(rr)

    if only_strong_buy:
        buy_candidates = [r for r in filtered_picks if str(r.get("action", "hold")) == "strong_buy"]
    else:
        buy_candidates = [r for r in filtered_picks if str(r.get("action", "hold")) in {"buy", "strong_buy"}]
    score_power = float(trade_cfg.get("score_weight_power", 1.0) or 1.0)
    weighted = []
    for r in buy_candidates:
        s = max(0.0, float(r.get("score", 0) or 0))
        weighted.append(max(0.0001, s**score_power))
    total_weight = sum(weighted) or 1.0

    alloc_by_symbol = {}
    for r, w in zip(buy_candidates, weighted):
        alloc_by_symbol[str(r.get("symbol", ""))] = (w / total_weight) * target_exposure

    plan = []
    for r in filtered_picks:
        target_pct = round(float(alloc_by_symbol.get(str(r.get("symbol", "")), 0.0)), 4)
        plan.append(
            {
                "symbol": _normalize_symbol(r.get("symbol", "")),
                "action": r.get("action", "hold"),
                "rank": int(r.get("rank", 0) or 0),
                "score": float(r.get("score", 0) or 0),
                "target_position_pct": target_pct,
                "reason": r.get("reason", ""),
                "risk_factors": r.get("risk_factors", []),
                "regime_score_100": regime_score_100,
            }
        )
    return plan


def _fallback_rankings(candidates: List[dict], top_n: int) -> List[dict]:
    """快速兜底精排：按规则评分排序，生成结构化 ranking。"""
    out = []
    for i, c in enumerate(candidates[:top_n], 1):
        score = float(c.get("score", 0) or 0)
        if i <= 2:
            action = "strong_buy"
        elif i <= 5:
            action = "buy"
        else:
            action = "hold"
        out.append(
            {
                "rank": i,
                "symbol": c.get("symbol", ""),
                "score": score,
                "action": action,
                "reason": "fallback_by_rule_score",
                "risk_factors": [],
            }
        )
    return out


def _write_logs(payload: dict) -> Path:
    out_dir = Path(PROJECT_ROOT) / cfg.get("trade_live", {}).get("log_dir", "output/trade_logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = out_dir / f"trade_{ts}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fp_jsonl = out_dir / "trade_history.jsonl"
    with fp_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return fp


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 双层实盘决策与交易日志")
    parser.add_argument("--execute", action="store_true", help="开启执行模式（需 config 允许）")
    parser.add_argument("--max-universe", type=int, default=None, help="扫描股票上限，覆盖 config")
    parser.add_argument("--top-k", type=int, default=None, help="规则初筛候选数量，覆盖 config")
    parser.add_argument("--broker", type=str, default=None, help="执行适配器: eastmoney_paper/eastmoney_sim")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 Qwen 精排，使用规则评分兜底（联调用）")
    parser.add_argument("--signal-only", action="store_true", help="仅生成当日信号计划，不执行下单")
    parser.add_argument("--plan-file", type=str, default="", help="信号计划输出文件路径（配合 --signal-only）")
    parser.add_argument("--use-plan-file", type=str, default="", help="从已有信号计划文件读取并执行")
    parser.add_argument("--weekly-min-picks", type=int, default=None, help="周度买入清单最小数量，默认3")
    parser.add_argument("--weekly-max-picks", type=int, default=None, help="周度买入清单最大数量，默认10")
    args = parser.parse_args()

    trade_cfg = cfg.get("trade_live", {})
    max_universe = int(args.max_universe or trade_cfg.get("max_universe", 1200))
    prefilter_top_k = int(args.top_k or trade_cfg.get("prefilter_top_k", 50))
    max_positions = int(trade_cfg.get("max_positions", 10))
    allow_real = bool(trade_cfg.get("enable_real_orders", False))
    broker = str(args.broker or trade_cfg.get("broker", "eastmoney_paper"))
    weekly_min_picks = int(args.weekly_min_picks or trade_cfg.get("weekly_min_picks", 3))
    weekly_max_picks = int(args.weekly_max_picks or trade_cfg.get("weekly_max_picks", 10))

    t0 = time.time()
    if args.use_plan_file:
        plan_fp = Path(args.use_plan_file).expanduser()
        if not plan_fp.exists():
            raise SystemExit(f"计划文件不存在: {plan_fp}")
        plan_doc = _load_signal_plan(plan_fp)
        regime = str(plan_doc.get("market_regime", "未知"))
        candidates = list(plan_doc.get("candidates", []) or [])
        rankings = list(plan_doc.get("rankings", []) or [])
        stats = dict(plan_doc.get("stats", {}) or {})
        raw_plan = list(plan_doc.get("execution_plan", []) or [])
        if not raw_plan:
            raise SystemExit(f"计划文件中无 execution_plan: {plan_fp}")
    else:
        regime, candidates, stats = _build_candidates(max_universe=max_universe, prefilter_top_k=prefilter_top_k)
        if not candidates:
            raise SystemExit("未生成候选股票，请先检查 training-data/ashare/advanced 数据")

        if args.skip_llm:
            rankings = _fallback_rankings(candidates, top_n=max_positions)
        else:
            rankings = call_skill_stock_rank(regime, candidates, top_n=max_positions)
            if not rankings:
                rankings = _fallback_rankings(candidates, top_n=max_positions)

        raw_plan = _build_execution_plan(
            rankings,
            candidates=candidates,
            max_positions=max_positions,
            regime=regime,
            regime_score_100=float(stats.get("regime_score_100", 50.0)),
        )

    plan = sanitize_plan(raw_plan)

    if args.signal_only:
        plan_fp = Path(args.plan_file).expanduser() if args.plan_file else _default_plan_file()
        weekly_fp = _default_weekly_file()
        weekly_buylist = _build_weekly_buylist(
            rankings=rankings,
            candidates=candidates,
            min_n=weekly_min_picks,
            max_n=weekly_max_picks,
        )
        signal_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "market_regime": regime,
            "broker": broker,
            "stats": {
                **stats,
                "prefilter_top_k": prefilter_top_k,
                "max_positions": max_positions,
                "weekly_min_picks": weekly_min_picks,
                "weekly_max_picks": weekly_max_picks,
                "elapsed_sec": round(time.time() - t0, 2),
            },
            "candidates": candidates,
            "rankings": rankings,
            "execution_plan": plan,
            "weekly_buylist": weekly_buylist,
        }
        _save_signal_plan(plan_fp, signal_payload)
        _save_signal_plan(
            weekly_fp,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "week": datetime.now().strftime("%G-W%V"),
                "market_regime": regime,
                "stats": {
                    **stats,
                    "weekly_min_picks": weekly_min_picks,
                    "weekly_max_picks": weekly_max_picks,
                    "actual_picks": len(weekly_buylist),
                },
                "weekly_buylist": weekly_buylist,
            },
        )
        print(f"信号计划已生成: {plan_fp}")
        print(f"周度买入清单已生成: {weekly_fp}")
        print(f"市场环境: {regime} | 候选: {len(candidates)} | 精筛: {len(rankings)} | 计划条目: {len(plan)}")
        print(f"周度可买入优质股: {len(weekly_buylist)} (目标 {weekly_min_picks}-{weekly_max_picks})")
        return

    mode = "execute" if args.execute else "dry_run"
    if args.execute and not allow_real:
        mode = "dry_run_forced"

    execute_flag = mode == "execute"
    exec_result = execute_plan(plan=plan, execute=execute_flag, broker=broker)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "rules_prefilter -> qwen14b_rank_and_execute",
        "market_regime": regime,
        "mode": mode,
        "broker": broker,
        "stats": {
            **stats,
            "prefilter_top_k": prefilter_top_k,
            "max_positions": max_positions,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "rankings": rankings,
        "execution_plan": plan,
        "execution_result": exec_result,
        "risk_notice": "仅供研究与模拟盘使用，不构成投资建议。",
    }
    log_file = _write_logs(payload)

    print("=" * 60)
    print("Qwen 双层交易决策完成")
    print("=" * 60)
    print(f"市场环境: {regime}")
    print(f"候选数: {len(candidates)} -> 精筛: {len(rankings)}")
    print(f"模式: {mode}")
    print(f"执行适配器: {broker}")
    print(f"日志文件: {log_file}")
    print("-" * 60)
    for item in plan:
        print(
            f"#{item['rank']:>2} {item['symbol']:>10}  {item['action']:<10}  "
            f"score={item['score']:<6.1f}  target={item['target_position_pct']:.2%}"
        )
    print("-" * 60)
    print(
        f"订单意图: {len(exec_result.get('intents', []))} | "
        f"回执: {len(exec_result.get('receipts', []))} | "
        f"blocked={exec_result.get('blocked', False)}"
    )
    receipts = exec_result.get("receipts", []) if isinstance(exec_result, dict) else []
    if receipts:
        print("-" * 60)
        print("交易回执明细:")
        for r in receipts:
            side = str(r.get("side", "")).lower()
            side_cn = "买入" if side == "buy" else ("卖出" if side == "sell" else side)
            code = str(r.get("stock_code", "") or str(r.get("symbol", ""))[-6:])
            name = str(r.get("stock_name", "")).strip()
            name_show = name or "-"
            qty = int(r.get("quantity", 0) or 0)
            price = str(r.get("price", "") or "-")
            amount = float(r.get("order_amount_cny", 0) or 0)
            status = str(r.get("status", ""))
            trade_time = str(r.get("trade_time", "") or "-")
            wth = str(r.get("wth", "") or "-")
            price_source = str(r.get("price_source", "") or "-")
            reason = str(r.get("reason", "") or str(r.get("message", "") or "-"))
            print(
                f"{code:>6} {name_show:<8} {side_cn:<2} "
                f"status={status:<9} 价={price:<8} 量={qty:<6} "
                f"额={amount:>10.2f} 时间={trade_time} 委托号={wth} 价源={price_source} 原因={reason}"
            )
    if exec_result.get("blocked"):
        print(f"阻断原因: {exec_result.get('block_reason')}")
    print("=" * 60)


if __name__ == "__main__":
    main()

