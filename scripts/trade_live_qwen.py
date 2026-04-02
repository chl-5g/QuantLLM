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
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return regime, candidates[:prefilter_top_k], {"universe_files": len(files), "valid_candidates": len(candidates)}


def _build_execution_plan(rankings: List[dict], max_positions: int, regime: str) -> List[dict]:
    trade_cfg = cfg.get("trade_live", {})
    regime_key = "sideways"
    if "牛" in regime:
        regime_key = "bull"
    elif "熊" in regime:
        regime_key = "bear"

    exp_cfg = trade_cfg.get("regime_target_exposure", {})
    target_exposure = float(exp_cfg.get(regime_key, exp_cfg.get("sideways", 0.5)))
    target_exposure = max(0.0, min(1.0, target_exposure))

    regime_pos_cfg = trade_cfg.get("regime_max_positions", {})
    regime_cap = int(regime_pos_cfg.get(regime_key, max_positions) or max_positions)
    picks = rankings[: max(1, min(max_positions, regime_cap))]

    # 按 score 分配仓位：高分拿更多仓位；hold 目标仓位为0。
    buy_candidates = [r for r in picks if str(r.get("action", "hold")) in {"buy", "strong_buy"}]
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
    for r in picks:
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
    args = parser.parse_args()

    trade_cfg = cfg.get("trade_live", {})
    max_universe = int(args.max_universe or trade_cfg.get("max_universe", 1200))
    prefilter_top_k = int(args.top_k or trade_cfg.get("prefilter_top_k", 50))
    max_positions = int(trade_cfg.get("max_positions", 10))
    allow_real = bool(trade_cfg.get("enable_real_orders", False))
    broker = str(args.broker or trade_cfg.get("broker", "eastmoney_paper"))

    t0 = time.time()
    regime, candidates, stats = _build_candidates(max_universe=max_universe, prefilter_top_k=prefilter_top_k)
    if not candidates:
        raise SystemExit("未生成候选股票，请先检查 training-data/ashare/advanced 数据")

    if args.skip_llm:
        rankings = _fallback_rankings(candidates, top_n=max_positions)
    else:
        rankings = call_skill_stock_rank(regime, candidates, top_n=max_positions)
        if not rankings:
            rankings = _fallback_rankings(candidates, top_n=max_positions)

    raw_plan = _build_execution_plan(rankings, max_positions=max_positions, regime=regime)
    plan = sanitize_plan(raw_plan)

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

