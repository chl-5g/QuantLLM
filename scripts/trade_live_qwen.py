#!/usr/bin/env python3
"""两层架构实盘入口：规则初筛 -> Qwen14B精筛(可生成执行指令) -> 交易日志。"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from _config import PROJECT_ROOT, cfg
from backtest_signals import compute_score, detect_regime
from qwen_skills import call_skill_stock_rank


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
    if s.startswith(("sh", "sz", "bj")):
        return s
    if len(s) == 6 and s[0] in {"6", "9"}:
        return f"sh{s}"
    if len(s) == 6:
        return f"sz{s}"
    return s


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

        candidates.append({
            "symbol": symbol,
            "score": round(score, 2),
            "rsi_14": round(row.get("rsi_14", 50) or 50, 1),
            "trend_20d": trend_20d,
            "bb_position": round(row.get("bb_position", 0.5) or 0.5, 3),
            "hv_20": round(row.get("hv_20", 0) or 0, 1),
            "vol_ratio": round((row.get("volume") or 0) / vol_ma5, 2),
            "turnover_rate": round(row.get("turnover_rate", 0) or 0, 3),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return regime, candidates[:prefilter_top_k], {"universe_files": len(files), "valid_candidates": len(candidates)}


def _build_execution_plan(rankings: List[dict], max_positions: int) -> List[dict]:
    trade_cfg = cfg.get("trade_live", {})
    alloc_cfg = trade_cfg.get("allocation", {})
    strong = float(alloc_cfg.get("strong_buy", 0.15))
    normal = float(alloc_cfg.get("buy", 0.10))

    picks = rankings[:max_positions]
    weights = []
    for r in picks:
        action = r.get("action", "hold")
        if action == "strong_buy":
            w = strong
        elif action == "buy":
            w = normal
        else:
            w = 0.0
        weights.append(max(0.0, w))

    total = sum(weights) or 1.0
    plan = []
    for r, w in zip(picks, weights):
        target_pct = round(w / total, 4) if w > 0 else 0.0
        plan.append({
            "symbol": _normalize_symbol(r.get("symbol", "")),
            "action": r.get("action", "hold"),
            "rank": int(r.get("rank", 0) or 0),
            "score": float(r.get("score", 0) or 0),
            "target_position_pct": target_pct,
            "reason": r.get("reason", ""),
            "risk_factors": r.get("risk_factors", []),
        })
    return plan


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
    parser.add_argument("--execute", action="store_true", help="开启真实下单模式（需 config 允许）")
    parser.add_argument("--max-universe", type=int, default=None, help="扫描股票上限，覆盖 config")
    parser.add_argument("--top-k", type=int, default=None, help="规则初筛候选数量，覆盖 config")
    args = parser.parse_args()

    trade_cfg = cfg.get("trade_live", {})
    max_universe = int(args.max_universe or trade_cfg.get("max_universe", 1200))
    prefilter_top_k = int(args.top_k or trade_cfg.get("prefilter_top_k", 50))
    max_positions = int(trade_cfg.get("max_positions", 10))
    allow_real = bool(trade_cfg.get("enable_real_orders", False))

    t0 = time.time()
    regime, candidates, stats = _build_candidates(max_universe=max_universe, prefilter_top_k=prefilter_top_k)
    if not candidates:
        raise SystemExit("未生成候选股票，请先检查 training-data/ashare/advanced 数据")

    rankings = call_skill_stock_rank(regime, candidates, top_n=max_positions)
    if not rankings:
        raise SystemExit("Qwen 精筛失败，未返回可用 JSON 排名")

    plan = _build_execution_plan(rankings, max_positions=max_positions)
    mode = "execute" if args.execute else "dry_run"
    if args.execute and not allow_real:
        mode = "dry_run_forced"

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pipeline": "rules_prefilter -> qwen14b_rank_and_execute",
        "market_regime": regime,
        "mode": mode,
        "stats": {
            **stats,
            "prefilter_top_k": prefilter_top_k,
            "max_positions": max_positions,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "rankings": rankings,
        "execution_plan": plan,
        "risk_notice": "仅供研究与模拟盘使用，不构成投资建议。",
    }
    log_file = _write_logs(payload)

    print("=" * 60)
    print("Qwen 双层交易决策完成")
    print("=" * 60)
    print(f"市场环境: {regime}")
    print(f"候选数: {len(candidates)} -> 精筛: {len(rankings)}")
    print(f"模式: {mode}")
    print(f"日志文件: {log_file}")
    print("-" * 60)
    for item in plan:
        print(
            f"#{item['rank']:>2} {item['symbol']:>10}  {item['action']:<10}  "
            f"score={item['score']:<6.1f}  target={item['target_position_pct']:.2%}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
