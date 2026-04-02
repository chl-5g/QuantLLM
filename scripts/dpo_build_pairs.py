#!/usr/bin/env python3
"""从实盘交易日志构建 DPO 偏好对数据。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from _config import PROJECT_ROOT


def _iter_trade_logs(log_dir: Path, days: int) -> List[dict]:
    files = sorted(log_dir.glob("trade_*.json"))
    if days > 0:
        cutoff = datetime.now().timestamp() - days * 86400
        files = [f for f in files if f.stat().st_mtime >= cutoff]
    out = []
    for fp in files:
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def _score_item(x: dict) -> float:
    base = float(x.get("score", 0) or 0)
    action = str(x.get("action", "hold"))
    bonus = 6.0 if action == "strong_buy" else (3.0 if action == "buy" else 0.0)
    return base + bonus


def build_pairs(rows: List[dict], min_gap: float) -> List[dict]:
    by_day: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        ts = str(row.get("timestamp", ""))
        day = ts[:10] if len(ts) >= 10 else "unknown"
        for r in row.get("rankings", []) or []:
            if not isinstance(r, dict):
                continue
            item = dict(r)
            item["day"] = day
            by_day[day].append(item)

    pairs = []
    for day, items in by_day.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=_score_item, reverse=True)
        top = items[:5]
        low = list(reversed(items[-5:]))
        for chosen in top:
            for rejected in low:
                gap = _score_item(chosen) - _score_item(rejected)
                if gap < min_gap:
                    continue
                pairs.append(
                    {
                        "date": day,
                        "chosen": {
                            "symbol": chosen.get("symbol", ""),
                            "action": chosen.get("action", ""),
                            "score": float(chosen.get("score", 0) or 0),
                            "reason": str(chosen.get("reason", "")),
                        },
                        "rejected": {
                            "symbol": rejected.get("symbol", ""),
                            "action": rejected.get("action", ""),
                            "score": float(rejected.get("score", 0) or 0),
                            "reason": str(rejected.get("reason", "")),
                        },
                        "score_gap": round(gap, 3),
                        "source": "live_rankings",
                    }
                )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from trade logs")
    parser.add_argument("--log-dir", default="output/trade_logs", help="交易日志目录（相对项目根）")
    parser.add_argument("--days", type=int, default=90, help="最近 N 天日志")
    parser.add_argument("--min-gap", type=float, default=8.0, help="最小打分差")
    parser.add_argument("--out", default="output/dpo_pairs_live.jsonl", help="输出文件（相对项目根）")
    args = parser.parse_args()

    log_dir = Path(PROJECT_ROOT) / args.log_dir
    out_fp = Path(PROJECT_ROOT) / args.out
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    rows = _iter_trade_logs(log_dir, days=args.days)
    pairs = build_pairs(rows, min_gap=args.min_gap)

    with out_fp.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[DPO] logs={len(rows)} pairs={len(pairs)} out={out_fp}")


if __name__ == "__main__":
    main()
