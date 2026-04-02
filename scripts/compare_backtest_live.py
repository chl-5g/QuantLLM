#!/usr/bin/env python3
"""回测与实盘收益对比（基于账户净值快照）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from _config import PROJECT_ROOT
from trade_execution import _build_eastmoney_api_context, _to_float


def _append_snapshot(fp: Path, asset: float) -> None:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "total_asset": round(asset, 2)}
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(fp: Path):
    out = []
    if not fp.exists():
        return out
    for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _load_backtest(fp: Path) -> dict:
    if not fp.exists():
        return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest and live performance")
    parser.add_argument("--backtest", default="output/backtest_results.json")
    parser.add_argument("--snapshots", default="output/trade_logs/live_balance_snapshots.jsonl")
    parser.add_argument("--out", default="output/trade_logs/backtest_live_compare.json")
    args = parser.parse_args()

    api, zjzh, err = _build_eastmoney_api_context()
    if err:
        raise SystemExit(f"balance_api_unavailable: {err}")

    bal = api.get_balance(zjzh)
    total_asset = _to_float(bal.get("zzc", 0), 0.0)
    if total_asset <= 0:
        raise SystemExit("invalid_total_asset")

    snap_fp = Path(PROJECT_ROOT) / args.snapshots
    snap_fp.parent.mkdir(parents=True, exist_ok=True)
    _append_snapshot(snap_fp, total_asset)
    snaps = _load_jsonl(snap_fp)

    first_asset = float(snaps[0].get("total_asset", total_asset)) if snaps else total_asset
    live_ret_pct = ((total_asset / first_asset) - 1.0) * 100 if first_asset > 0 else 0.0

    backtest = _load_backtest(Path(PROJECT_ROOT) / args.backtest)
    s1 = ((backtest.get("strategy1", {}) or {}).get("metrics", {}) or {})
    bm = ((backtest.get("benchmark", {}) or {}).get("metrics", {}) or {})
    bt_ret = float(s1.get("total_return_pct", 0) or 0)
    bm_ret = float(bm.get("total_return_pct", 0) or 0)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "zjzh": zjzh,
        "live": {
            "snapshot_count": len(snaps),
            "first_asset": round(first_asset, 2),
            "latest_asset": round(total_asset, 2),
            "return_pct": round(live_ret_pct, 3),
        },
        "backtest": {
            "strategy1_total_return_pct": bt_ret,
            "benchmark_total_return_pct": bm_ret,
            "excess_vs_benchmark_pct": round(bt_ret - bm_ret, 3),
        },
        "note": "live return is computed from balance snapshots; keep running daily for stable comparison.",
    }

    out_fp = Path(PROJECT_ROOT) / args.out
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[COMPARE] live_ret={payload['live']['return_pct']}% snapshots={len(snaps)} out={out_fp}")


if __name__ == "__main__":
    main()
