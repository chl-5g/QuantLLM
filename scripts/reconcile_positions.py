#!/usr/bin/env python3
"""策略持仓与东方财富账户持仓对账。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from _config import PROJECT_ROOT
from trade_execution import _build_eastmoney_api_context, _symbol_from_code


def _load_strategy_positions(fp: Path) -> Dict[str, float]:
    if not fp.exists():
        return {}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in (data.get("positions", {}) or {}).items()}
    except Exception:
        return {}


def _load_broker_positions() -> Dict[str, float]:
    api, zjzh, err = _build_eastmoney_api_context()
    if err:
        raise RuntimeError(err)
    pos = {}
    for r in api.get_positions(zjzh) or []:
        code = str(r.get("stkCode", "")).strip()
        qty = float(r.get("stkQty", r.get("currentQty", 0)) or 0)
        if len(code) == 6 and code.isdigit() and qty > 0:
            pos[_symbol_from_code(code)] = qty
    return pos


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile strategy and broker positions")
    parser.add_argument("--state-file", default="output/trade_logs/paper_positions_state.json")
    parser.add_argument("--out", default="output/trade_logs/reconcile_latest.json")
    parser.add_argument("--qty-tol", type=float, default=1.0, help="数量容差（股）")
    args = parser.parse_args()

    st = _load_strategy_positions(Path(PROJECT_ROOT) / args.state_file)
    br = _load_broker_positions()

    symbols = sorted(set(st.keys()) | set(br.keys()))
    mismatches = []
    for s in symbols:
        sv = float(st.get(s, 0.0))
        bv = float(br.get(s, 0.0))
        if abs(sv - bv) > args.qty_tol:
            mismatches.append({"symbol": s, "strategy_qty": sv, "broker_qty": bv, "diff": round(sv - bv, 2)})

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "strategy_count": len(st),
        "broker_count": len(br),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    out = Path(PROJECT_ROOT) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[RECONCILE] mismatch={len(mismatches)} out={out}")
    if mismatches:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
