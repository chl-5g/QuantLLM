#!/usr/bin/env python3
"""Trade execution core: broker adapter + risk controls + idempotency."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import requests
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from _config import PROJECT_ROOT, cfg


@dataclass
class OrderIntent:
    symbol: str
    side: str
    delta_pct: float
    target_pct: float
    score: float
    reason: str
    rank: int


def _logs_dir() -> Path:
    rel = cfg.get("trade_live", {}).get("log_dir", "output/trade_logs")
    p = Path(PROJECT_ROOT) / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_file() -> Path:
    return _logs_dir() / "paper_positions_state.json"


def _idem_file() -> Path:
    return _logs_dir() / "executed_orders.jsonl"


def _pending_file() -> Path:
    return _logs_dir() / "eastmoney_pending_orders.jsonl"


def _rollback_file() -> Path:
    return _logs_dir() / "eastmoney_rollback_orders.jsonl"


def _records_file() -> Path:
    return _logs_dir() / "trade_records.jsonl"


def _kill_file() -> Path:
    rel = cfg.get("trade_live", {}).get("kill_switch_file", "output/trade_live.KILL")
    return Path(PROJECT_ROOT) / rel


def _load_state() -> Dict[str, float]:
    fp = _state_file()
    if not fp.exists():
        return {}
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in data.get("positions", {}).items()}
    except Exception:
        return {}


def _save_state(positions: Dict[str, float]) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "positions": positions,
    }
    _state_file().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_idem_today(broker: str) -> set[str]:
    """
    加载今日已执行的幂等 key。
    - accepted/submitted/filled → 直接屏蔽（已成交不重复）
    - failed → 允许重试，但超过 max_retry_per_symbol 次后屏蔽
    """
    fp = _idem_file()
    if not fp.exists():
        return set()
    today = datetime.now().strftime("%Y-%m-%d")
    max_retry = int(cfg.get("trade_live", {}).get("max_retry_per_symbol", 3))

    blocked = set()
    fail_count: Dict[str, int] = {}  # key -> failed count

    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not str(row.get("date", "")).startswith(today):
                continue
            if str(row.get("broker", "")) != broker:
                continue
            key = str(row.get("idempotency_key", ""))
            status = str(row.get("status", "")).lower()
            if status in ("accepted", "submitted", "filled"):
                blocked.add(key)
            elif status == "failed":
                fail_count[key] = fail_count.get(key, 0) + 1
                if fail_count[key] >= max_retry:
                    blocked.add(key)
    return blocked


def _append_idem(key: str, intent: OrderIntent, status: str, broker: str, receipt: Optional[dict] = None) -> None:
    receipt = receipt or {}
    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "broker": broker,
        "idempotency_key": key,
        "status": status,
        "symbol": intent.symbol,
        "stock_code": str(receipt.get("stock_code", "")),
        "stock_name": str(receipt.get("stock_name", "")),
        "side": intent.side,
        "delta_pct": intent.delta_pct,
        "target_pct": intent.target_pct,
        "trade_time": str(receipt.get("trade_time", "")),
        "order_amount_cny": float(receipt.get("order_amount_cny", 0) or 0),
        "price": str(receipt.get("price", "")),
        "quantity": int(receipt.get("quantity", 0) or 0),
        "wth": str(receipt.get("wth", "")),
    }
    with _idem_file().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _daily_trade_count(broker: str) -> int:
    fp = _idem_file()
    if not fp.exists():
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    n = 0
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith('{"date": "' + today):
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("broker", "")) == broker:
                n += 1
    return n


def _idem_key(intent: OrderIntent, broker: str) -> str:
    """幂等 key 只含 (日期|broker|symbol|side)，不含 delta/target 避免微变绕过。"""
    base = f"{datetime.now().strftime('%Y-%m-%d')}|{broker}|{intent.symbol}|{intent.side}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def sanitize_plan(plan: List[dict]) -> List[dict]:
    out = []
    for row in plan:
        sym = str(row.get("symbol", "")).strip()
        if not sym:
            continue
        action = str(row.get("action", "hold")).strip().lower()
        if action not in {"strong_buy", "buy", "hold", "sell", "strong_sell"}:
            action = "hold"
        risk_factors = row.get("risk_factors", [])
        if not isinstance(risk_factors, list):
            risk_factors = [str(risk_factors)]
        out.append(
            {
                "symbol": sym,
                "action": action,
                "rank": int(row.get("rank", 0) or 0),
                "score": float(row.get("score", 0) or 0),
                "target_position_pct": 0.0 if action in ("sell", "strong_sell") else float(row.get("target_position_pct", 0) or 0),
                "reason": str(row.get("reason", ""))[:300],
                "risk_factors": [str(x)[:80] for x in risk_factors[:8]],
            }
        )
    return out


def _cap_targets(plan: List[dict]) -> Tuple[List[dict], Dict[str, str]]:
    rcfg = cfg.get("risk_control", {})
    max_pos = float(rcfg.get("max_position_pct", 0.10))
    max_total = float(rcfg.get("max_total_position_pct", 0.80))
    reason = {}

    capped = []
    for r in plan:
        t = max(0.0, min(float(r["target_position_pct"]), max_pos))
        rr = dict(r)
        if t < float(r["target_position_pct"]):
            reason[rr["symbol"]] = f"cap_single:{max_pos:.2%}"
        rr["target_position_pct"] = t
        capped.append(rr)

    total = sum(x["target_position_pct"] for x in capped)
    if total > max_total and total > 0:
        scale = max_total / total
        for rr in capped:
            rr["target_position_pct"] = round(rr["target_position_pct"] * scale, 6)
            reason[rr["symbol"]] = reason.get(rr["symbol"], "") + f"|cap_total:{max_total:.2%}"
    return capped, reason


def build_order_intents(plan: List[dict], current_positions: Dict[str, float]) -> Tuple[List[OrderIntent], Dict[str, str]]:
    tcfg = cfg.get("trade_live", {})
    min_delta = float(tcfg.get("min_rebalance_delta", 0.01))

    safe_plan, risk_marks = _cap_targets(plan)
    desired = {x["symbol"]: float(x["target_position_pct"]) for x in safe_plan if x["target_position_pct"] > 0}
    universe = set(desired.keys()) | set(current_positions.keys())

    by_symbol = {x["symbol"]: x for x in safe_plan}
    intents: List[OrderIntent] = []

    for sym in sorted(universe):
        curr = float(current_positions.get(sym, 0.0))
        tgt = float(desired.get(sym, 0.0))
        delta = round(tgt - curr, 6)
        if abs(delta) < min_delta:
            continue
        row = by_symbol.get(sym, {"score": 0, "reason": "rebalance"})
        intents.append(
            OrderIntent(
                symbol=sym,
                side="buy" if delta > 0 else "sell",
                delta_pct=abs(delta),
                target_pct=tgt,
                score=float(row.get("score", 0.0)),
                reason=str(row.get("reason", "")),
                rank=int(row.get("rank", 0) or 0),
            )
        )
    intents.sort(key=lambda x: (0 if x.side == "sell" else 1, x.rank))
    return intents, risk_marks


def _paper_execute(intents: List[OrderIntent], execute: bool) -> Dict[str, object]:
    before = _load_state()
    after = deepcopy(before)
    receipts = []
    rollback = False
    err_msg = ""
    try:
        for it in intents:
            cur = float(after.get(it.symbol, 0.0))
            nxt = max(0.0, it.target_pct)
            if execute:
                after[it.symbol] = nxt
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "filled" if execute else "simulated",
                    "filled_pct": it.delta_pct if execute else 0.0,
                    "before_pct": cur,
                    "after_pct": nxt if execute else cur,
                }
            )
        if execute:
            after = {k: v for k, v in after.items() if v > 1e-9}
            _save_state(after)
    except Exception as e:  # rollback guard
        rollback = True
        err_msg = str(e)
        _save_state(before)

    return {
        "broker": "eastmoney_paper",
        "rollback": rollback,
        "error": err_msg,
        "receipts": receipts,
        "positions_before": before,
        "positions_after": _load_state() if execute and not rollback else before,
    }


def _eastmoney_queue(intents: List[OrderIntent], execute: bool) -> Dict[str, object]:
    rows = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for it in intents:
        rows.append(
            {
                "ts": now,
                "symbol": it.symbol,
                "side": it.side,
                "delta_pct": it.delta_pct,
                "target_pct": it.target_pct,
                "score": it.score,
                "reason": it.reason,
                "status": "queued_manual" if execute else "simulated",
            }
        )
    if rows:
        with _pending_file().open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "broker": "eastmoney_sim",
        "rollback": False,
        "error": "",
        "receipts": rows,
        "positions_before": {},
        "positions_after": {},
        "note": "已写入待执行队列，需人工或后续 API 适配器执行。",
    }


def _load_eastmoney_cookie_str() -> str:
    env_cookie = os.getenv("EASTMONEY_COOKIE_STR", "").strip()
    if env_cookie:
        return env_cookie

    state_fp = Path(PROJECT_ROOT) / ".eastmoney_cookies.json"
    if not state_fp.exists():
        return ""
    try:
        state = json.loads(state_fp.read_text(encoding="utf-8"))
        cookies = state.get("cookies", [])
        pairs = []
        for c in cookies:
            if not isinstance(c, dict):
                continue
            domain = str(c.get("domain", ""))
            if "eastmoney.com" not in domain:
                continue
            name = str(c.get("name", "")).strip()
            value = str(c.get("value", ""))
            if name:
                pairs.append(f"{name}={value}")
        return "; ".join(pairs)
    except Exception:
        return ""


def _extract_code(symbol: str) -> str:
    s = str(symbol or "").strip().lower()
    if len(s) >= 8 and s[:2] in {"sh", "sz", "bj"} and s[2:].isdigit():
        return s[2:8]
    if len(s) >= 6 and s[-6:].isdigit():
        return s[-6:]
    return ""


def _symbol_from_code(code: str) -> str:
    c = str(code or "").strip()
    if len(c) != 6 or not c.isdigit():
        return c
    if c[0] in {"6", "9"}:
        return f"sh{c}"
    return f"sz{c}"


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _is_result_ok(resp: dict) -> bool:
    if not isinstance(resp, dict):
        return False
    try:
        return int(resp.get("result", -1)) == 0
    except Exception:
        return False


def _append_trade_records(receipts: List[dict], broker: str, execute: bool) -> None:
    if not receipts:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "execute" if execute else "dry_run"
    with _records_file().open("a", encoding="utf-8") as f:
        for r in receipts:
            side = str(r.get("side", "")).lower()
            side_cn = "买入" if side == "buy" else ("卖出" if side == "sell" else side)
            code = str(r.get("stock_code", "") or _extract_code(str(r.get("symbol", ""))))
            name = str(r.get("stock_name", ""))
            row = {
                "timestamp": ts,
                "broker": broker,
                "mode": mode,
                "zjzh": str(r.get("zjzh", "")),
                "symbol": str(r.get("symbol", "")),
                "stock_code": code,
                "stock_name": name,
                "side": side,
                "side_cn": side_cn,
                "status": str(r.get("status", "")),
                "reason": str(r.get("reason", "")),
                "message": str(r.get("message", "")),
                "trade_time": str(r.get("trade_time", "")),
                "order_amount_cny": float(r.get("order_amount_cny", 0) or 0),
                "price": str(r.get("price", "")),
                "price_source": str(r.get("price_source", "")),
                "quantity": int(r.get("quantity", 0) or 0),
                "wth": str(r.get("wth", "")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _score_factor(score: float, low: float, high: float, reverse: bool = False) -> float:
    lo = float(low)
    hi = float(high)
    if hi <= lo:
        base = 0.5
    else:
        base = (float(score) - lo) / (hi - lo)
    base = max(0.0, min(1.0, base))
    if reverse:
        base = 1.0 - base
    # 限制在 0.6x~1.4x，避免过度激进。
    return 0.6 + 0.8 * base


def _asym_sell_factor(score: float, low: float, high: float) -> float:
    """
    非对称卖出：高位分批卖，避免一次性清仓。
    将 score 因子映射到 [min_frac, max_frac]，默认约 [0.30, 0.65]。
    """
    tcfg = cfg.get("trade_live", {})
    min_frac = max(0.05, min(1.0, float(tcfg.get("asym_sell_min_fraction", 0.30))))
    max_frac = max(min_frac, min(1.0, float(tcfg.get("asym_sell_max_fraction", 0.65))))
    raw = _score_factor(score, low, high, reverse=True)  # raw in [0.6, 1.4]
    norm = max(0.0, min(1.0, (raw - 0.6) / 0.8))
    return min_frac + norm * (max_frac - min_frac)


def _fetch_realtime_quote(code: str, fallback_price: str) -> Dict[str, object]:
    secid = ("1." + code) if str(code).startswith(("6", "9")) else ("0." + code)
    sources = [
        ("eastmoney_push2", "https://push2.eastmoney.com/api/qt/stock/get"),
        ("eastmoney_push2delay", "https://push2delay.eastmoney.com/api/qt/stock/get"),
    ]
    s = requests.Session()
    s.trust_env = False
    for source_name, source_url in sources:
        for i in range(2):
            try:
                r = s.get(
                    source_url,
                    params={"secid": secid, "fields": "f43,f51,f52"},
                    timeout=8,
                )
                data = r.json().get("data", {}) if r.status_code == 200 else {}
                f43 = _to_float(data.get("f43", 0), 0.0)
                if f43 > 0:
                    px = round(f43 / 100.0, 2)
                    up = round(_to_float(data.get("f51", 0), 0.0) / 100.0, 2)
                    down = round(_to_float(data.get("f52", 0), 0.0) / 100.0, 2)
                    return {
                        "price": f"{px:.2f}",
                        "price_source": source_name,
                        "ok": True,
                        "limit_up": up if up > 0 else 0.0,
                        "limit_down": down if down > 0 else 0.0,
                    }
            except Exception:
                if i < 1:
                    time.sleep(0.25 * (i + 1))
                continue
    # 不再使用本地收盘价回退，避免“看起来像实时价但其实不是”。
    return {
        "price": str(fallback_price),
        "price_source": "fallback_default",
        "ok": False,
        "limit_up": 0.0,
        "limit_down": 0.0,
    }


def _fetch_realtime_price(code: str, fallback_price: str) -> Tuple[str, str, bool]:
    q = _fetch_realtime_quote(code, fallback_price)
    return str(q.get("price", fallback_price)), str(q.get("price_source", "fallback_default")), bool(q.get("ok", False))


def _resolve_eastmoney_zjzh(api) -> str:
    env_zjzh = os.getenv("EASTMONEY_ZJZH", "").strip()
    if env_zjzh:
        return env_zjzh
    cfg_zjzh = str(cfg.get("trade_live", {}).get("eastmoney_zjzh", "")).strip()
    if cfg_zjzh:
        return cfg_zjzh
    portfolios = api.list_portfolios()
    if not portfolios:
        # 兜底：在未拿到组合列表时，用常见/历史组合号探测可用账户。
        probe_candidates = []
        for key in ("eastmoney_zjzh", "default_zjzh"):
            v = str(cfg.get("trade_live", {}).get(key, "")).strip()
            if v:
                probe_candidates.append(v)
        probe_candidates.extend(["260914300000052248", "260680400000080882"])
        seen = set()
        for cand in probe_candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            try:
                bal = api.get_balance(cand)
                if isinstance(bal, dict) and any(k in bal for k in ("zzc", "kyye", "mktVal")):
                    return cand
            except Exception:
                continue
        return ""
    for p in portfolios:
        if int(p.get("permit", 0) or 0) == 1 and p.get("zjzh"):
            return str(p.get("zjzh"))
    return str(portfolios[0].get("zjzh", ""))


def _build_eastmoney_api_context():
    cookie_str = _load_eastmoney_cookie_str()
    if not cookie_str:
        return None, "", "missing_cookie:EASTMONEY_COOKIE_STR_or_.eastmoney_cookies.json"

    try:
        from eastmoney_http_api import EastMoneyAPI, EastMoneyConfig
    except Exception as e:
        return None, "", f"import_eastmoney_http_api_failed:{e}"

    em_cfg = EastMoneyConfig.from_cookies(cookie_str)
    api = EastMoneyAPI(em_cfg)
    zjzh = _resolve_eastmoney_zjzh(api)
    if not zjzh:
        return None, "", "resolve_zjzh_failed"
    return api, zjzh, ""


def _get_eastmoney_positions_pct() -> Tuple[Dict[str, float], str]:
    api, zjzh, err = _build_eastmoney_api_context()
    if err:
        return {}, err
    try:
        bal = api.get_balance(zjzh)
        total_asset = _to_float(bal.get("zzc", 0), 0.0)
        if total_asset <= 0:
            return {}, ""
        positions = api.get_positions(zjzh)
        out: Dict[str, float] = {}
        for p in positions:
            code = str(p.get("stkCode", "")).strip()
            if len(code) != 6 or not code.isdigit():
                continue
            mv = _to_float(p.get("mktVal", 0), 0.0)
            if mv <= 0:
                qty = _to_float(p.get("stkQty", p.get("currentQty", 0)), 0.0)
                price = _to_float(p.get("lastPrice", p.get("currentPrice", 0)), 0.0)
                mv = qty * price
            if mv <= 0:
                continue
            out[_symbol_from_code(code)] = round(mv / total_asset, 6)
        return out, ""
    except Exception as e:
        return {}, f"positions_fetch_failed:{e}"


def _eastmoney_api_execute(intents: List[OrderIntent], execute: bool) -> Dict[str, object]:
    api, zjzh, err = _build_eastmoney_api_context()
    if err:
        return {
            "broker": "eastmoney_sim",
            "rollback": False,
            "error": err,
            "receipts": [],
            "positions_before": {},
            "positions_after": {},
        }

    bal = api.get_balance(zjzh)
    total_asset = _to_float(bal.get("zzc", 0), 0.0)
    available_cash = _to_float(bal.get("kyye", 0), 0.0)
    pos_rows = api.get_positions(zjzh)
    held_codes = set()
    for p in pos_rows:
        code = str(p.get("stkCode", "")).strip()
        qty = _to_int(p.get("stkQty", p.get("currentQty", 0)), 0)
        if len(code) == 6 and code.isdigit() and qty > 0:
            held_codes.add(code)
    tcfg = cfg.get("trade_live", {})
    min_amt = float(tcfg.get("min_order_amount_cny", 1000))
    max_amt = float(tcfg.get("max_order_amount_cny", 50000))
    fallback_price = str(tcfg.get("default_order_price", "12.00"))

    receipts = []
    order_cache: Dict[str, dict] = {}
    open_order_keys = set()
    today_buy_codes = set()

    def _enrich_from_orders(wth: str) -> dict:
        if not wth:
            return {}
        if wth in order_cache:
            return order_cache[wth]
        try:
            rows = api.get_today_orders(zjzh)
            if isinstance(rows, list):
                for row in rows:
                    ww = str(row.get("wth", ""))
                    if ww:
                        order_cache[ww] = row
            return order_cache.get(wth, {})
        except Exception:
            return {}

    try:
        today_orders = api.get_today_orders(zjzh)
        for row in (today_orders or []):
            code = str(row.get("stkCode", "")).strip()
            mmflag = str(row.get("mmflag", "")).strip()
            side = "buy" if mmflag == "0" else ("sell" if mmflag == "1" else "")
            wtsl = _to_int(row.get("wtsl", 0), 0)
            cjsl = _to_int(row.get("cjsl", 0), 0)
            if code and side == "buy" and cjsl > 0:
                today_buy_codes.add(code)
            if code and side and (wtsl <= 0 or cjsl < wtsl):
                open_order_keys.add((code, side))
    except Exception:
        pass

    sell_intents = [x for x in intents if x.side == "sell"]
    buy_intents = [x for x in intents if x.side == "buy"]
    buy_low = min([x.score for x in buy_intents], default=0.0)
    buy_high = max([x.score for x in buy_intents], default=0.0)
    sell_low = min([x.score for x in sell_intents], default=0.0)
    sell_high = max([x.score for x in sell_intents], default=0.0)

    # 阶段1：先根据当前持仓执行 hold/sell，不持仓的卖单直接跳过。
    for it in sell_intents:
        code = _extract_code(it.symbol)
        if not code:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "failed",
                    "reason": "invalid_symbol",
                }
            )
            continue

        if code not in held_codes:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "no_position_to_sell",
                    "zjzh": zjzh,
                }
            )
            continue
        if code in today_buy_codes:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "t1_same_day_buy_block",
                    "zjzh": zjzh,
                }
            )
            continue
        if (code, "sell") in open_order_keys:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "existing_open_order_today",
                    "zjzh": zjzh,
                }
            )
            continue

        quote = _fetch_realtime_quote(code, fallback_price)
        order_price = str(quote.get("price", fallback_price))
        price_source = str(quote.get("price_source", "fallback_default"))
        price_ok = bool(quote.get("ok", False))
        if not price_ok:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "quote_api_unavailable",
                    "price_source": price_source,
                    "zjzh": zjzh,
                }
            )
            continue
        px = _to_float(order_price, 0.0)
        limit_down = _to_float(quote.get("limit_down", 0.0), 0.0)
        if limit_down > 0 and px <= limit_down + 1e-6:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "limit_down_locked",
                    "price": order_price,
                    "limit_down": f"{limit_down:.2f}",
                    "price_source": price_source,
                    "zjzh": zjzh,
                }
            )
            continue
        # 查询最大可卖，避免“明明没仓位还卖”。
        max_sell_resp = api.get_max_sell(code, order_price, zjzh=zjzh)
        max_sell = _to_int(max_sell_resp.get("orderLimit", 0), 0)
        sell_factor = _asym_sell_factor(it.score, sell_low, sell_high)
        qty_raw = min(max_sell, int(max_sell * sell_factor))
        qty = max(0, (qty_raw // 100) * 100)
        if qty < 100:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "max_sell_is_zero",
                    "zjzh": zjzh,
                }
            )
            continue

        if not execute:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "simulated",
                    "price": order_price,
                    "price_source": price_source,
                    "quantity": qty,
                    "sell_factor": round(sell_factor, 4),
                    "zjzh": zjzh,
                }
            )
            continue

        try:
            resp = api.place_order(stk_code=code, price=order_price, quantity=qty, side="sell", zjzh=zjzh)
            ok = _is_result_ok(resp)
            wth = str(resp.get("wth", ""))
            od = _enrich_from_orders(wth) if ok else {}
            qty_done = _to_int(od.get("wtsl", qty), qty)
            price_done = str(od.get("wtjg", order_price))
            trade_time = ""
            if od.get("wtrq") or od.get("wtsj"):
                trade_time = f"{od.get('wtrq', '')} {od.get('wtsj', '')}".strip()
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "stock_name": str(od.get("stkName", "")),
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "submitted" if ok else "failed",
                    "reason": "" if ok else str(resp.get("message", "order_failed")),
                    "message": str(resp.get("message", "")),
                    "result": resp.get("result"),
                    "wth": wth,
                    "trade_time": trade_time,
                    "price": price_done,
                    "price_source": price_source,
                    "quantity": qty_done,
                    "sell_factor": round(sell_factor, 4),
                    "order_amount_cny": round(_to_float(price_done, _to_float(fallback_price, 12.0)) * qty_done, 2),
                    "zjzh": zjzh,
                    "verified": ok and bool(wth),
                }
            )
            if ok:
                held_codes.discard(code)
                open_order_keys.add((code, "sell"))
        except Exception as e:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "failed",
                    "reason": f"exception:{e}",
                    "price": fallback_price,
                    "price_source": price_source,
                    "quantity": qty,
                    "zjzh": zjzh,
                }
            )

    # 阶段2：卖出后刷新可用资金，再决定是否买入。
    try:
        bal2 = api.get_balance(zjzh)
        available_cash = _to_float(bal2.get("kyye", available_cash), available_cash)
        total_asset = _to_float(bal2.get("zzc", total_asset), total_asset)
    except Exception:
        pass

    for it in buy_intents:
        code = _extract_code(it.symbol)
        if not code:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "failed",
                    "reason": "invalid_symbol",
                    "zjzh": zjzh,
                }
            )
            continue

        est_amt = max(min_amt, total_asset * max(0.0, float(it.delta_pct)))
        est_amt = min(est_amt, max_amt)
        buy_factor = _score_factor(it.score, buy_low, buy_high, reverse=False)
        est_amt = max(min_amt, min(max_amt, est_amt * buy_factor))
        quote = _fetch_realtime_quote(code, fallback_price)
        order_price = str(quote.get("price", fallback_price))
        price_source = str(quote.get("price_source", "fallback_default"))
        price_ok = bool(quote.get("ok", False))
        if not price_ok:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "quote_api_unavailable",
                    "price_source": price_source,
                    "zjzh": zjzh,
                }
            )
            continue
        px = max(_to_float(order_price, 12.0), 0.01)
        if (code, "buy") in open_order_keys:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "existing_open_order_today",
                    "zjzh": zjzh,
                }
            )
            continue
        px = _to_float(order_price, 0.0)
        limit_up = _to_float(quote.get("limit_up", 0.0), 0.0)
        if limit_up > 0 and px >= limit_up - 1e-6:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "limit_up_locked",
                    "price": order_price,
                    "limit_up": f"{limit_up:.2f}",
                    "price_source": price_source,
                    "zjzh": zjzh,
                }
            )
            continue

        max_buy_resp = api.get_max_buy(code, order_price, zjzh=zjzh)
        max_buy = _to_int(max_buy_resp.get("orderLimit", 0), 0)
        cash_limited = int(available_cash / px)
        budget_limited = int(est_amt / px)
        if budget_limited > 0:
            raw_qty = min(max_buy, cash_limited, budget_limited)
        else:
            raw_qty = min(max_buy, cash_limited)
        qty = max(0, (raw_qty // 100) * 100)

        if qty < 100:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "skipped",
                    "reason": "insufficient_cash_or_limit",
                    "available_cash": round(available_cash, 2),
                    "order_limit": max_buy,
                    "zjzh": zjzh,
                }
            )
            continue

        if not execute:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "simulated",
                    "order_amount_cny": round(est_amt, 2),
                    "price": order_price,
                    "price_source": price_source,
                    "quantity": qty,
                    "zjzh": zjzh,
                }
            )
            continue

        try:
            resp = api.place_order(
                stk_code=code,
                price=order_price,
                quantity=qty,
                side=it.side,
                zjzh=zjzh,
            )
            ok = _is_result_ok(resp)
            wth = str(resp.get("wth", ""))
            od = _enrich_from_orders(wth) if ok else {}
            qty_done = _to_int(od.get("wtsl", qty), qty)
            price_done = str(od.get("wtjg", order_price))
            order_amt = round(qty_done * _to_float(price_done, px), 2)
            trade_time = ""
            if od.get("wtrq") or od.get("wtsj"):
                trade_time = f"{od.get('wtrq', '')} {od.get('wtsj', '')}".strip()
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "stock_name": str(od.get("stkName", "")),
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "submitted" if ok else "failed",
                    "reason": "" if ok else str(resp.get("message", "order_failed")),
                    "message": str(resp.get("message", "")),
                    "result": resp.get("result"),
                    "wth": wth,
                    "trade_time": trade_time,
                    "order_amount_cny": order_amt,
                    "price": price_done,
                    "price_source": price_source,
                    "quantity": qty_done,
                    "zjzh": zjzh,
                    "verified": ok and bool(wth),
                }
            )
            if ok:
                available_cash = max(0.0, available_cash - order_amt)
                open_order_keys.add((code, "buy"))
        except Exception as e:
            receipts.append(
                {
                    "symbol": it.symbol,
                    "stock_code": code,
                    "side": it.side,
                    "delta_pct": it.delta_pct,
                    "target_pct": it.target_pct,
                    "status": "failed",
                    "reason": f"exception:{e}",
                    "price": order_price,
                    "price_source": price_source,
                    "quantity": qty,
                    "zjzh": zjzh,
                }
            )

    return {
        "broker": "eastmoney_sim",
        "rollback": False,
        "error": "",
        "receipts": receipts,
        "positions_before": {"held_codes": sorted(list(held_codes))},
        "positions_after": {},
        "executor_ok": True,
        "mode": "api_http",
    }


def _eastmoney_executor(intents: List[OrderIntent], execute: bool) -> Dict[str, object]:
    rows = [
        {
            "symbol": it.symbol,
            "side": it.side,
            "delta_pct": it.delta_pct,
            "target_pct": it.target_pct,
            "score": it.score,
            "reason": it.reason,
            "rank": it.rank,
        }
        for it in intents
    ]
    if not rows:
        return {
            "broker": "eastmoney_sim",
            "rollback": False,
            "error": "",
            "receipts": [],
            "positions_before": {},
            "positions_after": {},
        }

    script = Path(PROJECT_ROOT) / "scripts" / "eastmoney_executor.py"
    if not script.exists():
        return _eastmoney_queue(intents, execute)

    with tempfile.TemporaryDirectory(prefix="em_exec_") as td:
        orders_fp = Path(td) / "orders.json"
        result_fp = Path(td) / "result.json"
        orders_fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        cmd = ["python3", str(script), "--orders-file", str(orders_fp), "--result-file", str(result_fp)]
        if execute:
            cmd.append("--apply")

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0 or not result_fp.exists():
            return {
                "broker": "eastmoney_sim",
                "rollback": False,
                "error": f"executor_failed:{p.stderr[-400:]}",
                "receipts": [],
                "positions_before": {},
                "positions_after": {},
            }
        result = json.loads(result_fp.read_text(encoding="utf-8"))

    # 失败补偿单（回滚队列）
    rb = result.get("rollback_orders", []) or []
    if rb:
        with _rollback_file().open("a", encoding="utf-8") as f:
            for r in rb:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "broker": "eastmoney_sim",
        "rollback": bool(rb),
        "error": result.get("error", ""),
        "receipts": result.get("receipts", []),
        "positions_before": {},
        "positions_after": {},
        "executor_ok": bool(result.get("ok", False)),
        "rollback_count": len(rb),
    }


def execute_plan(plan: List[dict], execute: bool, broker: str) -> Dict[str, object]:
    if execute and _kill_file().exists():
        return {
            "blocked": True,
            "block_reason": f"kill_switch_exists:{_kill_file()}",
            "intents": [],
            "receipts": [],
        }

    if broker == "eastmoney_paper":
        current_positions = _load_state()
    elif broker == "eastmoney_sim":
        current_positions, _ = _get_eastmoney_positions_pct()
    else:
        current_positions = {}
    intents, risk_marks = build_order_intents(plan, current_positions)

    max_daily = int(cfg.get("risk_control", {}).get("max_daily_trades", 5))
    today_n = _daily_trade_count(broker)
    allowed_n = max(0, max_daily - today_n)
    intents = intents[:allowed_n]

    seen = _load_idem_today(broker)
    final_intents = []
    for it in intents:
        key = _idem_key(it, broker)
        if key in seen:
            continue
        setattr(it, "_idem_key", key)
        final_intents.append(it)

    if broker == "eastmoney_sim":
        result = _eastmoney_api_execute(final_intents, execute=execute)
    else:
        result = _paper_execute(final_intents, execute=execute)

    if execute:
        receipts = result.get("receipts", []) if isinstance(result, dict) else []
        accepted_pairs = set()
        for r in receipts:
            st = str(r.get("status", "")).lower()
            if broker == "eastmoney_paper" and st == "filled":
                accepted_pairs.add((str(r.get("symbol", "")), str(r.get("side", "")).lower()))
            if broker == "eastmoney_sim" and st == "submitted":
                accepted_pairs.add((str(r.get("symbol", "")), str(r.get("side", "")).lower()))
        # Build a status map per (symbol, side)
        receipt_map: Dict[tuple, dict] = {}
        for r in receipts:
            key = (str(r.get("symbol", "")), str(r.get("side", "")).lower())
            receipt_map[key] = r

        for it in final_intents:
            pair = (it.symbol, it.side)
            matched = receipt_map.get(pair, {})
            st = str(matched.get("status", "")).lower()
            if pair in accepted_pairs:
                _append_idem(getattr(it, "_idem_key"), it, "accepted", broker=broker, receipt=matched)
            elif st == "failed":
                _append_idem(getattr(it, "_idem_key"), it, "failed", broker=broker, receipt=matched)

    result.update(
        {
            "blocked": False,
            "risk_marks": risk_marks,
            "intents": [asdict(x) for x in final_intents],
            "daily_trade_cap": max_daily,
            "daily_trade_used_before": today_n,
            "daily_trade_allowed_this_run": allowed_n,
            "kill_switch_file": str(_kill_file()),
        }
    )
    _append_trade_records(result.get("receipts", []), broker=broker, execute=execute)
    return result

