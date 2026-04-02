#!/usr/bin/env python3
"""交易守护监控与告警（Webhook/控制台）。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

import requests

from _config import PROJECT_ROOT


def _read_jsonl_tail(fp: Path, limit: int = 200) -> List[dict]:
    if not fp.exists():
        return []
    lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _check_consecutive_failures(recs: List[dict], n: int) -> bool:
    fails = 0
    for r in reversed(recs):
        st = str(r.get("status", "")).lower()
        if st in {"failed"}:
            fails += 1
            if fails >= n:
                return True
        else:
            fails = 0
    return False


def _send_webhook(url: str, text: str) -> None:
    if not url.strip():
        return
    try:
        requests.post(url, json={"msgtype": "text", "text": {"content": text}}, timeout=8)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor and alert for trading runtime")
    parser.add_argument("--log-dir", default="output/trade_logs", help="交易日志目录")
    parser.add_argument("--max-idle-min", type=int, default=30, help="超过 N 分钟无新 trade 文件告警")
    parser.add_argument("--fail-threshold", type=int, default=5, help="连续失败 N 次告警")
    parser.add_argument("--webhook", default="", help="企业微信/飞书机器人 webhook")
    args = parser.parse_args()

    log_dir = Path(PROJECT_ROOT) / args.log_dir
    now = datetime.now().timestamp()
    trade_files = sorted(log_dir.glob("trade_*.json"))
    alert_msgs = []

    if not trade_files:
        alert_msgs.append("未发现 trade_*.json 日志文件")
    else:
        last = trade_files[-1]
        idle_min = (now - last.stat().st_mtime) / 60.0
        if idle_min > args.max_idle_min:
            alert_msgs.append(f"交易日志超过 {args.max_idle_min} 分钟未更新（当前 {idle_min:.1f} 分钟）")

    recs = _read_jsonl_tail(log_dir / "executed_orders.jsonl", limit=400)
    if _check_consecutive_failures(recs, args.fail_threshold):
        alert_msgs.append(f"出现连续失败订单（>= {args.fail_threshold}）")

    if alert_msgs:
        msg = "[QuantLLM 监控告警]\n" + "\n".join(f"- {x}" for x in alert_msgs)
        print(msg)
        _send_webhook(args.webhook, msg)
        raise SystemExit(2)

    print("[MONITOR] OK")


if __name__ == "__main__":
    main()
