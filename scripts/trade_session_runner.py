#!/usr/bin/env python3
"""交易时段调度器：仅在 A 股时段运行 trade_live_qwen 并归档日志。"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from _config import cfg  # 加载 .env（ollama 模型配置）

try:
    from chinese_calendar import is_workday as _cc_is_workday
except Exception:  # optional dependency fallback
    _cc_is_workday = None


BJ_TZ = ZoneInfo("Asia/Shanghai")
PROJECT_ROOT = Path("/opt/quant-llm")
TRADE_SCRIPT = PROJECT_ROOT / "scripts" / "trade_live_qwen.py"
EXECUTED_ORDERS = PROJECT_ROOT / "output" / "trade_logs" / "executed_orders.jsonl"
TRADE_LOG_RE = re.compile(r"日志文件:\s*(/opt/quant-llm/output/trade_logs/trade_\d{8}_\d{6}\.json)")
PLAN_DIR = PROJECT_ROOT / "output" / "trade_logs" / "plans"


STOP = False


@dataclass
class SessionState:
    last_run_at: Optional[datetime] = None
    plan_date: str = ""


def _sig_handler(signum, frame):
    del signum, frame
    global STOP
    STOP = True


def _resolve_log_dir(user_input: str | None) -> Path:
    if user_input:
        p = Path(user_input).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    env_p = os.getenv("STOCK_LOG_DIR", "").strip()
    if env_p:
        p = Path(env_p).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    candidates = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "output" / "logs",
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            return c
        except Exception:
            continue
    raise RuntimeError("无法创建日志目录，请通过 --log-dir 指定可写路径")


def _session_mode(now_bj: datetime) -> str:
    """
    返回当前时段模式：
    - preopen: 09:00-09:25（仅 dry-run）
    - live: 09:30-11:30, 13:00-15:00（允许 execute）
    - closed: 其它时段
    """
    if not _is_cn_workday(now_bj):
        return "closed"
    hm = now_bj.hour * 60 + now_bj.minute
    if (9 * 60 + 0) <= hm < (9 * 60 + 25):
        return "preopen"
    if (9 * 60 + 30) <= hm < (11 * 60 + 30) or (13 * 60) <= hm < (15 * 60):
        return "live"
    return "closed"


def _is_cn_workday(now_bj: datetime) -> bool:
    """
    优先使用 chinese_calendar（支持中国法定节假日与调休）。
    若依赖不可用，回退为周一到周五。
    """
    if _cc_is_workday is not None:
        try:
            return bool(_cc_is_workday(now_bj.date()))
        except Exception:
            pass
    # 兜底：若当前解释器未安装 chinese_calendar，尝试调用项目 venv。
    try:
        cmd = [
            "/opt/quant-llm/finetune-env/bin/python3",
            "-c",
            (
                "import datetime as d; "
                "from chinese_calendar import is_workday; "
                f"print('1' if is_workday(d.date({now_bj.year},{now_bj.month},{now_bj.day})) else '0')"
            ),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if p.returncode == 0 and p.stdout.strip() in {"0", "1"}:
            return p.stdout.strip() == "1"
    except Exception:
        pass
    return now_bj.weekday() < 5


def _next_session_time(now_bj: datetime) -> datetime:
    base = now_bj.replace(second=0, microsecond=0)
    for add_days in range(0, 8):
        d = base + timedelta(days=add_days)
        if not _is_cn_workday(d):
            continue
        p_start = d.replace(hour=9, minute=0, second=0, microsecond=0)
        p_end = d.replace(hour=9, minute=25, second=0, microsecond=0)
        m_start = d.replace(hour=9, minute=30, second=0, microsecond=0)
        m_end = d.replace(hour=11, minute=30, second=0, microsecond=0)
        a_start = d.replace(hour=13, minute=0, second=0, microsecond=0)
        a_end = d.replace(hour=15, minute=0, second=0, microsecond=0)
        if now_bj < p_start:
            return p_start
        if p_start <= now_bj < p_end:
            return now_bj
        if now_bj < m_start:
            return m_start
        if m_start <= now_bj < m_end:
            return now_bj
        if m_end <= now_bj < a_start:
            return a_start
        if a_start <= now_bj < a_end:
            return now_bj
    return base + timedelta(days=1)


def _write_text(fp: Path, text: str) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(text, encoding="utf-8")


def _copy_if_exists(src: Path, dst_dir: Path, dst_name: str | None = None) -> Optional[Path]:
    if not src.exists():
        return None
    dst = dst_dir / (dst_name or src.name)
    shutil.copy2(src, dst)
    return dst


def _daily_dir(base_dir: Path, now_bj: datetime) -> Path:
    day_dir = base_dir / now_bj.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def _plan_file_for_date(now_bj: datetime) -> Path:
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    return PLAN_DIR / f"signal_plan_{now_bj.strftime('%Y%m%d')}.json"


def _safe_name_fragment(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:40]


def _build_trade_tag(trade_fp: Path) -> str:
    try:
        data = json.loads(trade_fp.read_text(encoding="utf-8"))
        receipts = data.get("execution_result", {}).get("receipts", []) or []
        for r in receipts:
            code = str(r.get("stock_code", "")).strip() or str(r.get("symbol", ""))[-6:]
            name = _safe_name_fragment(str(r.get("stock_name", "")).strip())
            if code and name:
                return f"{code}_{name}"
            if code:
                return code
        plan = data.get("execution_plan", []) or []
        if plan:
            sym = str(plan[0].get("symbol", "")).strip()
            code = sym[-6:] if len(sym) >= 6 else sym
            return _safe_name_fragment(code)
    except Exception:
        return ""
    return ""


def _run_once(
    log_dir: Path,
    broker: str,
    execute: bool,
    model_override: str = "",
    signal_only: bool = False,
    plan_file: Optional[Path] = None,
    use_plan_file: Optional[Path] = None,
) -> int:
    now_bj = datetime.now(BJ_TZ)
    ts = now_bj.strftime("%Y%m%d_%H%M%S")
    day_dir = _daily_dir(log_dir, now_bj)
    cmd = ["python3", str(TRADE_SCRIPT), "--broker", broker]
    if signal_only:
        cmd.insert(2, "--signal-only")
    if plan_file is not None:
        cmd.extend(["--plan-file", str(plan_file)])
    if use_plan_file is not None:
        cmd.extend(["--use-plan-file", str(use_plan_file)])
    if execute:
        cmd.insert(2, "--execute")
    env = os.environ.copy()
    if model_override:
        env["STOCK_RANK_MODEL"] = model_override
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), env=env)
    stdout = p.stdout or ""
    stderr = p.stderr or ""
    full = (
        f"[time_bj] {datetime.now(BJ_TZ).isoformat(timespec='seconds')}\n"
        f"[cmd] {' '.join(cmd)}\n"
        f"[stock_rank_model] {model_override or '<default>'}\n"
        f"[return_code] {p.returncode}\n"
        "----- stdout -----\n"
        f"{stdout}\n"
        "----- stderr -----\n"
        f"{stderr}\n"
    )
    strategy_fp = day_dir / f"strategy_{ts}.log"
    _write_text(strategy_fp, full)

    m = TRADE_LOG_RE.search(stdout + "\n" + stderr)
    if m:
        trade_src = Path(m.group(1))
        copied = _copy_if_exists(trade_src, day_dir)
        if copied is not None:
            tag = _build_trade_tag(copied)
            if tag:
                tagged_trade = day_dir / f"{copied.stem}_{tag}{copied.suffix}"
                try:
                    copied.rename(tagged_trade)
                    copied = tagged_trade
                except Exception:
                    pass
                tagged_strategy = day_dir / f"strategy_{ts}_{tag}.log"
                try:
                    strategy_fp.rename(tagged_strategy)
                    strategy_fp = tagged_strategy
                except Exception:
                    pass
    _copy_if_exists(EXECUTED_ORDERS, day_dir, dst_name=f"executed_orders_{ts}.jsonl")

    print(f"[RUN] rc={p.returncode} model={model_override or '<default>'} strategy_log={strategy_fp}")
    return p.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="A股交易时段自动执行器")
    parser.add_argument("--log-dir", default=None, help="日志输出目录，默认 /opt/quant-llm/logs")
    parser.add_argument("--interval-sec", type=int, default=300, help="交易时段内执行间隔（秒）")
    parser.add_argument("--broker", default="eastmoney_sim", help="执行适配器，默认 eastmoney_sim")
    parser.add_argument("--dry-run", action="store_true", help="仅策略模拟，不带 --execute")
    parser.add_argument("--once", action="store_true", help="仅运行一次后退出")
    parser.add_argument("--force-run", action="store_true", help="忽略交易时段限制（用于测试）")
    parser.add_argument("--force-mode", choices=["preopen", "live"], default=None, help="强制时段模式（用于测试）")
    parser.add_argument("--preopen-model", default=cfg["ollama"]["generation_model"], help="09:15-09:25 预热时段模型（.env 配置）")
    parser.add_argument("--live-model", default=cfg["ollama"]["live_rank_model"], help="09:30后执行时段模型（.env 配置）")
    args = parser.parse_args()

    if args.interval_sec < 30:
        raise SystemExit("interval-sec 不能小于 30 秒")

    log_dir = _resolve_log_dir(args.log_dir)
    state = SessionState()

    print(f"[INIT] log_dir={log_dir}")
    print(f"[INIT] broker={args.broker} interval={args.interval_sec}s")
    print("[INIT] policy: workday 09:00-09:25 生成日计划+周度3-10只买入清单 | 09:30-11:30/13:00-15:00 按计划执行")
    print(f"[INIT] models: preopen={args.preopen_model} live={args.live_model}")

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    while not STOP:
        now_bj = datetime.now(BJ_TZ)
        mode = args.force_mode or ("live" if args.force_run else _session_mode(now_bj))
        if mode in {"preopen", "live"}:
            can_run = (
                state.last_run_at is None
                or (now_bj - state.last_run_at).total_seconds() >= args.interval_sec
            )
            if can_run:
                day_key = now_bj.strftime("%Y%m%d")
                day_plan = _plan_file_for_date(now_bj)
                if mode == "preopen":
                    # 预热阶段只生成一次当日信号计划，避免频繁重算。
                    if state.plan_date != day_key or not day_plan.exists():
                        print(
                            f"[MODE] {mode} now={now_bj.strftime('%Y-%m-%d %H:%M:%S')} "
                            f"signal_only=True model={args.preopen_model} plan={day_plan}"
                        )
                        _run_once(
                            log_dir=log_dir,
                            broker=args.broker,
                            execute=False,
                            model_override=args.preopen_model,
                            signal_only=True,
                            plan_file=day_plan,
                        )
                        state.plan_date = day_key
                    else:
                        print(f"[MODE] {mode} 复用当日计划: {day_plan}")
                else:
                    execute_now = not args.dry_run
                    # 若无预热计划，则在 live 首轮补生成（同日仅一次）。
                    if not day_plan.exists():
                        print(f"[MODE] live 缺失计划，先补生成: {day_plan}")
                        _run_once(
                            log_dir=log_dir,
                            broker=args.broker,
                            execute=False,
                            model_override=args.live_model,
                            signal_only=True,
                            plan_file=day_plan,
                        )
                        state.plan_date = day_key
                    print(
                        f"[MODE] {mode} now={now_bj.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"execute={execute_now} use_plan={day_plan} model={args.live_model}"
                    )
                    _run_once(
                        log_dir=log_dir,
                        broker=args.broker,
                        execute=execute_now,
                        model_override=args.live_model,
                        use_plan_file=day_plan,
                    )
                state.last_run_at = now_bj
                if args.once:
                    break
            sleep_s = min(10, max(1, args.interval_sec // 10))
            time.sleep(sleep_s)
            continue

        next_t = _next_session_time(now_bj)
        wait_s = int((next_t - now_bj).total_seconds())
        print(
            f"[IDLE] 非交易时段 now={now_bj.strftime('%Y-%m-%d %H:%M:%S')} "
            f"next={next_t.strftime('%Y-%m-%d %H:%M:%S')} wait={max(wait_s, 1)}s"
        )
        if args.once:
            break
        time.sleep(min(max(wait_s, 1), 60))

    print("[EXIT] trade_session_runner stopped")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        raise
