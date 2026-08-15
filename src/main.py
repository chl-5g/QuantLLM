"""QuantLLM — A股量化多 Agent 交易系统

用法:
    python -m src.main --ticker 600519,000858,300750
    python -m src.main --ticker 600519 --start-date 2025-01-01 --end-date 2025-06-30 --show-reasoning
    python -m src.main --ticker 600519,000858 --ollama
"""

import argparse
import json
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.workflow import run_quant_llm
from src.utils.analysts import get_analysts_list


def print_results(result: dict):
    """格式化打印结果"""
    decisions = result.get("decisions", {})
    signals = result.get("analyst_signals", {})

    print("\n" + "=" * 60)
    print("  QuantLLM 多 Agent 交易决策")
    print("=" * 60)

    for ticker, decision in decisions.items():
        action = decision.get("action", "?")
        qty = decision.get("quantity", 0)
        conf = decision.get("confidence", 0)
        reason = decision.get("reasoning", "")

        emoji = {"buy": "🟢", "sell": "🔴", "hold": "⚪"}.get(action, "❓")
        print(f"\n  {emoji} {ticker}: {action.upper()} {qty}股 (置信度{conf}%)")
        print(f"     {reason}")

    print("\n" + "-" * 60)
    print("  各分析师信号汇总:")
    print("-" * 60)

    for agent_name, sigs in signals.items():
        if agent_name.startswith("risk_management"):
            continue
        if agent_name == "regime_analyst_agent":
            regime = sigs.get("regime", "?")
            score = sigs.get("regime_score", "?")
            print(f"  📊 市场环境: {regime} (分数:{score})")
            continue

        if isinstance(sigs, dict):
            for t, s in sigs.items():
                if isinstance(s, dict) and "signal" in s:
                    print(f"  {agent_name:30s} | {t:10s} | {s.get('signal','?'):8s} | 置信度:{s.get('confidence','?')}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="QuantLLM — A股量化多 Agent 交易系统")
    parser.add_argument("--ticker", type=str, required=True,
                        help="股票代码，逗号分隔 (如 600519,000858)")
    parser.add_argument("--start-date", type=str, default=None,
                        help="开始日期 (YYYY-MM-DD)，默认60天前")
    parser.add_argument("--end-date", type=str, default=None,
                        help="结束日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument("--cash", type=float, default=100000.0,
                        help="初始资金 (默认10万)")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="显示每个Agent的详细推理过程")
    parser.add_argument("--ollama", action="store_true",
                        help="使用本地 ollama (默认)")
    parser.add_argument("--model", type=str, default=os.environ.get("OLLAMA_GENERATION_MODEL", "qwen3.8:27b"),
                        help="模型名称 (默认 .env 配置，当前 qwen3.8:27b)")
    parser.add_argument("--analysts", type=str, default=None,
                        help="指定分析师 (逗号分隔)，默认全部")
    parser.add_argument("--list-analysts", action="store_true",
                        help="列出所有分析师")

    args = parser.parse_args()

    if args.list_analysts:
        print("\n可用分析师:")
        for a in get_analysts_list():
            print(f"  {a['key']:30s} — {a['display_name']} ({a['description']})")
        return

    # 处理参数
    tickers = [t.strip() for t in args.ticker.split(",")]
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        from datetime import timedelta
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    selected_analysts = None
    if args.analysts:
        selected_analysts = [a.strip() for a in args.analysts.split(",")]

    provider = "ollama" if args.ollama else "openai"

    print(f"\n{'~' * 50}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  日期: {start_date} ~ {end_date}")
    print(f"  初始资金: {args.cash:,.0f}")
    print(f"  模型: {args.model} ({provider})")
    if selected_analysts:
        print(f"  分析师: {', '.join(selected_analysts)}")
    else:
        print(f"  分析师: 全部 {len(get_analysts_list())} 个")
    print(f"{'~' * 50}")

    result = run_quant_llm(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_cash=args.cash,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=args.model,
        model_provider=provider,
    )

    print_results(result)

    # 保存结果
    out_dir = "/opt/quant-llm/logs"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"agent_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
