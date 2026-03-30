#!/usr/bin/env python3
"""验证 StockRankSkill：取一天的 top50 候选 → 精排 → 检查输出。"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from _config import cfg, PROJECT_ROOT
from qwen_skills import call_skill_stock_rank
from backtest_signals import compute_score, detect_regime


def load_candidates(n_files=80, top_n=50):
    """从 advanced JSONL 加载多只股票最后一天数据，评分取 top_n。"""
    data_dir = os.path.join(PROJECT_ROOT, cfg["data"]["ashare_dir"], "advanced")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jsonl"))[:n_files]

    candidates = []
    regime_used = None

    for fname in files:
        path = os.path.join(data_dir, fname)
        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
        if len(rows) < 120:
            continue

        idx = len(rows) - 1
        row = rows[idx]
        prev = rows[idx - 1]

        # detect regime once
        if regime_used is None:
            regime, ma120, slope, detail = detect_regime(rows, idx)
            regime_used = regime
            regime_info = (regime, ma120, slope, detail)

        score, _ = compute_score(row, rows, idx, prev, regime_info=regime_info)
        sym = row.get("symbol", fname.replace(".jsonl", ""))

        close_20 = rows[idx - 20]["close"] if idx >= 20 else row["close"]
        trend_20d = round((row["close"] - close_20) / close_20 * 100, 2) if close_20 > 0 else 0
        vol_ma5 = row.get("volume_ma_5", 1) or 1

        candidates.append({
            "symbol": sym,
            "score": round(score, 2),
            "rsi_14": round(row.get("rsi_14", 50), 1),
            "trend_20d": trend_20d,
            "bb_position": round(row.get("bb_position", 0.5), 3),
            "hv_20": round(row.get("hv_20", 0) or 0, 1),
            "vol_ratio": round(row["volume"] / vol_ma5, 2),
            "turnover_rate": round(row.get("turnover_rate", 0) or 0, 3),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:top_n]
    print(f"候选数: {len(candidates)}, 分数范围: {candidates[-1]['score']:.1f} ~ {candidates[0]['score']:.1f}")
    return regime_used, candidates


def main():
    print("加载候选数据...")
    regime, candidates = load_candidates()
    if not candidates:
        print("无法加载候选数据")
        sys.exit(1)

    print(f"市场环境: {regime}")
    print(f"\n调用 StockRankSkill 精排 top{len(candidates)} → top10 ...\n")

    rankings = call_skill_stock_rank(regime, candidates)
    if rankings is None:
        print("精排失败!")
        sys.exit(1)

    print(f"精排结果 ({len(rankings)} 只):")
    print("-" * 70)
    for r in rankings:
        risk = ", ".join(r.get("risk_factors", []))
        print(f"  #{r['rank']:2d} {r['symbol']:>10s}  score={r['score']:5.1f}  "
              f"action={r['action']:10s}  risk=[{risk}]")
        print(f"       reason: {r['reason']}")
    print("-" * 70)

    # 确定性验证
    print("\n确定性验证（第二次调用）...")
    rankings2 = call_skill_stock_rank(regime, candidates)
    if rankings2:
        match = all(r1["symbol"] == r2["symbol"] and r1["rank"] == r2["rank"]
                    for r1, r2 in zip(rankings, rankings2))
        print(f"两次结果一致: {'YES' if match else 'NO (非确定性!)'}")
    else:
        print("第二次调用失败")

    print("\n完整 JSON:")
    print(json.dumps({"rankings": rankings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
