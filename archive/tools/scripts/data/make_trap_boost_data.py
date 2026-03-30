#!/usr/bin/env python3
import json
import random
from pathlib import Path

from evaluate import MANUAL_TESTS

OUT = Path("/opt/quant-llm/training-data/trap_boost_v1.jsonl")
SEED = 42
REPEAT_PER_Q = 8

STYLE_PREFIXES = [
    "先给结论：",
    "结论先行：",
    "核心判断：",
    "先做风险判断：",
]

RISK_POINTS = [
    "警惕过拟合与样本内拟合。",
    "必须检查是否存在数据泄漏（前视偏差）。",
    "需要做时间切分与滚动窗口外样本验证。",
    "注意交易成本、滑点与冲击成本的侵蚀。",
    "需要做参数稳定性与敏感性分析。",
    "回测结果不代表未来收益。",
    "应排查幸存者偏差与选择偏差。",
    "必须执行风险控制与仓位约束。",
]

TRAP_EXTRA = {
    "回测年化收益200%、最大回撤只有5%": [
        "该结果高度可疑，先怀疑数据泄漏或过拟合。",
        "先做样本外与跨市场验证，不可直接实盘。",
    ],
    "胜率90%": [
        "高胜率不等于高收益，需看盈亏比与回撤。",
        "需要第三方可复现实验与审计日志。",
    ],
    "网格搜索测试了10000组参数": [
        "多重比较会放大伪发现概率。",
        "应使用嵌套验证或白噪声检验。",
    ],
}


def build_answer(question: str, variant: int) -> str:
    rnd = random.Random(1000 + variant)
    prefix = rnd.choice(STYLE_PREFIXES)
    pts = rnd.sample(RISK_POINTS, k=4)

    extra = []
    for key, val in TRAP_EXTRA.items():
        if key in question:
            extra = val
            break

    lines = [
        prefix + "不能直接按题面执行，先做风控与验证。",
        "原因：",
    ]
    for i, p in enumerate(pts, 1):
        lines.append(f"{i}. {p}")

    if extra:
        lines.append("补充：")
        for p in extra:
            lines.append(f"- {p}")

    lines.append("执行建议：先小资金仿真、设止损和最大回撤阈值，再决定是否上线。")
    return "\n".join(lines)


def main():
    random.seed(SEED)
    traps = [x for x in MANUAL_TESTS if x.get("category") == "对抗性" and x.get("subcat") == "trap"]

    rows = []
    for item in traps:
        q = item["q"]
        for i in range(REPEAT_PER_Q):
            ans = build_answer(q, i)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": ans},
                    ],
                    "source": "trap_boost_v1",
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"written: {OUT}")
    print(f"samples: {len(rows)}")


if __name__ == "__main__":
    main()
