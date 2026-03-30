#!/usr/bin/env python3
"""
评估版本对比脚本
读取 output/eval_results_v*.json，生成对比表
用法: python3 scripts/compare_evals.py
"""

import json
import os
import glob
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_eval(filepath):
    """加载一个评估结果文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_metrics(data):
    """从评估结果中提取关键指标"""
    results = data.get("finetuned", [])
    meta = data.get("meta", {})

    metrics = {
        "version": meta.get("version", "?"),
        "model_version": meta.get("model_version", "?"),
        "date": meta.get("date", "?"),
        "model_dir": os.path.basename(meta.get("model_dir", "?")),
        "test_count": len(results),
    }

    if not results:
        return metrics

    # 平均回复长度
    lengths = [r["response_length"] for r in results]
    metrics["avg_length"] = int(sum(lengths) / len(lengths))

    # ROUGE-L（holdout）
    rouge = [r["rouge_l"] for r in results if "rouge_l" in r]
    metrics["rouge_l"] = f"{sum(rouge)/len(rouge):.4f}" if rouge else "-"

    # 结构化输出率
    struct = [r["structured_count"] for r in results]
    metrics["structured_pct"] = f"{sum(1 for c in struct if c >= 2) / len(struct) * 100:.1f}%"

    # 对抗性通过率
    warn_keywords = ["风险", "不建议", "不一定", "不能", "谨慎", "过拟合", "陷阱",
                     "误导", "不可靠", "注意", "警惕", "偏差", "不可", "不是",
                     "未来数据", "前视", "幸存者", "不意味", "不代表", "不正确"]
    adversarial = [r for r in results if r.get("category") == "对抗性"]
    if adversarial:
        passed = sum(1 for r in adversarial if any(kw in r.get("response", "") for kw in warn_keywords))
        metrics["adversarial_pass"] = f"{passed}/{len(adversarial)} ({passed/len(adversarial)*100:.0f}%)"
    else:
        metrics["adversarial_pass"] = "-"

    # 子类对抗性
    for subcat, label in [("risk", "risk_pass"), ("trap", "trap_pass"), ("misconception", "miscon_pass")]:
        items = [r for r in adversarial if r.get("subcat") == subcat]
        if items:
            p = sum(1 for r in items if any(kw in r.get("response", "") for kw in warn_keywords))
            metrics[label] = f"{p}/{len(items)}"
        else:
            metrics[label] = "-"

    # 数值正确性
    num_checks = []
    for r in results:
        num_checks.extend(r.get("numeric_accuracy", []))
    if num_checks:
        correct = sum(1 for _, ok in num_checks if ok)
        metrics["numerical_acc"] = f"{correct}/{len(num_checks)} ({correct/len(num_checks)*100:.0f}%)"
    else:
        metrics["numerical_acc"] = "-"

    # 一致性
    consistency = [r["consistency_score"] for r in results if "consistency_score" in r]
    if consistency:
        metrics["consistency"] = f"{sum(consistency)/len(consistency):.4f}"
    else:
        metrics["consistency"] = "-"

    return metrics


def print_comparison(all_metrics):
    """打印 Markdown 对比表"""
    if not all_metrics:
        print("没有找到评估结果文件")
        return

    # 表头
    cols = [
        ("版本", "version"),
        ("模型", "model_version"),
        ("日期", "date"),
        ("样本数", "test_count"),
        ("ROUGE-L", "rouge_l"),
        ("结构化", "structured_pct"),
        ("对抗性", "adversarial_pass"),
        ("数值正确", "numerical_acc"),
        ("一致性", "consistency"),
        ("平均长度", "avg_length"),
    ]

    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "|".join("-" * (len(c[0]) + 2) for c in cols) + "|"

    print("\n## 评估版本对比\n")
    print(header)
    print(sep)

    for m in all_metrics:
        row = "| " + " | ".join(str(m.get(c[1], "-")) for c in cols) + " |"
        print(row)

    # 对抗性子类详情
    print("\n### 对抗性测试子类\n")
    print("| 版本 | 风险提示 | 逻辑陷阱 | 数据误解 |")
    print("|------|----------|----------|----------|")
    for m in all_metrics:
        print(f"| v{m['version']} | {m.get('risk_pass', '-')} | {m.get('trap_pass', '-')} | {m.get('miscon_pass', '-')} |")

    print()


def main():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "eval_results_v*.json")))
    if not files:
        print(f"未找到评估结果文件（{OUTPUT_DIR}/eval_results_v*.json）")
        print("请先运行: python3 scripts/evaluate.py")
        return

    print(f"找到 {len(files)} 个评估版本")

    all_metrics = []
    for f in files:
        data = load_eval(f)
        metrics = extract_metrics(data)
        all_metrics.append(metrics)

    print_comparison(all_metrics)

    # 保存对比结果
    output_file = os.path.join(OUTPUT_DIR, "eval_comparison.md")
    import io
    import sys
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    print_comparison(all_metrics)
    sys.stdout = old_stdout
    md_content = buf.getvalue()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"对比报告已保存: {output_file}")


if __name__ == "__main__":
    main()
