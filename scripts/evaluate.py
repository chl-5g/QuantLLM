#!/usr/bin/env python3
"""
模型评估脚本
评估微调后的模型质量，支持与基座模型对比
输出：output/eval_results.json
"""

import json
import os
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_ROOT, "training-data", "merged_train_v2.jsonl")
HOLDOUT_FILE = os.path.join(PROJECT_ROOT, "training-data", "eval_holdout.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "eval_results.json")
MODEL_DIR = os.path.join(OUTPUT_DIR, "quant-qwen2.5-14b-lora")
BASE_MODEL = "unsloth/Qwen2.5-14B-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# ============================================================
# 手写测试题（50 条）
# ============================================================

MANUAL_TESTS = [
    # 技术分析（10 条）
    {"category": "技术分析", "q": "请解释RSI指标的计算方法和使用技巧。RSI超买超卖的阈值一般设为多少？"},
    {"category": "技术分析", "q": "MACD指标由哪三部分组成？如何利用MACD金叉死叉进行交易？"},
    {"category": "技术分析", "q": "布林带的上轨、中轨、下轨分别代表什么？布林带收口和开口有什么含义？"},
    {"category": "技术分析", "q": "什么是KDJ指标？它与RSI有什么区别和联系？"},
    {"category": "技术分析", "q": "请解释均线系统中的'多头排列'和'空头排列'，以及如何利用均线交叉判断趋势。"},
    {"category": "技术分析", "q": "什么是成交量-价格背离（量价背离）？在上涨趋势和下跌趋势中分别意味着什么？"},
    {"category": "技术分析", "q": "请解释支撑位和压力位的概念，以及如何利用它们设置止损止盈。"},
    {"category": "技术分析", "q": "什么是头肩顶和头肩底形态？如何判断这些形态的有效性？"},
    {"category": "技术分析", "q": "请解释OBV（能量潮）指标的原理和用法。"},
    {"category": "技术分析", "q": "什么是ATR（平均真实波幅）指标？如何用它来设定动态止损？"},

    # 策略代码（10 条）
    {"category": "策略代码", "q": "请用Python实现一个简单的双均线交叉策略，包含买入卖出逻辑。"},
    {"category": "策略代码", "q": "如何用Python计算一个股票组合的夏普比率？请给出完整代码。"},
    {"category": "策略代码", "q": "请用Python实现一个海龟交易策略的核心逻辑（突破通道+ATR止损）。"},
    {"category": "策略代码", "q": "如何用pandas计算股票的RSI指标？请写出完整的函数。"},
    {"category": "策略代码", "q": "请用Python实现一个简单的网格交易策略框架。"},
    {"category": "策略代码", "q": "如何用Python实现一个基于因子的选股模型？请用市盈率和ROE作为示例因子。"},
    {"category": "策略代码", "q": "请用Python实现一个简单的回测框架，包含净值计算和绩效指标。"},
    {"category": "策略代码", "q": "如何用Python实现配对交易策略？请包含协整检验和交易信号生成。"},
    {"category": "策略代码", "q": "请用Python实现一个基于布林带的均值回复策略。"},
    {"category": "策略代码", "q": "如何用Python计算投资组合的最大回撤？请给出高效的实现。"},

    # 量化计算（10 条）
    {"category": "量化计算", "q": "一个策略年化收益20%，年化波动率15%，无风险利率3%。请计算夏普比率和索提诺比率（假设下行波动率为10%）。"},
    {"category": "量化计算", "q": "某组合持有A（权重60%，波动率20%）和B（权重40%，波动率30%），相关系数0.5。求组合波动率。"},
    {"category": "量化计算", "q": "用Black-Scholes公式计算：标的50元，行权价55元，无风险5%，波动率25%，到期6个月的欧式看涨期权价格。"},
    {"category": "量化计算", "q": "某策略交易200次，胜率55%，平均盈利3%，平均亏损2%。计算期望收益和凯利公式最优仓位。"},
    {"category": "量化计算", "q": "一个组合净值从100涨到180，再跌到120，然后涨到200。请计算最大回撤和卡玛比率。"},
    {"category": "量化计算", "q": "某因子月度IC序列均值为0.04，标准差0.025。计算ICIR和年化ICIR，并判断该因子是否显著。"},
    {"category": "量化计算", "q": "一个3年期债券，面值100，票面利率4%年付，市场收益率3.5%。计算债券价格和修正久期。"},
    {"category": "量化计算", "q": "组合日收益率均值0.08%，标准差1.5%，求95%和99%置信度下的参数法VaR。"},
    {"category": "量化计算", "q": "两个资产预期收益10%和6%，波动率25%和10%，相关系数-0.2。求最小方差组合权重。"},
    {"category": "量化计算", "q": "某CTA策略的Hurst指数为0.65，这对策略选择有什么指导意义？"},

    # 风控（10 条）
    {"category": "风控", "q": "请解释VaR和CVaR的区别，以及为什么CVaR被认为是更好的风险度量？"},
    {"category": "风控", "q": "什么是尾部风险？如何在量化投资中度量和管理尾部风险？"},
    {"category": "风控", "q": "请解释仓位管理中的凯利公式，以及实际使用中为什么通常用半凯利？"},
    {"category": "风控", "q": "什么是最大回撤？如何设定合理的最大回撤止损线？"},
    {"category": "风控", "q": "请解释压力测试在风险管理中的作用，给出3个A股市场的典型压力测试场景。"},
    {"category": "风控", "q": "什么是流动性风险？在量化交易中如何度量和控制流动性风险？"},
    {"category": "风控", "q": "请解释相关性崩溃（correlation breakdown）现象，以及它对多元化投资的影响。"},
    {"category": "风控", "q": "如何设计一个量化基金的风控框架？需要监控哪些关键指标？"},
    {"category": "风控", "q": "什么是回撤修复比？如果最大回撤50%，需要多少收益才能回到前高？"},
    {"category": "风控", "q": "请解释杠杆对收益和风险的影响，以及在量化策略中如何确定合适的杠杆水平。"},

    # 可转债/ETF/期货（10 条）
    {"category": "可转债", "q": "什么是可转债的双低策略？双低值如何计算？一般以多少为筛选阈值？"},
    {"category": "可转债", "q": "请解释可转债的转股溢价率和纯债溢价率，它们分别反映了什么？"},
    {"category": "可转债", "q": "可转债下修转股价的条件和流程是什么？对投资者有什么影响？"},
    {"category": "ETF", "q": "什么是ETF套利？请解释折价套利和溢价套利的操作流程。"},
    {"category": "ETF", "q": "如何用ETF构建行业轮动策略？请给出具体的轮动逻辑。"},
    {"category": "ETF", "q": "请比较场内ETF和场外指数基金的优缺点，适合什么类型的投资者？"},
    {"category": "期货", "q": "什么是期货的基差和升贴水？如何利用基差进行期现套利？"},
    {"category": "期货", "q": "请解释商品期货中的仓单和交割逻辑，以及它们对近月合约价格的影响。"},
    {"category": "期货", "q": "什么是CTA策略？请介绍常见的CTA策略类型和它们的收益特征。"},
    {"category": "期货", "q": "请解释商品期货的季节性规律，以及如何在量化交易中利用季节性因子。"},
]


def prepare_holdout(data_file, holdout_file, ratio=0.02, seed=42):
    """从训练数据中抽取 holdout 测试集"""
    if os.path.exists(holdout_file):
        with open(holdout_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        print(f"已有 holdout: {len(records)} 条")
        return records

    records = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    random.seed(seed)
    n = max(1, int(len(records) * ratio))
    holdout = random.sample(records, n)

    with open(holdout_file, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"抽取 holdout: {n} 条 (共 {len(records)} 条)")
    return holdout


def generate_response(model, tokenizer, question, system_prompt=None, max_tokens=512):
    """用模型生成回答"""
    if system_prompt is None:
        system_prompt = "你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。"

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def compute_rouge_l(prediction, reference):
    """计算 ROUGE-L F1"""
    # 简单的 LCS 实现，避免强依赖 rouge-score 库
    pred_tokens = list(prediction)
    ref_tokens = list(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS 长度
    m, n = len(pred_tokens), len(ref_tokens)
    # 优化：只保留两行
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    precision = lcs_len / m if m > 0 else 0
    recall = lcs_len / n if n > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def evaluate_model(model, tokenizer, test_data, label="model"):
    """评估模型"""
    results = []

    for i, item in enumerate(test_data):
        if i % 10 == 0:
            print(f"  [{label}] {i+1}/{len(test_data)}...", flush=True)

        if "q" in item:
            # 手写测试题
            question = item["q"]
            reference = None
            category = item.get("category", "unknown")
        else:
            # holdout 数据
            messages = item.get("messages", [])
            question = next((m["content"] for m in messages if m["role"] == "user"), "")
            reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            category = "holdout"

        response = generate_response(model, tokenizer, question)

        result = {
            "question": question[:100],
            "category": category,
            "response_length": len(response),
            "response": response[:500],
        }

        # ROUGE-L（仅 holdout 有参考答案）
        if reference:
            result["rouge_l"] = compute_rouge_l(response, reference)
            result["reference_length"] = len(reference)

        # 结构化检查
        structured_markers = ["**", "##", "1.", "- ", "```", "指标", "策略", "风险"]
        result["structured_count"] = sum(1 for m in structured_markers if m in response)

        results.append(result)

    return results


def print_summary(results, label="Model"):
    """打印评估摘要"""
    print(f"\n{'='*60}")
    print(f"评估结果: {label}")
    print(f"{'='*60}")

    # 总体统计
    lengths = [r["response_length"] for r in results]
    print(f"  总样本数: {len(results)}")
    print(f"  平均回复长度: {sum(lengths)/len(lengths):.0f} 字符")
    print(f"  中位数长度: {sorted(lengths)[len(lengths)//2]} 字符")

    # ROUGE-L（holdout）
    rouge_scores = [r["rouge_l"] for r in results if "rouge_l" in r]
    if rouge_scores:
        print(f"\n  ROUGE-L (holdout):")
        print(f"    平均: {sum(rouge_scores)/len(rouge_scores):.4f}")
        print(f"    中位数: {sorted(rouge_scores)[len(rouge_scores)//2]:.4f}")

    # 结构化输出率
    struct_counts = [r["structured_count"] for r in results]
    structured_rate = sum(1 for c in struct_counts if c >= 2) / len(struct_counts) * 100
    print(f"\n  结构化输出率: {structured_rate:.1f}%")

    # 分类别统计
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\n  分类别统计:")
    for cat, items in sorted(by_cat.items()):
        avg_len = sum(r["response_length"] for r in items) / len(items)
        rouge = [r["rouge_l"] for r in items if "rouge_l" in r]
        rouge_str = f", ROUGE-L={sum(rouge)/len(rouge):.4f}" if rouge else ""
        print(f"    {cat}: {len(items)}条, 平均长度={avg_len:.0f}{rouge_str}")


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--baseline", action="store_true", help="同时评估基座模型作为对比")
    parser.add_argument("--manual-only", action="store_true", help="仅评估手写测试题")
    parser.add_argument("--holdout-only", action="store_true", help="仅评估 holdout 数据")
    parser.add_argument("--max-holdout", type=int, default=100, help="最多评估的 holdout 条数")
    args = parser.parse_args()

    print("=" * 60)
    print("模型评估")
    print("=" * 60)

    # 准备测试数据
    test_data = []

    if not args.holdout_only:
        test_data.extend(MANUAL_TESTS)
        print(f"手写测试题: {len(MANUAL_TESTS)} 条")

    if not args.manual_only and os.path.exists(DATA_FILE):
        holdout = prepare_holdout(DATA_FILE, HOLDOUT_FILE)
        if len(holdout) > args.max_holdout:
            random.seed(42)
            holdout = random.sample(holdout, args.max_holdout)
        test_data.extend(holdout)
        print(f"Holdout 测试: {len(holdout)} 条")

    print(f"总测试: {len(test_data)} 条")

    # 加载微调模型
    print(f"\n加载微调模型: {MODEL_DIR}")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # 评估微调模型
    print("\n评估微调模型...")
    finetuned_results = evaluate_model(model, tokenizer, test_data, label="finetuned")
    print_summary(finetuned_results, label="微调模型")

    all_results = {"finetuned": finetuned_results}

    # 评估基座模型（可选）
    if args.baseline:
        # 释放微调模型
        del model
        import torch
        torch.cuda.empty_cache()

        print(f"\n加载基座模型: {BASE_MODEL}")
        base_model, base_tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(base_model)

        print("\n评估基座模型...")
        baseline_results = evaluate_model(base_model, base_tokenizer, test_data, label="baseline")
        print_summary(baseline_results, label="基座模型")
        all_results["baseline"] = baseline_results

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
