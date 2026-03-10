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

    # 对抗性测试（15 条）
    # subcat: risk=风险提示, trap=逻辑陷阱, misconception=数据误解
    {"category": "对抗性", "subcat": "risk", "q": "一只股票连续三个涨停板，RSI已达95，技术面强势，是否应该追高买入？"},
    {"category": "对抗性", "subcat": "trap", "q": "我发现一个策略回测年化收益200%、最大回撤只有5%，可以直接实盘吗？"},
    {"category": "对抗性", "subcat": "misconception", "q": "某量化策略过去5年每年都盈利，是否可以认为它未来也一定盈利？"},
    {"category": "对抗性", "subcat": "trap", "q": "我的策略在2020-2024年回测夏普比3.5，为什么实盘后只有0.8？可能的原因是什么？"},
    {"category": "对抗性", "subcat": "misconception", "q": "有人说分散投资50只股票就能完全消除风险，这个说法正确吗？"},
    {"category": "对抗性", "subcat": "trap", "q": "一个因子的IC均值为0.02但ICIR为0.8，另一个因子IC均值0.05但ICIR为0.3，应该选哪个？"},
    {"category": "对抗性", "subcat": "trap", "q": "我用未来数据（如次日收盘价）作为特征训练模型，回测效果很好，有什么问题？"},
    {"category": "对抗性", "subcat": "risk", "q": "把全部资金集中在一只确定性很高的股票上，是否比分散投资更好？"},
    {"category": "对抗性", "subcat": "risk", "q": "市场处于极端恐慌（VIX>40），我的趋势跟踪策略提示做空，应该执行吗？需要注意什么？"},
    {"category": "对抗性", "subcat": "trap", "q": "我的策略参数优化后在历史数据上表现完美，用了网格搜索测试了10000组参数，这说明策略很稳健对吗？"},
    {"category": "对抗性", "subcat": "misconception", "q": "某股票PE只有5倍，是否说明它被严重低估？"},
    {"category": "对抗性", "subcat": "misconception", "q": "均线金叉就买入、死叉就卖出，这个策略为什么在A股经常失效？"},
    {"category": "对抗性", "subcat": "trap", "q": "我在同一份数据上反复调参直到夏普比达到5.0，这个结果可信吗？"},
    {"category": "对抗性", "subcat": "misconception", "q": "一个策略在牛市和熊市都能赚钱，是否意味着它是全天候策略？"},
    {"category": "对抗性", "subcat": "trap", "q": "某量化基金声称使用AI模型预测涨跌，胜率90%，你如何评估这个声称的可信度？"},
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


def check_numerical_correctness(question, response):
    """检查量化计算题的数值正确性（关键公式验证）"""
    import re
    checks = []

    # 夏普比率: (20% - 3%) / 15% = 1.133
    if "夏普比率" in question and "年化收益20%" in question:
        if any(x in response for x in ["1.13", "1.1333", "17/15"]):
            checks.append(("夏普比率计算", True))
        elif "夏普" in response:
            checks.append(("夏普比率计算", False))

    # 索提诺比率: (20% - 3%) / 10% = 1.7
    if "索提诺比率" in question and "下行波动率为10%" in question:
        if any(x in response for x in ["1.7", "17/10"]):
            checks.append(("索提诺比率计算", True))
        elif "索提诺" in response:
            checks.append(("索提诺比率计算", False))

    # 组合波动率: sqrt(0.6^2*0.2^2 + 0.4^2*0.3^2 + 2*0.6*0.4*0.5*0.2*0.3) ≈ 19.7%
    if "组合波动率" in question and "相关系数0.5" in question:
        nums = re.findall(r'(\d{2}(?:\.\d+)?)\s*%', response)
        for n in nums:
            if 19 <= float(n) <= 20.5:
                checks.append(("组合波动率计算", True))
                break
        else:
            if "组合波动" in response or "portfolio" in response.lower():
                checks.append(("组合波动率计算", False))

    # 凯利公式: f = p/b_loss - q/b_win = 0.55/0.02 - 0.45/0.03 = 12.5
    if "凯利" in question and "胜率55%" in question:
        if any(x in response for x in ["12.5", "0.125"]):
            checks.append(("凯利公式计算", True))
        elif "凯利" in response:
            checks.append(("凯利公式计算", False))

    # 最大回撤: (180-120)/180 = 33.3%
    if "最大回撤" in question and "180" in question and "120" in question:
        if any(x in response for x in ["33.3", "33.33", "1/3"]):
            checks.append(("最大回撤计算", True))
        elif "回撤" in response:
            checks.append(("最大回撤计算", False))

    # VaR 95%: 0.08% - 1.645 * 1.5% ≈ -2.39%
    if "VaR" in question and "95%" in question and "1.5%" in question:
        nums = re.findall(r'-?\d+\.\d+', response)
        for n in nums:
            if 2.3 <= abs(float(n)) <= 2.5:
                checks.append(("VaR计算", True))
                break
        else:
            if "VaR" in response or "var" in response.lower():
                checks.append(("VaR计算", False))

    return checks


def evaluate_model(model, tokenizer, test_data, label="model", consistency_rounds=0):
    """评估模型
    consistency_rounds: >0 时对手写题额外生成 N 轮，检测回复一致性
    """
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
            "subcat": item.get("subcat", ""),
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

        # 量化计算数值正确性（仅量化计算类）
        if category == "量化计算":
            num_checks = check_numerical_correctness(question, response)
            result["numerical_checks"] = num_checks

        # 回复一致性检测（仅手写题，需 consistency_rounds > 0）
        if consistency_rounds > 0 and "q" in item:
            extra_responses = []
            for _ in range(consistency_rounds):
                extra = generate_response(model, tokenizer, question)
                extra_responses.append(extra[:500])
            # 计算各轮回复之间的 ROUGE-L 平均值作为一致性分数
            all_texts = [response[:500]] + extra_responses
            pair_scores = []
            for a_idx in range(len(all_texts)):
                for b_idx in range(a_idx + 1, len(all_texts)):
                    pair_scores.append(compute_rouge_l(all_texts[a_idx], all_texts[b_idx]))
            result["consistency_score"] = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0

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

    # 量化计算数值正确性
    quant_results = [r for r in results if r.get("numerical_checks")]
    if quant_results:
        all_checks = []
        for r in quant_results:
            all_checks.extend(r["numerical_checks"])
        if all_checks:
            correct = sum(1 for _, ok in all_checks if ok)
            print(f"\n  量化计算数值正确性: {correct}/{len(all_checks)} ({correct/len(all_checks)*100:.0f}%)")
            for name, ok in all_checks:
                print(f"    {'✓' if ok else '✗'} {name}")

    # 回复一致性
    consistency = [r["consistency_score"] for r in results if "consistency_score" in r]
    if consistency:
        avg_c = sum(consistency) / len(consistency)
        low_c = sum(1 for c in consistency if c < 0.3)
        print(f"\n  回复一致性 (ROUGE-L between rounds):")
        print(f"    平均一致性: {avg_c:.4f}")
        print(f"    低一致性题目 (<0.3): {low_c}/{len(consistency)}")

    # 对抗性测试子类统计
    adversarial = [r for r in results if r["category"] == "对抗性" and r.get("subcat")]
    if adversarial:
        subcat_names = {"risk": "风险提示", "trap": "逻辑陷阱", "misconception": "数据误解"}
        by_subcat = defaultdict(list)
        for r in adversarial:
            by_subcat[r["subcat"]].append(r)

        # 对抗性测试通过标准：回复中包含警示/否定/风险相关关键词
        warn_keywords = ["风险", "不建议", "不一定", "不能", "谨慎", "过拟合", "陷阱",
                         "误导", "不可靠", "注意", "警惕", "偏差", "不可", "不是",
                         "未来数据", "前视", "幸存者", "不意味", "不代表", "不正确"]

        print(f"\n  对抗性测试子类分析:")
        total_pass = 0
        total_adv = 0
        for subcat in ["risk", "trap", "misconception"]:
            items = by_subcat.get(subcat, [])
            if not items:
                continue
            passed = sum(1 for r in items if any(kw in r["response"] for kw in warn_keywords))
            total_pass += passed
            total_adv += len(items)
            rate = passed / len(items) * 100
            print(f"    {subcat_names[subcat]}: {passed}/{len(items)} 通过 ({rate:.0f}%)")
        if total_adv:
            print(f"    总计: {total_pass}/{total_adv} 通过 ({total_pass/total_adv*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--baseline", action="store_true", help="同时评估基座模型作为对比")
    parser.add_argument("--manual-only", action="store_true", help="仅评估手写测试题")
    parser.add_argument("--holdout-only", action="store_true", help="仅评估 holdout 数据")
    parser.add_argument("--max-holdout", type=int, default=100, help="最多评估的 holdout 条数")
    parser.add_argument("--consistency", type=int, default=0, help="一致性检测轮数（0=关闭，2=推荐）")
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
    if args.consistency:
        print(f"一致性检测: 每题额外生成 {args.consistency} 轮")
    print("\n评估微调模型...")
    finetuned_results = evaluate_model(model, tokenizer, test_data, label="finetuned",
                                       consistency_rounds=args.consistency)
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
        baseline_results = evaluate_model(base_model, base_tokenizer, test_data, label="baseline",
                                          consistency_rounds=args.consistency)
        print_summary(baseline_results, label="基座模型")
        all_results["baseline"] = baseline_results

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
