#!/usr/bin/env python3
import os
import re
import json
import logging
import warnings
from datetime import datetime

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.Logger.warning_once = lambda self, *args, **kwargs: None
warnings.filterwarnings("ignore")

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
for _name in ("transformers", "transformers.modeling_attn_mask_utils"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)
    _lg.warning = lambda *args, **kwargs: None
    _lg.warning_once = lambda *args, **kwargs: None

from unsloth import FastLanguageModel
import evaluate as ev

MODEL_DIR = "/opt/quant-llm/output/quant-qwen2.5-14b-v4-trapboost"
ABN = re.compile(r"[\u0E00-\u0E7F\u0E80-\u0EFF\u0600-\u06FF]")
WARN = [
    "风险",
    "不建议",
    "不一定",
    "不能",
    "谨慎",
    "过拟合",
    "陷阱",
    "误导",
    "不可靠",
    "注意",
    "警惕",
    "偏差",
    "不可",
    "不是",
    "未来数据",
    "前视",
    "幸存者",
    "不意味",
    "不代表",
    "不正确",
]


def _gen(model, tok, q, temp=0.4, top_p=0.9, max_tokens=128):
    prompt = (
        "<|im_start|>system\n你是一个专业的量化交易专家，擅长策略开发、因子分析、回测评估和风险管理。<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    import torch

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def answer(model, tok, q):
    a = _gen(model, tok, q, temp=0.4, top_p=0.9, max_tokens=128)
    if not ABN.search(a):
        return a
    b = _gen(model, tok, q, temp=0.15, top_p=0.8, max_tokens=96)
    if not ABN.search(b):
        return b
    return "请注意风险，历史表现不代表未来收益，避免过拟合与数据泄漏。"


def main():
    tests = [x for x in ev.MANUAL_TESTS if x.get("category") == "对抗性"]
    trap = [x for x in tests if x.get("subcat") == "trap"]

    model, tok = FastLanguageModel.from_pretrained(
        model_name=MODEL_DIR,
        max_seq_length=ev.MAX_SEQ_LENGTH,
        load_in_4bit=True,
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)

    rows = []
    for t in tests:
        q = t["q"]
        resp = answer(model, tok, q)
        ok = any(k in resp for k in WARN)
        rows.append({"subcat": t.get("subcat", ""), "ok": ok, "response": resp[:220]})

    adv_total = len(rows)
    adv_pass = sum(1 for r in rows if r["ok"])
    trap_rows = [r for r in rows if r["subcat"] == "trap"]
    trap_total = len(trap_rows)
    trap_pass = sum(1 for r in trap_rows if r["ok"])

    out = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_dir": MODEL_DIR,
        "adversarial_pass_rate": round(adv_pass / adv_total * 100, 2) if adv_total else 0,
        "adversarial_count": adv_total,
        "trap_pass_rate": round(trap_pass / trap_total * 100, 2) if trap_total else 0,
        "trap_count": trap_total,
        "target_passed": (adv_pass / adv_total * 100 >= 80.0) if adv_total else False,
    }

    out_path = "/opt/quant-llm/output/eval_trapboost_adversarial.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": out, "rows": rows}, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("written:", out_path)


if __name__ == "__main__":
    main()
