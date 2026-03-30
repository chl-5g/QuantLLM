#!/usr/bin/env python3
import json
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

from _config import cfg, path, MAX_SEQ_LENGTH

DATA_FILE = path("training-data/trap_boost_v1.jsonl")
BASE_MODEL_DIR = path(cfg["model"]["output_dir"])
OUTPUT_DIR = path("output/quant-qwen2.5-14b-v4-trapboost")
SEED = cfg["training"].get("seed", 42)


def format_chatml(record):
    msgs = record["messages"]
    text = ""
    for m in msgs:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(DATA_FILE)

    print("=" * 60)
    print("Trap Boost 快训")
    print("base:", BASE_MODEL_DIR)
    print("data:", DATA_FILE)
    print("out :", OUTPUT_DIR)

    print("1) 加载当前 v4 模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        local_files_only=True,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   可训练参数: {trainable:,}/{total:,} ({trainable/total*100:.4f}%)")

    print("2) 读取定向样本...")
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if len(records) < 20:
        raise RuntimeError("trap_boost 样本过少")

    texts = [format_chatml(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=0.15, seed=SEED)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"   train={len(train_ds)} eval={len(eval_ds)}")

    print("3) 开始快训...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=5,
        max_steps=120,
        bf16=cfg["training"].get("bf16", True),
        fp16=False,
        logging_steps=10,
        save_steps=60,
        eval_steps=30,
        eval_strategy="steps",
        optim=cfg["training"].get("optim", "adamw_8bit"),
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )

    stats = trainer.train()

    print("4) 保存模型...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("done")
    print("train_loss", stats.metrics.get("train_loss"))
    print("train_runtime", stats.metrics.get("train_runtime"))


if __name__ == "__main__":
    main()
