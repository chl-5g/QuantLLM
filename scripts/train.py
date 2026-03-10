#!/usr/bin/env python3
"""
Qwen2.5-14B QLoRA 微调训练脚本
基座: unsloth/Qwen2.5-14B-bnb-4bit
配置: config.yaml
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import Dataset

from _config import cfg, MODEL_NAME, MAX_SEQ_LENGTH, OUTPUT_DIR, DATA_DIR

# ============================================================
# 配置
# ============================================================
DATA_FILE = os.path.join(DATA_DIR, "merged_train_v2.jsonl")
EVAL_RATIO = cfg["training"].get("eval_ratio", 0.02)
SEED = cfg["training"]["seed"]

# ============================================================
# 1. 加载模型
# ============================================================
print("=" * 60)
print("1. 加载基座模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# ============================================================
# 2. 配置 LoRA
# ============================================================
print("2. 配置 LoRA 适配器...")
lcfg = cfg["lora"]
model = FastLanguageModel.get_peft_model(
    model,
    r=lcfg["r"],
    target_modules=lcfg["target_modules"],
    lora_alpha=lcfg["alpha"],
    lora_dropout=lcfg["dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",  # 节省显存
    use_rslora=lcfg.get("use_rslora", True),
    random_state=SEED,
)

# 打印可训练参数
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"   可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

# ============================================================
# 3. 加载数据
# ============================================================
print("3. 加载训练数据...")
records = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# 转成 text 格式 (ChatML)
def format_chatml(record):
    messages = record["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"  # 训练时的生成起点
    return text

texts = [format_chatml(r) for r in records]
full_dataset = Dataset.from_dict({"text": texts})

# 划分训练集/验证集
split = full_dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)
dataset = split["train"]
eval_dataset = split["test"]
print(f"   训练集: {len(dataset)} 条，验证集: {len(eval_dataset)} 条")

# ============================================================
# 4. 训练
# ============================================================
print("4. 开始训练...")
print("=" * 60)

tcfg = cfg["training"]
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        warmup_steps=tcfg["warmup_steps"],
        num_train_epochs=tcfg["num_train_epochs"],
        learning_rate=tcfg["learning_rate"],
        fp16=False,
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg["save_total_limit"],
        eval_steps=tcfg["eval_steps"],
        eval_strategy="steps",
        optim=tcfg["optim"],
        seed=SEED,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    ),
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=tcfg.get("early_stopping_patience", 3),
    )],
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    packing=True,  # 多条短样本打包，提高效率
)

# 打印显存使用
import torch
print(f"训练前 GPU 显存: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# 自动从最新 checkpoint 恢复
import glob as _glob
checkpoints = sorted(_glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume_ckpt = checkpoints[-1] if checkpoints else None
if resume_ckpt:
    print(f"从 {resume_ckpt} 恢复训练...")
trainer_stats = trainer.train(resume_from_checkpoint=resume_ckpt)

# ============================================================
# 5. 保存
# ============================================================
print("\n" + "=" * 60)
print("5. 保存模型...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n训练完成！")
print(f"  训练时长: {trainer_stats.metrics['train_runtime']:.0f} 秒")
print(f"  训练损失: {trainer_stats.metrics['train_loss']:.4f}")
print(f"  模型保存: {OUTPUT_DIR}")
