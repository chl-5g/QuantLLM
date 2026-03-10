#!/usr/bin/env python3
"""
Qwen2.5-14B QLoRA 微调训练脚本
基座: unsloth/Qwen2.5-14B-bnb-4bit
数据: /tmp/training-data/merged_train.jsonl (30k 条中文金融+量化)
输出: /tmp/training-data/output/quant-qwen2.5-14b-lora
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "unsloth/Qwen2.5-14B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "quant-qwen2.5-14b-lora")
DATA_FILE = os.path.join(PROJECT_ROOT, "training-data", "merged_train.jsonl")

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
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 节省显存
    random_state=42,
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
dataset = Dataset.from_dict({"text": texts})
print(f"   数据条数: {len(dataset)}")

# ============================================================
# 4. 训练
# ============================================================
print("4. 开始训练...")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # 等效 batch_size=8
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    ),
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
