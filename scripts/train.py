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
sources = [r.get("source", "unknown") for r in records]
full_dataset = Dataset.from_dict({"text": texts, "source": sources})

# 分层抽样：确保每个数据源在验证集中都有代表性
from collections import Counter
source_counts = Counter(sources)
print(f"   数据源分布: {dict(source_counts)}")

unique_sources = list(source_counts.keys())
if len(unique_sources) > 1:
    # 将 source 列转为 ClassLabel 以支持分层抽样
    from datasets import ClassLabel
    full_dataset = full_dataset.cast_column(
        "source", ClassLabel(names=sorted(unique_sources))
    )
    split = full_dataset.train_test_split(
        test_size=EVAL_RATIO, seed=SEED, stratify_by_column="source"
    )
else:
    # 只有一个来源，普通随机拆分即可
    print(f"   仅单一数据源 '{unique_sources[0]}'，使用随机拆分")
    split = full_dataset.train_test_split(test_size=EVAL_RATIO, seed=SEED)

dataset = split["train"].remove_columns("source")
eval_dataset = split["test"].remove_columns("source")
print(f"   训练集: {len(dataset)} 条，验证集: {len(eval_dataset)} 条（分层抽样）")

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
        lr_scheduler_type="cosine",
        seed=SEED,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
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

# 保存训练日志到文件
log_file = os.path.join(OUTPUT_DIR, "training_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(log_file, "w", encoding="utf-8") as lf:
    lf.write("=" * 60 + "\n")
    lf.write("QuantLLM 训练日志\n")
    lf.write("=" * 60 + "\n\n")
    lf.write(f"模型: {MODEL_NAME}\n")
    lf.write(f"LoRA: r={cfg['lora']['r']}, alpha={cfg['lora']['alpha']}, rslora={cfg['lora'].get('use_rslora')}\n")
    lf.write(f"训练集: {len(dataset)} 条，验证集: {len(eval_dataset)} 条\n")
    lf.write(f"Epochs: {cfg['training']['num_train_epochs']}, LR: {cfg['training']['learning_rate']}\n")
    lf.write(f"调度: cosine, Warmup: {cfg['training']['warmup_steps']} steps\n\n")
    for entry in trainer.state.log_history:
        lf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    lf.write(f"\n最终指标:\n")
    for k, v in trainer_stats.metrics.items():
        lf.write(f"  {k}: {v}\n")
print(f"  训练日志: {log_file}")
