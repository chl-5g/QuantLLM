#!/usr/bin/env python3
"""
将 QLoRA checkpoint 合并到基座模型并导出 GGUF (Q4_K_M)
输出: /opt/quant-llm/output/quant-qwen2.5-14b.gguf
"""
import os
import warnings
warnings.filterwarnings("ignore")

from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/Qwen2.5-14B-bnb-4bit"
MAX_SEQ_LENGTH = 2048
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(PROJECT_ROOT, "output", "quant-qwen2.5-14b-lora", "checkpoint-1000")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "gguf")

print("1. 加载基座模型 + LoRA...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CHECKPOINT,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

print("2. 导出 GGUF (q4_k_m)...")
model.save_pretrained_gguf(
    OUTPUT_DIR,
    tokenizer,
    quantization_method="q4_k_m",
)

print(f"\n导出完成！文件在: {OUTPUT_DIR}")
print("接下来用 ollama 导入:")
print(f"  ollama create quant-qwen2.5-14b -f <Modelfile>")
