#!/usr/bin/env python3
"""导出 LoRA/合并模型为 GGUF，供 Ollama 本地推理。"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

from _config import MAX_SEQ_LENGTH, OUTPUT_DIR, PROJECT_ROOT

warnings.filterwarnings("ignore")


def _default_checkpoint() -> Path:
    out_dir = Path(OUTPUT_DIR)
    if out_dir.exists():
        return out_dir
    # 兜底：兼容旧目录
    return Path(PROJECT_ROOT) / "output" / "quant-qwen2.5-14b-lora" / "checkpoint-1000"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to GGUF")
    parser.add_argument("--checkpoint", default=str(_default_checkpoint()), help="LoRA checkpoint 或 merged 模型目录")
    parser.add_argument("--out-dir", default=str(Path(PROJECT_ROOT) / "output" / "gguf"), help="GGUF 输出目录")
    parser.add_argument("--quant", default="q4_k_m", help="GGUF 量化格式，如 q4_k_m/q5_k_m/f16")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    from unsloth import FastLanguageModel  # 延迟导入，避免无依赖时脚本直接崩

    print(f"[GGUF] loading checkpoint: {ckpt}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ckpt),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    print(f"[GGUF] exporting to: {out_dir} (quant={args.quant})")
    model.save_pretrained_gguf(
        str(out_dir),
        tokenizer,
        quantization_method=args.quant,
    )
    print(f"[GGUF] done: {out_dir}")
    print("[GGUF] next: ollama create quant-qwen2.5-14b -f <Modelfile>")


if __name__ == "__main__":
    main()
