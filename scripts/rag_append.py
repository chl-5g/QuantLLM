"""
RAG 增量追加：将新文档追加到现有 FAISS 索引和元数据中
用法: python scripts/rag_append.py <jsonl文件>

JSONL 格式（每行）:
{"question": "...", "answer": "...", "source": "..."}
"""

import json
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg, path

import faiss
from sentence_transformers import SentenceTransformer


def append_to_index(input_file: str):
    rag_cfg = cfg["rag"]
    index_path = path(rag_cfg["index_file"])
    meta_path = path(rag_cfg["metadata_file"])

    # 读取新数据
    print(f"[RAG] 读取新数据: {input_file}")
    new_records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("question"):
                new_records.append(record)

    if not new_records:
        print("[RAG] 没有有效记录，退出")
        return

    print(f"[RAG] 新增 {len(new_records)} 条记录")

    # 加载现有索引
    print(f"[RAG] 加载现有索引: {index_path}")
    index = faiss.read_index(index_path)
    old_count = index.ntotal
    print(f"[RAG] 现有索引: {old_count} 条向量")

    # 加载 embedding 模型
    print(f"[RAG] 加载 embedding 模型: {rag_cfg['embedding_model']} (CPU)")
    model = SentenceTransformer(rag_cfg["embedding_model"], device="cpu")

    # 编码新文本
    questions = [r["question"] for r in new_records]
    print(f"[RAG] 编码 {len(questions)} 条文本...")
    t0 = time.time()
    embeddings = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    print(f"[RAG] 编码完成，耗时 {elapsed:.1f}s")

    # 追加到索引
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)
    print(f"[RAG] 索引更新: {old_count} → {index.ntotal} 条向量")

    # 追加到元数据
    with open(meta_path, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[RAG] 元数据已追加")

    print(f"[RAG] 完成！新增 {len(new_records)} 条")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <jsonl文件>")
        sys.exit(1)
    append_to_index(sys.argv[1])
