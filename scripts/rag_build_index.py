"""
RAG 索引构建
读取 merged_train_v2.jsonl → bge-large-zh-v1.5 编码 → FAISS IndexFlatIP
"""

import json
import sys
import time
import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from _config import cfg, path, DATA_DIR

import faiss
from sentence_transformers import SentenceTransformer


def build_index():
    rag_cfg = cfg["rag"]
    input_file = f"{DATA_DIR}/merged_train_v2.jsonl"
    index_path = path(rag_cfg["index_file"])
    meta_path = path(rag_cfg["metadata_file"])

    # 加载数据
    print(f"[RAG] 读取训练数据: {input_file}")
    questions = []
    metadata = []
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            # 支持 messages（当前格式）和 conversations（兼容）
            convs = record.get("messages") or record.get("conversations", [])

            user_msg = ""
            assistant_msg = ""
            for c in convs:
                if c["role"] == "user":
                    user_msg = c["content"]
                elif c["role"] == "assistant":
                    assistant_msg = c["content"]

            if not user_msg:
                continue

            # 跳过含 [MARKET_DATA] 的行情记录
            if "[MARKET_DATA]" in user_msg:
                skipped += 1
                continue

            questions.append(user_msg)
            metadata.append({
                "question": user_msg,
                "answer": assistant_msg,
                "source": record.get("source", "unknown"),
            })

    print(f"[RAG] 有效记录: {len(questions)}, 跳过行情数据: {skipped}")

    # 加载 embedding 模型（强制 CPU，避免抢 GPU 显存）
    print(f"[RAG] 加载 embedding 模型: {rag_cfg['embedding_model']} (CPU)")
    model = SentenceTransformer(rag_cfg["embedding_model"], device="cpu")

    # 批量编码
    print(f"[RAG] 编码 {len(questions)} 条文本...")
    t0 = time.time()
    embeddings = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # bge 需要归一化
    )
    elapsed = time.time() - t0
    print(f"[RAG] 编码完成，耗时 {elapsed:.1f}s，维度 {embeddings.shape}")

    # 构建 FAISS 索引（内积，因为已归一化 = 余弦相似度）
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"[RAG] FAISS 索引构建完成，共 {index.ntotal} 条向量")

    # 保存
    import os
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    faiss.write_index(index, index_path)
    print(f"[RAG] 索引已保存: {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[RAG] 元数据已保存: {meta_path}")

    # 统计
    index_size_mb = os.path.getsize(index_path) / 1024 / 1024
    meta_size_mb = os.path.getsize(meta_path) / 1024 / 1024
    print(f"[RAG] 索引大小: {index_size_mb:.1f}MB, 元数据大小: {meta_size_mb:.1f}MB")


if __name__ == "__main__":
    build_index()
