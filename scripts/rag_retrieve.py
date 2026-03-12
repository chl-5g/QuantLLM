"""
RAG 检索模块
提供 RAGRetriever 类，支持独立测试
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg, path

import faiss
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self):
        rag_cfg = cfg["rag"]
        index_path = path(rag_cfg["index_file"])
        meta_path = path(rag_cfg["metadata_file"])

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS 索引不存在: {index_path}\n请先运行: bash run.sh rag-build")

        # 加载模型和索引
        print(f"[RAG] 加载 embedding 模型: {rag_cfg['embedding_model']} (CPU)")
        self.model = SentenceTransformer(rag_cfg["embedding_model"], device="cpu")

        print(f"[RAG] 加载 FAISS 索引: {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"[RAG] 加载元数据: {meta_path}")
        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.top_k = rag_cfg["top_k"]
        self.score_threshold = rag_cfg["score_threshold"]
        self.max_ref_tokens = rag_cfg["max_ref_tokens"]

        print(f"[RAG] 就绪: {self.index.ntotal} 条向量, top_k={self.top_k}, threshold={self.score_threshold}")

    def retrieve(self, query, top_k=None):
        """检索与 query 最相关的文档"""
        # 含行情数据的查询跳过 RAG
        if "[MARKET_DATA]" in query:
            return []

        if top_k is None:
            top_k = self.top_k

        # 编码查询
        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)

        # FAISS 搜索
        scores, indices = self.index.search(q_emb, top_k)

        # 过滤低分结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < self.score_threshold:
                continue
            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                "question": meta["question"],
                "answer": meta["answer"],
                "source": meta.get("source", "unknown"),
            })

        return results

    def format_context(self, results):
        """将检索结果格式化为参考资料块"""
        if not results:
            return ""

        parts = []
        total_chars = 0
        # 粗略估算：1 token ≈ 1.5 中文字符
        max_chars = int(self.max_ref_tokens * 1.5)

        for i, r in enumerate(results, 1):
            entry = f"【{i}】{r['question']}\n{r['answer']}"
            entry_len = len(entry)
            if total_chars + entry_len > max_chars:
                # 截断最后一条
                remaining = max_chars - total_chars
                if remaining > 50:
                    parts.append(entry[:remaining] + "...")
                break
            parts.append(entry)
            total_chars += entry_len

        if not parts:
            return ""

        return "[参考资料]\n" + "\n\n".join(parts) + "\n[/参考资料]"


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "什么是夏普比率"

    retriever = RAGRetriever()
    results = retriever.retrieve(query)

    print(f"\n查询: {query}")
    print(f"检索到 {len(results)} 条结果:\n")

    for i, r in enumerate(results, 1):
        print(f"--- [{i}] score={r['score']:.4f} source={r['source']} ---")
        print(f"Q: {r['question'][:100]}")
        print(f"A: {r['answer'][:200]}")
        print()

    context = retriever.format_context(results)
    if context:
        print("=== 格式化参考资料 ===")
        print(context)
