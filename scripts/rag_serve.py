"""
RAG 增强推理服务
用户问题 → RAG 检索 → 注入 system prompt → ollama 推理 → 返回回答
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg, SYSTEM_PROMPT, call_ollama
from rag_retrieve import RAGRetriever


def rag_inference(retriever, query, model=None):
    """单次 RAG 增强推理"""
    if model is None:
        model = cfg["ollama"]["generation_model"]

    # RAG 检索
    results = retriever.retrieve(query)
    context = retriever.format_context(results)

    # 构建 system prompt
    if context:
        system = f"{SYSTEM_PROMPT}\n\n{context}"
    else:
        system = SYSTEM_PROMPT

    # 调用 ollama
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    response = call_ollama(
        model=model,
        messages=messages,
        temperature=cfg["ollama"].get("temperature", 0.7),
        strip_think=True,
    )

    return {
        "query": query,
        "response": response,
        "rag_results": len(results),
        "rag_scores": [r["score"] for r in results] if results else [],
    }


def interactive_mode():
    """CLI 交互模式"""
    print("=" * 60)
    print(" QuantLLM RAG 增强推理服务")
    print(f" 模型: {cfg['ollama']['generation_model']}")
    print(" 输入问题，输入 q 退出")
    print("=" * 60)

    retriever = RAGRetriever()
    print()

    while True:
        try:
            query = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        result = rag_inference(retriever, query)

        if result["rag_results"] > 0:
            scores_str = ", ".join(f"{s:.3f}" for s in result["rag_scores"])
            print(f"\n[RAG] 检索到 {result['rag_results']} 条参考 (scores: {scores_str})")

        if result["response"]:
            print(f"\n回答:\n{result['response']}\n")
        else:
            print("\n[ERROR] 模型未返回回答\n")


if __name__ == "__main__":
    interactive_mode()
