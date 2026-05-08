"""LLM 调用工具 — 支持 ollama / OpenAI 兼容接口"""

import json
import time
import sys
import os
import requests
from pydantic import BaseModel

sys.path.insert(0, '/opt/quant-llm/scripts')
try:
    from _config import call_ollama, OLLAMA_URL
except ImportError:
    call_ollama = None
    OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/v1/chat/completions')


def call_llm(prompt, pydantic_model, agent_name=None, state=None,
             max_retries=3, default_factory=None):
    """调用 LLM，支持 JSON 模式输出"""
    model_name = state.get("metadata", {}).get("model_name", "qwen3:14b") if state else "qwen3:14b"
    model_provider = state.get("metadata", {}).get("model_provider", "ollama") if state else "ollama"

    for attempt in range(max_retries):
        try:
            if model_provider == "ollama":
                result = _call_ollama_json(prompt, pydantic_model, model_name, agent_name)
            else:
                result = _call_openai_compatible(prompt, pydantic_model, model_name, agent_name, state)
            if result is not None:
                return result
        except Exception as e:
            if agent_name:
                print(f"  [{agent_name}] LLM retry {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))

    if default_factory:
        return default_factory()
    return _create_default(pydantic_model)


def _call_ollama_json(prompt, pydantic_model, model_name, agent_name):
    """通过 ollama 调用，要求 JSON 输出"""
    # 提取消息列表
    if hasattr(prompt, 'messages'):
        messages = []
        for m in prompt.messages:
            role = 'user' if m.type == 'human' else ('assistant' if m.type == 'ai' else 'system')
            messages.append({"role": role, "content": m.content})
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    if call_ollama is not None:
        content = call_ollama(
            model=model_name, messages=messages,
            temperature=0.3, num_predict=2048, format="json",
        )
    else:
        resp = requests.post(
            OLLAMA_URL.replace('/v1/chat/completions', '/api/chat'),
            json={"model": model_name, "messages": messages, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 2048}, "format": "json"},
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]

    if not content:
        return None
    parsed = _extract_json(content)
    if parsed:
        return pydantic_model(**parsed)
    return None


def _call_openai_compatible(prompt, pydantic_model, model_name, agent_name, state):
    """通过 OpenAI 兼容接口调用"""
    base_url = state.get("metadata", {}).get("base_url", os.environ.get("OPENAI_BASE_URL", ""))
    api_key = state.get("metadata", {}).get("api_key", os.environ.get("OPENAI_API_KEY", ""))
    if not base_url or not api_key:
        return None

    if hasattr(prompt, 'messages'):
        messages = []
        for m in prompt.messages:
            role = 'user' if m.type == 'human' else ('assistant' if m.type == 'ai' else 'system')
            messages.append({"role": role, "content": m.content})
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model_name, "messages": messages, "temperature": 0.3,
              "response_format": {"type": "json_object"}},
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if parsed:
        return pydantic_model(**parsed)
    return None


def _extract_json(content):
    """从 LLM 输出中提取 JSON"""
    if not content:
        return None
    content = content.strip()

    # 1. markdown ```json 代码块
    start = content.find("```json")
    if start != -1:
        inner = content[start + 7:]
        end = inner.find("```")
        if end != -1:
            try:
                return json.loads(inner[:end].strip())
            except json.JSONDecodeError:
                pass

    # 2. markdown ``` 代码块
    start = content.find("```")
    if start != -1:
        inner = content[start + 3:]
        end = inner.find("```")
        if end != -1:
            try:
                return json.loads(inner[:end].strip())
            except json.JSONDecodeError:
                pass

    # 3. 直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 4. 找第一个完整 JSON 对象
    brace_start = content.find('{')
    if brace_start != -1:
        depth = 0
        for i, ch in enumerate(content[brace_start:], brace_start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def _create_default(model_class):
    """生成安全的默认响应"""
    defaults = {}
    for name, field in model_class.model_fields.items():
        if field.annotation == str:
            defaults[name] = "insufficient data"
        elif field.annotation == float:
            defaults[name] = 0.0
        elif field.annotation == int:
            defaults[name] = 0
        elif hasattr(field.annotation, '__origin__') and field.annotation.__origin__ == dict:
            defaults[name] = {}
        elif hasattr(field.annotation, '__args__') and field.annotation.__args__:
            defaults[name] = field.annotation.__args__[0]
        else:
            defaults[name] = None
    return model_class(**defaults)
