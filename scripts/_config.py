"""
中心化配置加载器
所有脚本通过 from _config import cfg, PROJECT_ROOT 使用
"""

import os
import sys
import yaml

# 项目根目录（config.yaml 所在位置）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def path(relative):
    """将相对路径转为绝对路径"""
    return os.path.join(PROJECT_ROOT, relative)


# ============================================================
# 配置合法性校验
# ============================================================
def _validate():
    errors = []

    # 必填顶级键
    for key in ("data", "model", "lora", "training", "ollama", "system_prompt"):
        if key not in cfg:
            errors.append(f"缺少必填配置项: {key}")

    if errors:
        # 顶级键缺失则无法继续校验子项
        for e in errors:
            print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 数值范围校验
    checks = [
        (cfg["lora"]["r"], 4, 256, "lora.r"),
        (cfg["lora"]["alpha"], 1, 512, "lora.alpha"),
        (cfg["lora"]["dropout"], 0, 0.5, "lora.dropout"),
        (cfg["training"]["batch_size"], 1, 64, "training.batch_size"),
        (cfg["training"]["gradient_accumulation_steps"], 1, 128, "training.gradient_accumulation_steps"),
        (cfg["training"]["learning_rate"], 1e-6, 1e-2, "training.learning_rate"),
        (cfg["training"]["num_train_epochs"], 1, 20, "training.num_train_epochs"),
        (cfg["model"]["max_seq_length"], 256, 32768, "model.max_seq_length"),
    ]
    for val, lo, hi, name in checks:
        if not (lo <= val <= hi):
            errors.append(f"{name}={val} 超出合理范围 [{lo}, {hi}]")

    # 必填字符串
    if not cfg["model"].get("base_model"):
        errors.append("model.base_model 不能为空")
    if not cfg.get("system_prompt"):
        errors.append("system_prompt 不能为空")

    # ollama URL 格式
    url = cfg["ollama"].get("url", "")
    if not url.startswith("http"):
        errors.append(f"ollama.url 格式错误: {url}")

    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        sys.exit(1)


_validate()

# 常用路径快捷访问（校验通过后才赋值）
DATA_DIR = path(cfg["data"]["training_dir"])
OUTPUT_DIR = path(cfg["model"]["output_dir"])
MODEL_NAME = cfg["model"]["base_model"]
MAX_SEQ_LENGTH = cfg["model"]["max_seq_length"]
SYSTEM_PROMPT = cfg["system_prompt"]
OLLAMA_URL = cfg["ollama"]["url"]


# ============================================================
# ollama 统一调用（带健康检查+重试）
# ============================================================
def call_ollama(model, messages, temperature=0.7, num_predict=4096,
                max_retries=3, timeout=None, strip_think=False):
    """
    统一的 ollama 调用函数。
    model: 模型名称（如 qwen3:14b, deepseek-r1:32b）
    messages: [{"role": "user", "content": "..."}]
    strip_think: 是否去除 <think> 标签（qwen3 场景）
    返回 content 字符串，失败返回 None
    """
    import time
    import requests as _requests

    if timeout is None:
        timeout = cfg["ollama"].get("timeout", 180)

    # 健康检查
    base_url = OLLAMA_URL.rsplit("/", 2)[0]  # http://127.0.0.1:11434
    try:
        _requests.get(f"{base_url}/api/tags", timeout=5)
    except Exception:
        print("[OLLAMA] 服务未运行或无法连接，请先启动 ollama", file=sys.stderr)
        return None

    for attempt in range(max_retries):
        try:
            resp = _requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": num_predict},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            if strip_think and "<think>" in content:
                content = content.split("</think>")[-1].strip()
            return content
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  [OLLAMA] 重试 ({attempt+1}/{max_retries}): {e}，{wait}s后重试...")
                time.sleep(wait)
            else:
                print(f"  [OLLAMA] 调用失败（已重试{max_retries}次）: {e}")
                return None
