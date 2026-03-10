"""
中心化配置加载器
所有脚本通过 from _config import cfg, PROJECT_ROOT 使用
"""

import os
import yaml

# 项目根目录（config.yaml 所在位置）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_config_path = os.path.join(PROJECT_ROOT, "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def path(relative):
    """将相对路径转为绝对路径"""
    return os.path.join(PROJECT_ROOT, relative)


# 常用路径快捷访问
DATA_DIR = path(cfg["data"]["training_dir"])
OUTPUT_DIR = path(cfg["model"]["output_dir"])
MODEL_NAME = cfg["model"]["base_model"]
MAX_SEQ_LENGTH = cfg["model"]["max_seq_length"]
SYSTEM_PROMPT = cfg["system_prompt"]
OLLAMA_URL = cfg["ollama"]["url"]
