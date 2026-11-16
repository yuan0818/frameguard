"""
配置管理模块：加载和解析 YAML 配置文件。
"""

import os
import yaml
from typing import Dict, Any


class Config:
    """配置类，支持属性式访问（config.data.root_dir）"""

    def __init__(self, config_dict: Dict[str, Any]):
        # 存储原始字典，用于处理非字符串键
        self._data = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 仅当所有键都是字符串时才递归为 Config，否则保持原生 dict
                if all(isinstance(k, str) for k in value.keys()):
                    setattr(self, str(key), Config(value))
                else:
                    setattr(self, str(key), value)
            else:
                setattr(self, str(key), value)

    def to_dict(self) -> Dict[str, Any]:
        """将配置递归转换为普通字典"""
        result = {}
        for key, value in self.__dict__.items():
            if key == "_data":
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(config_path: str) -> Config:
    """
    从 YAML 文件加载配置。

    Args:
        config_path: YAML 配置文件路径

    Returns:
        Config 对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def get_default_config() -> Config:
    """
    获取默认配置文件路径（项目根目录下的 config/default.yaml）。

    Returns:
        Config 对象
    """
    # config.py 位于 src/ 下，项目根目录是上一级
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config_path = os.path.join(project_root, "config", "default.yaml")
    return load_config(default_config_path)


def config_for_checkpoint(checkpoint_path, explicit_path=None):
    """Explicit config wins; otherwise use saved training metadata or legacy 7-class defaults."""
    if explicit_path:
        return load_config(explicit_path)
    import torch
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return Config(state["config"]) if isinstance(state, dict) and "config" in state else get_default_config()
