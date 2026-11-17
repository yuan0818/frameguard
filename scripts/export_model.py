"""
模型导出：从 checkpoint 提取干净的推理权重，支持多种格式。

用法:
    # 导出为 PyTorch 纯权重文件（推荐，最简单）
    python scripts/export_model.py --checkpoint outputs/checkpoints/best_model.pth

    # 导出为 TorchScript（支持 C++ / 无 Python 环境部署）
    python scripts/export_model.py --checkpoint outputs/checkpoints/best_model.pth --format torchscript

    # 导出为 ONNX（支持 ONNX Runtime / TensorRT / 其他框架）
    python scripts/export_model.py --checkpoint outputs/checkpoints/best_model.pth --format onnx

    # 一次性导出全部格式
    python scripts/export_model.py --checkpoint outputs/checkpoints/best_model.pth --format all

输出文件:
    outputs/export/
    ├── model.pth           # 纯 state_dict（推理用）
    ├── model_info.json     # 模型元信息（类别名、输入尺寸等）
    ├── model.pt            # TorchScript 模型
    └── model.onnx          # ONNX 模型
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model, ImageQualityCNN, PretrainedBackbone
from src.config import load_config, get_default_config, config_for_checkpoint


DEFAULT_OUTPUT_DIR = "outputs/export"


def load_model_from_checkpoint(checkpoint_path: str, config) -> torch.nn.Module:
    """从训练 checkpoint 加载模型"""
    backbone = getattr(config.model, "backbone", None)
    freeze_backbone = getattr(config.model, "freeze_backbone", True)

    model = create_model(
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        classifier_dropout=config.model.classifier_dropout,
        input_channels=config.model.input_channels,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        pretrained_path=checkpoint_path,
    )
    model.eval()
    return model


def export_pytorch(model: torch.nn.Module, config, output_dir: str) -> str:
    """导出纯 state_dict（推理用）"""
    out_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), out_path)
    print(f"  [PyTorch] {out_path}")
    return out_path


def export_torchscript(model: torch.nn.Module, config, output_dir: str) -> str:
    """导出为 TorchScript（支持 C++ 部署）"""
    out_path = os.path.join(output_dir, "model.pt")

    image_size = tuple(config.data.image_size)
    example_input = torch.randn(1, 3, *image_size)

    # 尝试 trace，失败则用 script
    try:
        traced = torch.jit.trace(model, example_input, strict=False)
    except Exception:
        traced = torch.jit.script(model)

    traced.save(out_path)

    # 验证
    loaded = torch.jit.load(out_path)
    with torch.no_grad():
        out1 = model(example_input)
        out2 = loaded(example_input)
    assert torch.allclose(out1, out2, atol=1e-5), "TorchScript 输出不一致！"

    print(f"  [TorchScript] {out_path}")
    return out_path


def export_onnx(model: torch.nn.Module, config, output_dir: str) -> str:
    """导出为 ONNX"""
    out_path = os.path.join(output_dir, "model.onnx")

    image_size = tuple(config.data.image_size)
    example_input = torch.randn(1, 3, *image_size)

    torch.onnx.export(
        model,
        example_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=14,
    )

    # 验证 ONNX 模型结构
    try:
        import onnx
        onnx_model = onnx.load(out_path)
        onnx.checker.check_model(onnx_model)
        print(f"  [ONNX] {out_path}  (验证通过)")
    except ImportError:
        print(f"  [ONNX] {out_path}  (pip install onnx 可验证)")

    return out_path


def export_info(config, output_dir: str) -> str:
    """导出模型元信息"""
    out_path = os.path.join(output_dir, "model_info.json")

    # 类别名（处理 Config 对象）
    class_names = config.model.class_names
    if hasattr(class_names, '_data'):
        class_names = class_names._data
    if hasattr(class_names, 'to_dict'):
        class_names = class_names.to_dict()
    if not isinstance(class_names, dict):
        class_names = {}

    # 规范化键为 int
    class_names = {int(k): v for k, v in class_names.items()}

    info = {
        "model_type": getattr(config.model, "backbone", None) or "ImageQualityCNN",
        "letterbox": getattr(config.data, "letterbox", False),
        "num_classes": config.model.num_classes,
        "input_size": config.data.image_size,  # [H, W]
        "input_channels": config.model.input_channels,
        "mean": config.data.mean,
        "std": config.data.std,
        "class_names": class_names,
        "class_labels": {
            v: int(k) for k, v in class_names.items()
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  [Info] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="导出训练好的模型为推理格式"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, default="outputs/checkpoints/best_model.pth",
        help="训练 checkpoint 路径",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径（默认使用 config/default.yaml）",
    )
    parser.add_argument(
        "--format", "-f", type=str, default="pytorch",
        choices=["pytorch", "torchscript", "onnx", "all"],
        help="导出格式",
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 加载配置
    config = config_for_checkpoint(args.checkpoint, args.config)

    # 加载模型
    print(f"加载 checkoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, config)

    # 设备
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 导出
    print(f"\n导出格式: {args.format}")
    print("-" * 40)

    formats = ["pytorch", "torchscript", "onnx"] if args.format == "all" else [args.format]

    exported = []
    for fmt in formats:
        if fmt == "pytorch":
            exported.append(export_pytorch(model, config, output_dir))
        elif fmt == "torchscript":
            exported.append(export_torchscript(model, config, output_dir))
        elif fmt == "onnx":
            exported.append(export_onnx(model, config, output_dir))

    # 始终导出元信息
    export_info(config, output_dir)

    # 打印使用说明
    print(f"\n{'='*55}")
    print("导出完成！使用方法:")
    print(f"{'='*55}")

    if "pytorch" in formats or args.format == "all":
        print("""
# === Python 推理 (model.pth) ===
import torch
import torchvision.transforms as T
from PIL import Image
import json

# 1. 加载元信息
with open("outputs/export/model_info.json") as f:
    info = json.load(f)

# 2. 加载模型（需要本项目 src/model.py）
from src.model import ImageQualityCNN
model = ImageQualityCNN(num_classes=info["num_classes"])
model.load_state_dict(torch.load("outputs/export/model.pth", map_location="cpu"))
model.eval()

# 3. 预处理
transform = T.Compose([
    T.Resize(info["input_size"]),
    T.ToTensor(),
    T.Normalize(mean=info["mean"], std=info["std"]),
])
img = transform(Image.open("test.jpg").convert("RGB")).unsqueeze(0)

# 4. 推理
with torch.no_grad():
    logits = model(img)
    pred = torch.argmax(logits, dim=1).item()
print(f"预测: {info['class_names'][str(pred)]}")
""")

    if "torchscript" in formats or args.format == "all":
        print("""
# === C++ 推理 (model.pt) ===
import torch
model = torch.jit.load("outputs/export/model.pt")
# 其他代码同上，无需 src/model.py
""")

    if "onnx" in formats or args.format == "all":
        print("""
# === ONNX Runtime 推理 (model.onnx) ===
import onnxruntime as ort
session = ort.InferenceSession("outputs/export/model.onnx")
outputs = session.run(None, {"input": img.numpy()})
""")

    print(f"\n输出目录: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
