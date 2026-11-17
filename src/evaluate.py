"""
评估模块：在验证集/测试集上评估模型性能。

用法:
    python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth
    python -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth --config custom.yaml
"""

import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, get_default_config, config_for_checkpoint
from src.model import ImageQualityCNN, create_model
from src.dataset import create_data_loaders, create_eval_loader, heldout_overlap, source_group


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在给定数据集上运行模型，收集所有预测结果和真实标签。

    Returns:
        (all_labels, all_predictions) - numpy arrays
    """
    model.eval()
    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc="Evaluating")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    """
    绘制并保存混淆矩阵图。

    Args:
        cm: 混淆矩阵 (num_classes × num_classes)
        class_names: 类别名称列表
        save_path: 图片保存路径
        title: 图表标题
    """
    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 1.2), max(6, num_classes * 1.0)))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个单元格中标注数值
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"混淆矩阵已保存至: {save_path}")


def evaluate(
    checkpoint_path: str,
    config,
    output_confusion_matrix: bool = True,
    allow_legacy_evaluation: bool = False,
    data_root: Optional[str] = None,
) -> Dict:
    """
    执行完整评估流程。

    Args:
        checkpoint_path: 模型权重路径
        config: Config 对象
        output_confusion_matrix: 是否输出混淆矩阵图
        data_root: 独立留出数据的根目录（与训练数据相同的 normal/ + exception/ 布局）。
            指定时评估其中全部样本，并拒绝与训练/验证来源重叠的数据；
            不指定时评估 checkpoint 记录的验证划分。

    Returns:
        评估指标字典
    """
    device = get_device()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print(f"\n{'='*60}")
    print("模型评估")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"模型: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    saved_split = state.get("split_manifest")
    if saved_split is None and not allow_legacy_evaluation:
        raise ValueError("Legacy checkpoint has no source-group split provenance. "
                         "Retrain before reporting validation metrics, or use "
                         "--allow-legacy-evaluation for diagnostics only.")
    if saved_split is None:
        print("WARNING: legacy checkpoint; these metrics are NOT an independent benchmark.")

    # --- 加载数据 ---
    use_letterbox = getattr(config.data, "letterbox", False)

    if data_root:
        data_root = os.path.abspath(data_root)
        print(f"留出数据: {data_root}")
        eval_loader = create_eval_loader(
            normal_dir=os.path.join(data_root, config.data.normal_dir),
            exception_dir=os.path.join(data_root, config.data.exception_dir),
            csv_filename=config.data.csv_filename,
            image_exts=config.data.image_extensions,
            video_exts=config.data.video_extensions,
            image_size=tuple(config.data.image_size),
            mean=tuple(config.data.mean),
            std=tuple(config.data.std),
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            video_max_frames=config.data.video.max_frames_per_video,
            video_frame_interval=config.data.video.frame_interval_sec,
            use_letterbox=use_letterbox,
        )
        if saved_split is not None:
            overlap = heldout_overlap(eval_loader.dataset.samples, saved_split)
            if overlap:
                raise ValueError(
                    f"Held-out data shares {len(overlap)} source group(s) with training/validation "
                    f"(e.g. {overlap[:5]}); it is not an independent test set."
                )
    else:
        normal_dir = os.path.join(project_root, config.data.root_dir, config.data.normal_dir)
        exception_dir = os.path.join(project_root, config.data.root_dir, config.data.exception_dir)

        _, eval_loader = create_data_loaders(
            normal_dir=normal_dir,
            exception_dir=exception_dir,
            csv_filename=config.data.csv_filename,
            image_exts=config.data.image_extensions,
            video_exts=config.data.video_extensions,
            image_size=tuple(config.data.image_size),
            mean=tuple(config.data.mean),
            std=tuple(config.data.std),
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            train_ratio=config.data.split.train_ratio,
            val_ratio=config.data.split.val_ratio,
            random_seed=config.data.split.random_seed,
            video_max_frames=config.data.video.max_frames_per_video,
            video_frame_interval=config.data.video.frame_interval_sec,
            use_letterbox=use_letterbox,
        )

        if saved_split is not None:
            actual_groups = {source_group(s) for s in eval_loader.dataset.samples}
            if actual_groups != set(saved_split["val_groups"]) or actual_groups & set(saved_split["train_groups"]):
                raise ValueError("Dataset split changed since training; use the original data/config")

    if any(s.label >= config.model.num_classes for s in eval_loader.dataset.samples):
        raise ValueError("Dataset labels exceed configured num_classes")

    # --- 加载模型 ---
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
    model = model.to(device)

    # --- 评估 ---
    labels, preds = evaluate_model(model, eval_loader, device)

    # --- 计算指标 ---
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(config.model.num_classes)), average=None, zero_division=0
    )
    all_class_ids = list(range(config.model.num_classes))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, labels=all_class_ids, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, labels=all_class_ids, average="weighted", zero_division=0
    )

    # 类别名称
    class_names_dict = config.model.class_names
    if hasattr(class_names_dict, '_data'):
        # Config 对象：从 _data 获取
        class_names_dict = class_names_dict._data
    if not isinstance(class_names_dict, dict):
        class_names_dict = {}

    num_classes = config.model.num_classes
    class_names = [class_names_dict.get(i, f"class_{i}") for i in range(num_classes)]

    # --- 输出结果 ---
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"总体准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"\nMacro Average:")
    print(f"  Precision: {precision_macro:.4f}")
    print(f"  Recall:    {recall_macro:.4f}")
    print(f"  F1-score:  {f1_macro:.4f}")
    print(f"\nWeighted Average:")
    print(f"  Precision: {precision_weighted:.4f}")
    print(f"  Recall:    {recall_weighted:.4f}")
    print(f"  F1-score:  {f1_weighted:.4f}")

    print(f"\n各类别详细指标:")
    print(f"{'类别':<20} {'Precision':<12} {'Recall':<12} {'F1-score':<12} {'Support':<10}")
    print("-" * 70)
    for i in range(num_classes):
        print(
            f"{class_names[i]:<20} "
            f"{precision[i]:<12.4f} "
            f"{recall[i]:<12.4f} "
            f"{f1[i]:<12.4f} "
            f"{support[i]:<10}"
        )

    # --- 分类报告 ---
    print(f"\n分类报告 (sklearn):")
    print(classification_report(
        labels, preds, labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    ))

    # --- 混淆矩阵 ---
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    print(f"\n混淆矩阵:")
    header = " " * 15 + "".join(f"{name:>8}" for name in class_names)
    print(header)
    for i in range(num_classes):
        row = f"{class_names[i]:>15}" + "".join(f"{cm[i, j]:8d}" for j in range(num_classes))
        print(row)

    if output_confusion_matrix:
        cm_path = os.path.join(
            project_root,
            config.output.log_dir,
            ("heldout_" if data_root else "") + config.output.confusion_matrix_name,
        )
        plot_confusion_matrix(cm, class_names, cm_path, "Confusion Matrix - Image Quality Classifier")

    # 返回结果字典
    results = {
        "accuracy": accuracy,
        "precision_per_class": {class_names[i]: precision[i] for i in range(num_classes)},
        "recall_per_class": {class_names[i]: recall[i] for i in range(num_classes)},
        "f1_per_class": {class_names[i]: f1[i] for i in range(num_classes)},
        "support_per_class": {class_names[i]: support[i] for i in range(num_classes)},
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="图片/视频质量检测 CNN 分类器 - 评估"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 路径（如 outputs/checkpoints/best_model.pth）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认使用 config/default.yaml）",
    )
    parser.add_argument(
        "--no-cm",
        action="store_true",
        help="不生成混淆矩阵图片",
    )
    parser.add_argument("--allow-legacy-evaluation", action="store_true", help="Diagnostic only: allow weights without split provenance")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="独立留出数据根目录（含 normal/ 与 exception/data.csv）；评估其中全部样本，"
             "并拒绝与训练/验证来源重叠的数据",
    )
    args = parser.parse_args()

    config = config_for_checkpoint(args.checkpoint, args.config)

    # 修复 class_names 从 Config 对象到 dict
    # 如果是原生 dict（键为非字符串），直接使用
    # 如果是 Config 对象，从 _data 获取
    cm_dict = config.model.class_names
    if hasattr(cm_dict, '_data'):
        cm_dict = cm_dict._data
    if not isinstance(cm_dict, dict):
        cm_dict = {}
    config.model.class_names = cm_dict

    evaluate(
        checkpoint_path=args.checkpoint,
        config=config,
        output_confusion_matrix=not args.no_cm,
        allow_legacy_evaluation=args.allow_legacy_evaluation,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
