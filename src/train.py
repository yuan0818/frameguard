"""
训练模块：模型训练主入口。

用法:
    python -m src.train                        # 使用默认配置训练
    python -m src.train --config custom.yaml   # 使用自定义配置训练
"""

import os
import sys
import argparse
import time
from typing import Dict, Tuple, Optional
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, get_default_config
from src.model import ImageQualityCNN, create_model, count_parameters
from src.dataset import create_data_loaders, split_manifest


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> optim.Adam:
    """构建 Adam 优化器"""
    return optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: optim.Optimizer,
    config,
) -> optim.lr_scheduler._LRScheduler:
    """
    构建学习率调度器。

    支持类型:
    - reduce_on_plateau: ReduceLROnPlateau
    - step: StepLR
    - cosine: CosineAnnealingLR
    """
    scheduler_type = config.train.lr_scheduler.type

    if scheduler_type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.train.lr_scheduler.factor,
            patience=config.train.lr_scheduler.patience,
            min_lr=config.train.lr_scheduler.min_lr,
        )
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lr_scheduler.step_size,
            gamma=config.train.lr_scheduler.gamma,
        )
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.num_epochs,
            eta_min=config.train.lr_scheduler.min_lr,
        )
    else:
        raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}")


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    filepath: str,
    extra_info: Optional[Dict] = None,
):
    """保存训练检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
    }
    if extra_info:
        checkpoint.update(extra_info)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """加载训练检查点"""
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


# ---------------------------------------------------------------------------
# 训练/验证一个 Epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    grad_clip_norm: float = 0.0,
) -> Tuple[float, float]:
    """
    训练一个 epoch。

    Returns:
        (avg_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 混合精度前向传播
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            # 梯度裁剪
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "acc": f"{100.0 * correct / total:.1f}%",
            })

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    在验证集上评估。

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validating", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss / len(dataloader):.4f}",
            "acc": f"{100.0 * correct / total:.1f}%",
        })

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train(config) -> None:
    """
    执行完整的训练流程。

    Args:
        config: Config 对象
    """
    seed = int(config.data.split.random_seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    # --- 设备 ---
    device = torch.device(getattr(config.train, "device", None) or get_device())
    print(f"\n{'='*60}")
    print(f"图片/视频质量检测 CNN 训练")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 数据路径 ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    normal_dir = os.path.join(project_root, config.data.root_dir, config.data.normal_dir)
    exception_dir = os.path.join(project_root, config.data.root_dir, config.data.exception_dir)

    print(f"数据目录:")
    print(f"  normal:    {normal_dir}")
    print(f"  exception: {exception_dir}")

    # --- 创建 DataLoader ---
    augmentation_config = config.data.augmentation.to_dict() if hasattr(config.data, 'augmentation') else None
    use_letterbox = getattr(config.data, "letterbox", False)

    train_loader, val_loader = create_data_loaders(
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
        augmentation_config=augmentation_config,
        class_weights_mode=config.train.class_weights,
        use_letterbox=use_letterbox,
    )

    if any(s.label >= config.model.num_classes for s in train_loader.dataset.samples + val_loader.dataset.samples):
        raise ValueError("Dataset labels exceed configured num_classes")

    # --- 创建模型 ---
    backbone = getattr(config.model, "backbone", None)
    freeze_backbone = getattr(config.model, "freeze_backbone", True)

    model = create_model(
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        classifier_dropout=config.model.classifier_dropout,
        input_channels=config.model.input_channels,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"\n模型可训练参数量: {num_params:,}")

    # --- 损失函数（带标签平滑） ---
    label_smoothing = getattr(config.train, "label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"标签平滑: {label_smoothing}")

    # --- 优化器 ---
    optimizer = build_optimizer(
        model,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # --- 学习率调度器 ---
    scheduler = build_scheduler(optimizer, config)

    # --- 混合精度训练 ---
    use_amp = getattr(config.train, "use_amp", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print(f"混合精度训练 (AMP): 已启用")

    # --- 梯度裁剪 ---
    grad_clip_norm = getattr(config.train, "grad_clip_norm", 0.0)
    if grad_clip_norm > 0:
        print(f"梯度裁剪: max_norm={grad_clip_norm}")

    # --- 断点续训 ---
    start_epoch = 1
    best_val_acc = float("-inf")
    early_stop_counter = 0
    early_stop_patience = config.train.early_stopping_patience

    checkpoint_dir = os.path.join(project_root, config.output.checkpoint_dir)
    best_model_path = os.path.join(checkpoint_dir, config.output.best_model_name)
    last_model_path = os.path.join(checkpoint_dir, config.output.last_model_name)

    if config.train.resume_checkpoint:
        resume_path = config.train.resume_checkpoint
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(project_root, resume_path)
        print(f"\n从检查点恢复训练: {resume_path}")
        metadata = torch.load(resume_path, map_location="cpu", weights_only=True)
        if metadata.get("split_manifest") != split_manifest(train_loader, val_loader):
            raise ValueError("Resume requires the same recorded source split; retrain legacy weights from scratch")
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"恢复到 Epoch {start_epoch}, 最佳验证准确率: {best_val_acc:.2f}%")

    # --- 训练循环 ---
    num_epochs = config.train.num_epochs
    print(f"\n开始训练: {num_epochs} epochs, batch_size={config.train.batch_size}")
    print(f"学习率: {config.train.learning_rate}, 早停耐心值: {early_stop_patience}")
    if label_smoothing > 0:
        print(f"标签平滑: {label_smoothing}")
    print("-" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch,
            log_interval=config.train.log_interval,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]["lr"]

        # 日志
        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        scheduler_type = config.train.lr_scheduler.type
        scheduler.step(val_acc) if scheduler_type == "reduce_on_plateau" else scheduler.step()
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        metadata = {
            "config": config.to_dict(),
            "split_manifest": split_manifest(train_loader, val_loader),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "early_stop_counter": early_stop_counter,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        }
        save_checkpoint(model, optimizer, epoch, best_val_acc, last_model_path, extra_info=metadata)
        if improved:
            save_checkpoint(model, optimizer, epoch, best_val_acc, best_model_path, extra_info=metadata)
            print(f"  >> 最佳模型已保存 (val_acc: {val_acc:.2f}%)")

        # 早停
        if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
            print(f"\n早停触发！连续 {early_stop_patience} 轮验证准确率未提升。")
            print(f"最佳验证准确率: {best_val_acc:.2f}%")
            break

    # --- 训练完成 ---
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最佳模型路径: {best_model_path}")
    print(f"最后模型路径: {last_model_path}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="图片/视频质量检测 CNN 分类器 - 训练"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认使用 config/default.yaml）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从 checkpoint 恢复训练（覆盖配置文件中的 resume_checkpoint）",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda or mps")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()

    # 覆盖 resume
    if args.resume:
        config.train.resume_checkpoint = args.resume

    # 执行训练
    if args.device:
        config.train.device = args.device
    train(config)


if __name__ == "__main__":
    main()
