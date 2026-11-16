"""
CNN 模型定义：用于图片/视频质量多分类的自定义卷积神经网络。

架构（改进版 — 残差连接 + 空间特征保留）：
  Block 1: Conv(3→32)×2 + Residual → MaxPool → 32×112×112
  Block 2: Conv(32→64)×2 + Residual → MaxPool → 64×56×56
  Block 3: Conv(64→128)×2 + Residual → MaxPool → 128×28×28
  Block 4: Conv(128→256)×2 + Residual → MaxPool → 256×14×14
  Block 5: Conv(256→512)×2 + Residual → MaxPool → 512×7×7
  Spatial Reduce: Conv(512→64, 1×1) → 64×7×7
  Dropout → Flatten → Linear(3136→512) → ReLU → Dropout → Linear(512→128) → ReLU → Dropout → Linear(128→num_classes)

支持：
- torchvision 预训练 backbone（ResNet / EfficientNet，可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """
    残差双卷积块：Conv2d → BN → ReLU → Conv2d → BN → + Residual → ReLU → MaxPool

    当 in_channels != out_channels 时，使用 1×1 卷积对齐残差通道。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        # 残差连接：当输入输出通道数不同时，用 1×1 卷积对齐
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = F.relu(x + identity, inplace=True)

        x = self.pool(x)
        return x


class ImageQualityCNN(nn.Module):
    """
    图片/视频质量检测 CNN 分类器 (改进版)。

    改进点（相对于原版）：
    - 残差连接：每个 Block 内部添加 skip connection，改善梯度流动
    - 空间保留：Block5 使用 MaxPool 替代 AdaptiveAvgPool(1,1)，保留 7×7 空间特征
    - 1×1 空间压缩：64 通道的 1×1 卷积降低 FC 参数量的同时保留空间信息
    - Dropout 降低：从 0.5 降至默认 0.3，避免过度正则化
    - 更深分类头：3 层 FC (3136→512→128→7)，增强分类能力

    输入: (B, 3, 224, 224) RGB 图像
    输出: (B, num_classes) 类别 logits

    Args:
        num_classes: 分类类别数（默认 7）
        dropout_rate: 空间特征后的 Dropout 比例
        classifier_dropout: 分类头中的 Dropout 比例
        input_channels: 输入通道数（默认 3，RGB）
    """

    def __init__(
        self,
        num_classes: int = 7,
        dropout_rate: float = 0.3,
        classifier_dropout: float = 0.3,
        input_channels: int = 3,
    ):
        super().__init__()

        # Block 1: 3 → 32, 224 → 112
        self.block1 = ConvBlock(input_channels, 32)

        # Block 2: 32 → 64, 112 → 56
        self.block2 = ConvBlock(32, 64)

        # Block 3: 64 → 128, 56 → 28
        self.block3 = ConvBlock(64, 128)

        # Block 4: 128 → 256, 28 → 14
        self.block4 = ConvBlock(128, 256)

        # Block 5: 256 → 512, 14 → 7
        self.block5 = ConvBlock(256, 512)

        # Dropout after conv blocks
        self.dropout = nn.Dropout(p=dropout_rate)

        # 空间特征压缩：512×7×7 → 64×7×7（1×1 conv 降维，保留空间信息）
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 分类器：64×7×7 = 3136 → 512 → 128 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化卷积层权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 输入图像张量，建议 H=W=224

        Returns:
            (B, num_classes) logits
        """
        x = self.block1(x)          # (B, 32, 112, 112)
        x = self.block2(x)          # (B, 64, 56, 56)
        x = self.block3(x)          # (B, 128, 28, 28)
        x = self.block4(x)          # (B, 256, 14, 14)
        x = self.block5(x)          # (B, 512, 7, 7)

        x = self.dropout(x)
        x = self.spatial_reduce(x)  # (B, 64, 7, 7)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = self.classifier(x)      # (B, num_classes)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别标签。"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别概率。"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# 预训练 Backbone 模型
# ---------------------------------------------------------------------------

# 支持的预训练 backbone 及其特征维度
PRETRAINED_BACKBONE_CONFIG = {
    "resnet18":           {"feat_dim": 512},
    "resnet34":           {"feat_dim": 512},
    "resnet50":           {"feat_dim": 2048},
    "efficientnet_b0":    {"feat_dim": 1280},
}


class PretrainedBackbone(nn.Module):
    """
    基于 torchvision 预训练模型的迁移学习分类器。

    自动替换原模型的分类头，适用于图片质量检测任务。

    支持的 backbone:
        resnet18, resnet34, resnet50, efficientnet_b0

    Args:
        backbone_name: 预训练模型名称
        num_classes: 分类类别数
        dropout_rate: 特征层后的 Dropout 比例
        classifier_dropout: 分类头中的 Dropout 比例
        freeze_backbone: 是否冻结 backbone 权重（仅训练分类头）
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 7,
        dropout_rate: float = 0.3,
        classifier_dropout: float = 0.3,
        freeze_backbone: bool = True,
        imagenet_init: bool = True,
    ):
        super().__init__()

        if backbone_name not in PRETRAINED_BACKBONE_CONFIG:
            raise ValueError(
                f"不支持的 backbone: '{backbone_name}'。"
                f"可选: {list(PRETRAINED_BACKBONE_CONFIG.keys())}"
            )

        cfg = PRETRAINED_BACKBONE_CONFIG[backbone_name]
        feat_dim = cfg["feat_dim"]

        # 延迟导入 torchvision，避免未安装时影响自定义 CNN 的使用
        import torchvision.models as tv_models

        # 获取预训练模型
        backbone_fn = getattr(tv_models, backbone_name)
        try:
            self.backbone = backbone_fn(weights="IMAGENET1K_V1" if imagenet_init else None)
        except Exception:
            # 兼容旧版 torchvision
            self.backbone = backbone_fn(pretrained=imagenet_init)

        # 移除原分类头，保留特征提取部分
        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

        # 冻结 backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 分类头
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes),
        )

        # 仅初始化分类头的权重（backbone 保持预训练权重）
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Kaiming 初始化分类头权重"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self, mode: bool = True):
        """冻结的 backbone 始终保持 eval：否则 BatchNorm 的 running stats 仍会被训练数据更新。"""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------

def create_model(
    num_classes: int = 7,
    dropout_rate: float = 0.3,
    classifier_dropout: float = 0.3,
    input_channels: int = 3,
    backbone: Optional[str] = None,
    freeze_backbone: bool = True,
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    """
    模型工厂函数：创建并可选加载预训练权重。

    Args:
        num_classes: 分类数
        dropout_rate: Dropout 比例
        classifier_dropout: 分类头 Dropout 比例
        input_channels: 输入通道数（仅对自定义 CNN 有效）
        backbone: 预训练 backbone 名称，None 则使用自定义 CNN
        freeze_backbone: 是否冻结 backbone（仅对 PretrainedBackbone 有效）
        pretrained_path: 项目 checkpoint 权重路径（可选）

    Returns:
        ImageQualityCNN 或 PretrainedBackbone 实例
    """
    if backbone:
        model = PretrainedBackbone(
            backbone_name=backbone,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            classifier_dropout=classifier_dropout,
            freeze_backbone=freeze_backbone,
            imagenet_init=pretrained_path is None,
        )
        model_type = f"PretrainedBackbone({backbone})"
    else:
        model = ImageQualityCNN(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            classifier_dropout=classifier_dropout,
            input_channels=input_channels,
        )
        model_type = "ImageQualityCNN"

    # 加载项目 checkpoint（如果提供）
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)

        # 兼容两种保存格式：完整模型 或 state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and any(
            k.startswith("block") or k.startswith("classifier") or
            k.startswith("backbone") or k.startswith("spatial_reduce")
            for k in checkpoint
        ):
            state_dict = checkpoint
        else:
            raise ValueError(f"无法识别的模型权重格式: {pretrained_path}")

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint architecture/classes do not match the configuration. "
                "Use the configuration saved with this checkpoint; legacy bundled weights have 7 classes."
            ) from exc
        print(f"已加载预训练权重: {pretrained_path}")

    print(f"模型类型: {model_type}, 参数量: {count_parameters(model):,}")
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
