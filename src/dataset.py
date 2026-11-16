"""
数据集模块：数据加载、预处理、DataLoader 构建。

功能：
- 扫描 normal/ 目录，收集所有图片和视频，标签为 normal(0)
- 读取 exception/data.csv，按映射收集异常数据
- 视频自动切帧，每帧作为独立样本
- 数据增强（训练集）/ 归一化（验证集）
- 分层划分训练/验证集
"""

import os
import csv
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from .video_utils import (
    is_image_file,
    is_video_file,
    extract_single_frame,
)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Letterbox Resize：保持宽高比，padding 到目标尺寸
# ---------------------------------------------------------------------------
class LetterboxResize:
    """
    保持原始宽高比缩放并居中填充到目标尺寸。

    用于非方形输入（如 UI 截图 9:16）防止强制拉伸破坏空间特征。
    """

    def __init__(self, target_size: Tuple[int, int], fill: Tuple[int, int, int] = (124, 116, 104)):
        self.target_size = target_size  # (H, W)
        self.fill = fill                # padding 颜色（ImageNet 均值对应 RGB）

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        th, tw = self.target_size  # (H, W)

        # 计算缩放比例（按短边 fit）
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)

        # 居中填充
        canvas = Image.new("RGB", (tw, th), self.fill)
        paste_x = (tw - new_w) // 2
        paste_y = (th - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        return canvas


# ---------------------------------------------------------------------------
# 异常类型名称 → 类别 ID 的标准化映射
# 支持中英文常见命名，用户在 CSV 中使用以下任一名称均可识别
# ---------------------------------------------------------------------------
EXCEPTION_TYPE_MAP: Dict[str, int] = {
    # 花屏
    "screen_corruption": 1,
    "花屏": 1,
    "huaping": 1,
    "screen_corrupt": 1,
    "corruption": 1,

    # 黑屏
    "black_screen": 2,
    "黑屏": 2,
    "heiping": 2,
    "blackscreen": 2,
    "black": 2,

    # 绿屏
    "green_screen": 3,
    "绿屏": 3,
    "lvping": 3,
    "greenscreen": 3,
    "green": 3,

    # 块效应
    "blocking_artifact": 4,
    "块效应": 4,
    "kuaiyinxiao": 4,
    "blocking": 4,
    "block_artifact": 4,

    # 马赛克
    "mosaic": 5,
    "马赛克": 5,
    "masaike": 5,

    # 色彩空间出错
    "color_space_error": 6,
    "色彩空间出错": 6,
    "secai_kongjian": 6,
    "colorspace": 6,
    "color_error": 6,

    # --- UI 布局异常 ---
    # 布局错位
    "ui_layout_misalignment": 7,
    "布局错位": 7,
    "layout_misalignment": 7,
    "misalignment": 7,

    # 元素重叠
    "ui_element_overlap": 8,
    "元素重叠": 8,
    "element_overlap": 8,

    # 文本重叠
    "ui_text_overlap": 9,
    "文本重叠": 9,
    "text_overlap": 9,
}

# normal 类别 ID
NORMAL_CLASS_ID = 0


def map_exception_type(exception_type: str) -> int:
    """
    将 CSV 中的异常类型字符串映射为类别 ID。

    匹配规则（按优先级）：
    1. 精确匹配 EXCEPTION_TYPE_MAP
    2. 小写化后精确匹配
    3. 标准化后模糊匹配（去除非字母数字字符、去空格、转小写）

    Args:
        exception_type: CSV 中的异常类型字段值

    Returns:
        类别 ID (1-6)，匹配失败返回 -1

    Raises:
        ValueError: 如果无法匹配到已知类型
    """
    # 1. 精确匹配
    if exception_type in EXCEPTION_TYPE_MAP:
        return EXCEPTION_TYPE_MAP[exception_type]

    # 2. 小写匹配
    lower_key = exception_type.lower().strip()
    if lower_key in EXCEPTION_TYPE_MAP:
        return EXCEPTION_TYPE_MAP[lower_key]

    # 3. 标准化匹配（移除特殊字符、空格）
    normalized = "".join(c.lower() for c in lower_key if c.isalnum())
    for key, value in EXCEPTION_TYPE_MAP.items():
        key_normalized = "".join(c.lower() for c in key if c.isalnum())
        if key_normalized == normalized:
            return value

    raise ValueError(
        f"无法识别的异常类型: '{exception_type}'。\n"
        f"支持的异常类型有: {list(EXCEPTION_TYPE_MAP.keys())}\n"
        f"请在 CSV 中使用上述类型之一，或扩展 src/dataset.py 中的 EXCEPTION_TYPE_MAP。"
    )


# ---------------------------------------------------------------------------
# 样本记录数据结构
# ---------------------------------------------------------------------------
class SampleRecord:
    """
    单个样本的元信息记录。

    Attributes:
        source_type: "image" 或 "video_frame"
        label: 类别 ID (0-6)
        image_path: 图片文件路径（source_type="image" 时有效）
        video_path: 视频文件路径（source_type="video_frame" 时有效）
        frame_index: 帧索引（source_type="video_frame" 时有效）
    """

    __slots__ = ("source_type", "label", "image_path", "video_path", "frame_index", "group_id")

    def __init__(
        self,
        source_type: str,
        label: int,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        frame_index: int = 0,
        group_id: Optional[str] = None,
    ):
        self.source_type = source_type
        self.label = label
        self.image_path = image_path
        self.video_path = video_path
        self.frame_index = frame_index
        self.group_id = group_id


# ---------------------------------------------------------------------------
# 目录扫描函数
# ---------------------------------------------------------------------------

def scan_directory(
    root_dir: str,
    image_exts: List[str],
    video_exts: List[str],
) -> Tuple[List[str], List[str]]:
    """
    递归扫描目录，收集图片和视频文件路径。

    Args:
        root_dir: 要扫描的根目录
        image_exts: 支持的图片扩展名列表
        video_exts: 支持的视频扩展名列表

    Returns:
        (image_paths, video_paths)
    """
    image_paths = []
    video_paths = []

    if not os.path.isdir(root_dir):
        return image_paths, video_paths

    image_ext_set = {ext.lower() for ext in image_exts}
    video_ext_set = {ext.lower() for ext in video_exts}

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 跳过隐藏文件和 CSV 文件
            if filename.startswith(".") or filename.lower().endswith(".csv"):
                continue

            file_path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext in image_ext_set:
                image_paths.append(file_path)
            elif ext in video_ext_set:
                video_paths.append(file_path)

    return image_paths, video_paths


def scan_normal_samples(
    normal_dir: str,
    image_exts: List[str],
    video_exts: List[str],
    video_max_frames: int = 0,
    video_frame_interval: float = 0.0,
) -> List[SampleRecord]:
    """
    扫描 normal 目录，生成所有样本记录。

    Args:
        normal_dir: normal 目录的绝对路径
        image_exts: 支持的图片扩展名
        video_exts: 支持的视频扩展名
        video_max_frames: 单个视频最大帧数
        video_frame_interval: 帧采样间隔（秒）

    Returns:
        SampleRecord 列表
    """
    samples = []

    image_paths, video_paths = scan_directory(normal_dir, image_exts, video_exts)

    # 图片样本
    for img_path in image_paths:
        samples.append(SampleRecord(
            source_type="image",
            label=NORMAL_CLASS_ID,
            image_path=img_path,
        ))

    # 视频样本：每帧作为一个独立样本
    for vid_path in video_paths:
        try:
            from .video_utils import get_video_frame_infos
            frame_infos = get_video_frame_infos(
                vid_path,
                max_frames=video_max_frames,
                frame_interval_sec=video_frame_interval,
            )
            for vp, frame_idx, _ in frame_infos:
                samples.append(SampleRecord(
                    source_type="video_frame",
                    label=NORMAL_CLASS_ID,
                    video_path=vp,
                    frame_index=frame_idx,
                ))
        except Exception as e:
            print(f"警告: 处理视频 {vid_path} 时出错: {e}，跳过。")

    return samples


def scan_exception_samples(
    exception_dir: str,
    csv_filename: str,
    image_exts: List[str],
    video_exts: List[str],
    video_max_frames: int = 0,
    video_frame_interval: float = 0.0,
) -> List[SampleRecord]:
    """
    扫描 exception 目录，读取 CSV 获取标签，生成所有样本记录。

    Args:
        exception_dir: exception 目录的绝对路径
        csv_filename: CSV 文件名（如 "data.csv"）
        image_exts: 支持的图片扩展名
        video_exts: 支持的视频扩展名
        video_max_frames: 单个视频最大帧数
        video_frame_interval: 帧采样间隔（秒）

    Returns:
        SampleRecord 列表
    """
    samples = []

    # 读取 CSV
    csv_path = os.path.join(exception_dir, csv_filename)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"exception 目录下找不到 CSV 文件: {csv_path}\n"
            f"请确保 {exception_dir} 下存在 {csv_filename}，"
            f"格式: file_name,exception_type"
        )

    # 读取 CSV（支持带/不带表头）
    df = pd.read_csv(csv_path)

    # 自动检测表头：如果第一列名为 file_name 或 filename，则有表头
    col0 = df.columns[0].lower().strip()
    has_header = col0 in ("file_name", "filename", "file", "name")
    if not has_header:
        # 无表头，重新读取并指定列名
        df = pd.read_csv(csv_path, header=None, names=["file_name", "exception_type"])

    # 标准化列名
    df.columns = [c.lower().strip() for c in df.columns]
    if "file_name" not in df.columns and "filename" not in df.columns:
        # 尝试使用第一列和第二列
        df = df.rename(columns={df.columns[0]: "file_name", df.columns[1]: "exception_type"})

    if "filename" in df.columns and "file_name" not in df.columns:
        df = df.rename(columns={"filename": "file_name"})

    # 构建文件名到异常类型的映射
    file_label_map: Dict[str, int] = {}
    source_ids = {}
    for _, row in df.iterrows():
        file_name = str(row["file_name"]).strip()
        exception_type = str(row["exception_type"]).strip()

        try:
            class_id = map_exception_type(exception_type)
        except ValueError as e:
            print(f"警告: {e}，跳过文件 {file_name}。")
            continue

        file_label_map[file_name] = class_id
        if "source_id" in row and pd.notna(row["source_id"]):
            source_ids[file_name] = str(row["source_id"])

    # 创建文件名集合用于匹配（支持仅文件名匹配，也支持相对路径匹配）
    file_label_by_basename: Dict[str, int] = {}
    for fname, label in file_label_map.items():
        basename = os.path.basename(fname)
        # 如果 CSV 中的文件名只是 basename，直接映射
        file_label_by_basename[basename] = label

    # 扫描目录中的实际文件
    image_paths, video_paths = scan_directory(exception_dir, image_exts, video_exts)

    # 构建实际文件名到标签的映射
    def find_label(file_path: str) -> Optional[int]:
        basename = os.path.basename(file_path)
        rel_path = os.path.relpath(file_path, exception_dir)

        # 1. 精确匹配完整文件名
        if basename in file_label_by_basename:
            return file_label_by_basename[basename]
        # 2. 精确匹配相对路径
        if rel_path in file_label_map:
            return file_label_map[rel_path]
        # 3. 模糊匹配（去掉扩展名后匹配）
        name_no_ext = os.path.splitext(basename)[0]
        if name_no_ext in file_label_by_basename:
            return file_label_by_basename[name_no_ext]
        # 4. 尝试在 CSV 的 file_name 中查找（CSV 可能包含扩展名也可能不包含）
        for csv_name, label in file_label_map.items():
            csv_no_ext = os.path.splitext(csv_name)[0]
            if csv_no_ext == name_no_ext:
                return label

        return None

    matched_count = 0
    unmatched_files = []

    # 处理图片文件
    for img_path in image_paths:
        label = find_label(img_path)
        if label is not None:
            samples.append(SampleRecord(
                source_type="image",
                label=label,
                image_path=img_path,
                group_id=source_ids.get(os.path.relpath(img_path, exception_dir), source_ids.get(os.path.basename(img_path))),
            ))
            matched_count += 1
        else:
            unmatched_files.append(img_path)

    # 处理视频文件
    for vid_path in video_paths:
        label = find_label(vid_path)
        if label is not None:
            try:
                from .video_utils import get_video_frame_infos
                frame_infos = get_video_frame_infos(
                    vid_path,
                    max_frames=video_max_frames,
                    frame_interval_sec=video_frame_interval,
                )
                for vp, frame_idx, _ in frame_infos:
                    samples.append(SampleRecord(
                        source_type="video_frame",
                        label=label,
                        video_path=vp,
                        frame_index=frame_idx,
                        group_id=source_ids.get(os.path.relpath(vid_path, exception_dir), source_ids.get(os.path.basename(vid_path))),
                    ))
                matched_count += 1
            except Exception as e:
                print(f"警告: 处理视频 {vid_path} 时出错: {e}，跳过。")
        else:
            unmatched_files.append(vid_path)

    # 报告匹配统计
    if unmatched_files:
        print(f"警告: {len(unmatched_files)} 个文件在 CSV 中未找到匹配标签，已跳过:")
        for f in unmatched_files[:10]:
            print(f"  - {f}")
        if len(unmatched_files) > 10:
            print(f"  ... 还有 {len(unmatched_files) - 10} 个")

    print(f"exception 目录: 成功匹配 {matched_count} 个文件（含视频帧拆分）")

    return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ImageQualityDataset(Dataset):
    """
    图片/视频质量分类数据集。

    支持：
    - 图片文件直接加载
    - 视频文件按帧索引实时提取（保持 RGB，不灰度化）
    - 训练/验证模式自动切换数据增强
    """

    def __init__(
        self,
        samples: List[SampleRecord],
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        is_train: bool = True,
        augmentation_config: Optional[Dict] = None,
        use_letterbox: bool = False,
    ):
        """
        Args:
            samples: 样本记录列表
            image_size: 目标图像尺寸 (H, W)
            mean: 归一化均值
            std: 归一化标准差
            is_train: 是否为训练模式（影响数据增强）
            augmentation_config: 数据增强配置字典
            use_letterbox: 是否保持宽高比（用于非方形 UI 截图等）
        """
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train

        # 构建变换
        if is_train and augmentation_config:
            self.transform = self._build_train_transform(
                image_size, mean, std, augmentation_config, use_letterbox
            )
        else:
            self.transform = self._build_val_transform(image_size, mean, std, use_letterbox)

    def _build_train_transform(
        self,
        image_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        aug_config: Dict,
        use_letterbox: bool = False,
    ) -> transforms.Compose:
        """构建训练数据增强管线"""
        aug_list = []

        # 随机裁剪缩放（提升尺度泛化能力）
        rrc_config = aug_config.get("random_resized_crop", {})
        if rrc_config.get("enable", False):
            aug_list.append(transforms.RandomResizedCrop(
                size=image_size,
                scale=tuple(rrc_config.get("scale", [0.8, 1.0])),
                ratio=tuple(rrc_config.get("ratio", [0.9, 1.11])),
            ))
        elif use_letterbox:
            # 保持宽高比，padding 到目标尺寸
            fill_rgb = tuple(int(m * 255) for m in mean)
            aug_list.append(LetterboxResize(image_size, fill=fill_rgb))
        else:
            aug_list.append(transforms.Resize(image_size))

        # 随机水平翻转
        if aug_config.get("random_horizontal_flip", True):
            aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # 随机旋转
        rotation_deg = aug_config.get("random_rotation_degrees", 10)
        if rotation_deg > 0:
            aug_list.append(transforms.RandomRotation(degrees=rotation_deg))

        # 颜色抖动
        cj = aug_config.get("color_jitter", {})
        if cj:
            aug_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.2),
                hue=cj.get("hue", 0.1),
            ))

        aug_list.append(transforms.ToTensor())
        aug_list.append(transforms.Normalize(mean=mean, std=std))

        return transforms.Compose(aug_list)

    def _build_val_transform(
        self,
        image_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        use_letterbox: bool = False,
    ) -> transforms.Compose:
        """构建验证/测试变换管线（仅 Resize + 归一化）"""
        if use_letterbox:
            fill_rgb = tuple(int(m * 255) for m in mean)
            resize_op = LetterboxResize(image_size, fill=fill_rgb)
        else:
            resize_op = transforms.Resize(image_size)
        return transforms.Compose([
            resize_op,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def _load_image(self, record: SampleRecord) -> Image.Image:
        """
        加载样本图片（保持 RGB 三通道，不灰度化）。

        Args:
            record: 样本记录

        Returns:
            PIL RGB Image
        """
        if record.source_type == "image":
            img = Image.open(record.image_path).convert("RGB")
            return img
        elif record.source_type == "video_frame":
            img = extract_single_frame(record.video_path, record.frame_index)
            if img is None:
                # 提取失败时返回一个黑色占位图
                print(f"警告: 提取视频帧失败 video={record.video_path} frame={record.frame_index}")
                raise ValueError(f"Cannot decode video frame: {record.video_path}:{record.frame_index}")
            return img
        else:
            raise ValueError(f"未知的 source_type: {record.source_type}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (image_tensor, label_tensor)
            image_tensor: shape (3, H, W), float32
            label_tensor: scalar, int64
        """
        record = self.samples[idx]

        # 加载 RGB 图片（不灰度化）
        image = self._load_image(record)

        # 应用变换
        tensor = self.transform(image)

        label = torch.tensor(record.label, dtype=torch.long)

        return tensor, label


# ---------------------------------------------------------------------------
# DataLoader 构建
# ---------------------------------------------------------------------------

def source_group(sample: SampleRecord) -> str:
    """Group all frames/known corruption variants with their original source.

    Custom derivations must supply source_id in exception/data.csv, matching
    the normal source stem. Conservative basename collisions stay together.
    """
    if sample.group_id:
        return sample.group_id
    stem = Path(sample.image_path or sample.video_path).stem
    kinds = "|".join(sorted(set(EXCEPTION_TYPE_MAP), key=len, reverse=True))
    return re.sub(r"_(?:" + kinds + r")_v[0-9]+$", "", stem)


def split_by_source(samples, train_ratio=0.8, val_ratio=0.2, seed=42):
    if train_ratio <= 0 or val_ratio <= 0 or not np.isclose(train_ratio + val_ratio, 1):
        raise ValueError("train_ratio and val_ratio must be positive and sum to 1")
    groups = sorted({source_group(s) for s in samples})
    if len(groups) < 2:
        raise ValueError("Need at least two independent sources for train/validation")
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups) - 1, max(1, round(len(groups) * val_ratio)))
    val_groups = set(groups[:n_val])
    train = [s for s in samples if source_group(s) not in val_groups]
    val = [s for s in samples if source_group(s) in val_groups]
    return train, val


def split_manifest(train_loader, val_loader):
    return {"method": "source_group_v1",
            "train_groups": sorted({source_group(s) for s in train_loader.dataset.samples}),
            "val_groups": sorted({source_group(s) for s in val_loader.dataset.samples})}


def create_data_loaders(
    normal_dir: str,
    exception_dir: str,
    csv_filename: str,
    image_exts: List[str],
    video_exts: List[str],
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    train_ratio: float,
    val_ratio: float,
    random_seed: int,
    video_max_frames: int = 0,
    video_frame_interval: float = 0.0,
    augmentation_config: Optional[Dict] = None,
    class_weights_mode: Optional[str] = None,
    use_letterbox: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练和验证 DataLoader。

    Args:
        normal_dir: normal 数据目录绝对路径
        exception_dir: exception 数据目录绝对路径
        csv_filename: exception 目录下的 CSV 文件名
        image_exts: 支持的图片扩展名
        video_exts: 支持的视频扩展名
        image_size: 目标图像尺寸 (H, W)
        mean: 归一化均值
        std: 归一化标准差
        batch_size: 批次大小
        num_workers: DataLoader 工作线程数
        pin_memory: 是否锁页内存
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
        video_max_frames: 单个视频最大帧数
        video_frame_interval: 帧采样间隔
        augmentation_config: 数据增强配置
        class_weights_mode: 类别权重模式 ("balanced" 或 None)

    Returns:
        (train_loader, val_loader, class_idx_to_name)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # 扫描数据
    print("正在扫描 normal 目录...")
    normal_samples = scan_normal_samples(
        normal_dir, image_exts, video_exts,
        video_max_frames=video_max_frames,
        video_frame_interval=video_frame_interval,
    )
    print(f"  normal 样本数: {len(normal_samples)}")

    print("正在扫描 exception 目录...")
    exception_samples = scan_exception_samples(
        exception_dir, csv_filename, image_exts, video_exts,
        video_max_frames=video_max_frames,
        video_frame_interval=video_frame_interval,
    )
    print(f"  exception 样本数: {len(exception_samples)}")

    all_samples = normal_samples + exception_samples

    if len(all_samples) == 0:
        raise ValueError("未找到任何样本数据！请检查 data/normal 和 data/exception 目录。")

    train_samples, val_samples = split_by_source(
        all_samples, train_ratio, val_ratio, random_seed
    )

    # 打印各类别分布
    print("\n数据集分布:")
    print(f"{'类别':<8} {'训练集':<8} {'验证集':<8} {'总计':<8}")
    print("-" * 35)
    all_labels = set()
    for s in all_samples:
        all_labels.add(s.label)
    for label in sorted(all_labels):
        train_count = sum(1 for s in train_samples if s.label == label)
        val_count = sum(1 for s in val_samples if s.label == label)
        total_count = sum(1 for s in all_samples if s.label == label)
        class_name = f"class_{label}"
        print(f"{class_name:<8} {train_count:<8} {val_count:<8} {total_count:<8}")
    print(f"\n总计样本: {len(all_samples)} (训练: {len(train_samples)}, 验证: {len(val_samples)})")

    # 构建 Dataset
    train_dataset = ImageQualityDataset(
        train_samples,
        image_size=image_size,
        mean=mean,
        std=std,
        is_train=True,
        augmentation_config=augmentation_config,
        use_letterbox=use_letterbox,
    )

    val_dataset = ImageQualityDataset(
        val_samples,
        image_size=image_size,
        mean=mean,
        std=std,
        is_train=False,
        use_letterbox=use_letterbox,
    )

    # 构建 DataLoader
    train_sampler = None

    # 类别权重采样
    if class_weights_mode == "balanced":
        # 计算各类别样本权重
        label_counts = defaultdict(int)
        for s in train_samples:
            label_counts[s.label] += 1

        total = len(train_samples)
        num_classes = len(label_counts)
        class_weights = {}
        for label, count in label_counts.items():
            class_weights[label] = total / (num_classes * count)

        sample_weights = [class_weights[s.label] for s in train_samples]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_samples),
            replacement=True,
        )
        print(f"\n已启用 Balanced 采样（类别权重: {dict(class_weights)}）")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
