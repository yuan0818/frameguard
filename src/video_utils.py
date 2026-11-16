"""
视频处理工具模块：视频帧提取、文件类型判断。
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# 支持的视频格式
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


def is_image_file(file_path: str) -> bool:
    """
    判断文件是否为支持的图片格式。

    Args:
        file_path: 文件路径

    Returns:
        True 如果是支持的图片格式
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in IMAGE_EXTENSIONS


def is_video_file(file_path: str) -> bool:
    """
    判断文件是否为支持的视频格式。

    Args:
        file_path: 文件路径

    Returns:
        True 如果是支持的视频格式
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def get_video_info(video_path: str) -> Tuple[float, int, float]:
    """
    获取视频基本信息。

    Args:
        video_path: 视频文件路径

    Returns:
        (fps, total_frames, duration_seconds)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()
    return fps, total_frames, duration


def compute_frame_indices(
    total_frames: int,
    fps: float,
    max_frames: int = 0,
    frame_interval_sec: float = 0.0,
) -> List[int]:
    """
    根据采样策略计算需要提取的帧索引。

    优先级：max_frames > frame_interval_sec > 全部帧

    Args:
        total_frames: 视频总帧数
        fps: 视频帧率
        max_frames: 最大提取帧数（均匀采样），0 表示不限制
        frame_interval_sec: 帧间隔（秒），0 表示提取全部帧

    Returns:
        帧索引列表
    """
    indices = list(range(total_frames))

    # 策略1: 按时间间隔采样
    if frame_interval_sec > 0 and fps > 0:
        step = max(1, int(frame_interval_sec * fps))
        indices = list(range(0, total_frames, step))

    # 策略2: 按最大帧数均匀采样（优先级高于时间间隔）
    if max_frames > 0 and len(indices) > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    return indices


def extract_single_frame(video_path: str, frame_index: int) -> Optional[Image.Image]:
    """
    从视频中提取单帧，返回 RGB PIL Image。

    Args:
        video_path: 视频文件路径
        frame_index: 帧索引（0-based）

    Returns:
        RGB PIL Image 对象，如果提取失败返回 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # OpenCV 读取的是 BGR，转换为 RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def extract_all_frames(
    video_path: str,
    max_frames: int = 0,
    frame_interval_sec: float = 0.0,
) -> List[Image.Image]:
    """
    从视频中提取所有帧（根据采样策略），返回 RGB PIL Image 列表。

    注意：此方法会将所有帧加载到内存，仅适用于短视频。
    对于长视频或大量视频，建议使用 extract_single_frame 逐帧提取。

    Args:
        video_path: 视频文件路径
        max_frames: 最大提取帧数，0 表示不限制
        frame_interval_sec: 帧间隔（秒），0 表示提取全部帧

    Returns:
        RGB PIL Image 列表
    """
    fps, total_frames, _ = get_video_info(video_path)
    indices = compute_frame_indices(total_frames, fps, max_frames, frame_interval_sec)

    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


def get_video_frame_infos(
    video_path: str,
    max_frames: int = 0,
    frame_interval_sec: float = 0.0,
) -> List[Tuple[str, int, float]]:
    """
    获取视频中需要提取的帧信息，不实际读取帧数据。

    用于数据集构建时记录每帧的元信息。

    Args:
        video_path: 视频文件路径
        max_frames: 最大提取帧数
        frame_interval_sec: 帧间隔（秒）

    Returns:
        List of (video_path, frame_index, timestamp_seconds)
    """
    fps, total_frames, _ = get_video_info(video_path)
    indices = compute_frame_indices(total_frames, fps, max_frames, frame_interval_sec)

    frame_infos = []
    for idx in indices:
        timestamp = idx / fps if fps > 0 else 0.0
        frame_infos.append((video_path, idx, timestamp))

    return frame_infos
