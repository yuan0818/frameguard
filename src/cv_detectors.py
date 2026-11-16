# -*- coding: utf-8 -*-
"""
传统 CV 质量检测器模块

提供基于 OpenCV 的图像质量检测功能，作为 CNN 分类的后处理补充。
包含以下检测器：
  - 模糊检测（Laplacian 方差）
  - 曝光异常检测（直方图分析）
  - 截断/溢出检测（边界元素分析）
  - 空白区域检测（连通域分析，支持 letterbox 感知）

所有检测器接受 RGB 格式的 numpy 数组，返回 DetectorResult。
统一入口函数 analyze_cv_quality() 接受 PIL Image，返回 CVAnalysisResult。
"""
from __future__ import annotations

import sys
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class DetectorResult:
    """单个检测器的结果"""
    detector_name: str       # 检测器名称
    is_anomaly: bool         # 是否检测到异常
    confidence: float        # 置信度/严重程度 (0-1)
    details: Dict[str, Any]  # 检测细节

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典"""
        return {
            "detector_name": self.detector_name,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass
class CVAnalysisResult:
    """CV 质量综合分析结果"""
    blur: Optional[DetectorResult] = None
    exposure: Optional[DetectorResult] = None
    truncation: Optional[DetectorResult] = None
    blank_region: Optional[DetectorResult] = None
    overall_quality_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化字典"""
        return {
            "blur": self.blur.to_dict() if self.blur else None,
            "exposure": self.exposure.to_dict() if self.exposure else None,
            "truncation": self.truncation.to_dict() if self.truncation else None,
            "blank_region": self.blank_region.to_dict() if self.blank_region else None,
            "overall_quality_score": self.overall_quality_score,
        }


# ---------------------------------------------------------------------------
# 检测器实现
# ---------------------------------------------------------------------------

def detect_blur(frame_rgb: np.ndarray, threshold: float = 100.0) -> DetectorResult:
    """
    基于 Laplacian 方差的图像模糊检测。

    原理：对灰度图计算 Laplacian 算子响应的方差，方差越低表示图像越模糊。
    当方差低于 threshold 时判定为模糊异常。

    Args:
        frame_rgb: RGB 格式的图像数组 (H, W, 3)
        threshold: Laplacian 方差阈值，低于此值判定为模糊（默认 100.0）

    Returns:
        DetectorResult，confidence 越高表示越模糊
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_anomaly = laplacian_var < threshold
    # 方差越低越模糊，confidence 越高
    confidence = 1.0 - min(laplacian_var / threshold, 1.0) if threshold > 0 else 0.0

    return DetectorResult(
        detector_name="blur",
        is_anomaly=is_anomaly,
        confidence=round(confidence, 4),
        details={
            "laplacian_variance": round(laplacian_var, 4),
            "threshold": threshold,
        },
    )


def detect_exposure(
    frame_rgb: np.ndarray,
    over_brightness: int = 240,
    under_brightness: int = 15,
    over_ratio: float = 0.95,
    under_ratio: float = 0.05,
) -> DetectorResult:
    """
    基于直方图分析的过曝/欠曝检测。

    原理：统计灰度直方图，计算过亮像素和过暗像素的占比。
    - 过曝：亮度 > over_brightness 的像素占比 > over_ratio
    - 欠曝：亮度 < under_brightness 的像素占比 > (1 - under_ratio)，即暗像素超过 95%

    Args:
        frame_rgb: RGB 格式的图像数组 (H, W, 3)
        over_brightness: 过曝亮度阈值（默认 240）
        under_brightness: 欠曝亮度阈值（默认 15）
        over_ratio: 过曝像素占比阈值（默认 0.95）
        under_ratio: 欠曝容忍比例，暗像素占比超过 (1 - under_ratio) 判定欠曝（默认 0.05）

    Returns:
        DetectorResult，type 字段标识 "overexposure"/"underexposure"/"normal"
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    total_pixels = h * w

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # 过曝分析：亮度 > over_brightness 的像素占比
    bright_pixels = float(np.sum(hist[over_brightness + 1:]))
    bright_ratio = bright_pixels / total_pixels

    # 欠曝分析：亮度 < under_brightness 的像素占比
    dark_pixels = float(np.sum(hist[:under_brightness]))
    dark_ratio = dark_pixels / total_pixels

    # 平均亮度
    mean_brightness = float(np.mean(gray))

    # 判定异常类型
    is_over = bright_ratio > over_ratio
    is_under = dark_ratio > (1.0 - under_ratio)

    if is_over:
        exposure_type = "overexposure"
        is_anomaly = True
        confidence = min(bright_ratio, 1.0)
    elif is_under:
        exposure_type = "underexposure"
        is_anomaly = True
        confidence = min(dark_ratio, 1.0)
    else:
        exposure_type = "normal"
        is_anomaly = False
        confidence = 0.0

    return DetectorResult(
        detector_name="exposure",
        is_anomaly=is_anomaly,
        confidence=round(confidence, 4),
        details={
            "type": exposure_type,
            "mean_brightness": round(mean_brightness, 2),
            "bright_ratio": round(bright_ratio, 4),
            "dark_ratio": round(dark_ratio, 4),
        },
    )


def detect_truncation(
    frame_rgb: np.ndarray,
    border_width_ratio: float = 0.05,
    min_area_ratio: float = 0.01,
) -> DetectorResult:
    """
    边界元素截断/溢出检测。

    原理：
    1. 取四角像素估算背景色
    2. 计算全图与背景色的色彩差异 mask
    3. 形态学闭合 + 膨胀 → 找轮廓
    4. 检查是否有 bbox 触及图像四条边界
    5. 触及边界且面积 > min_area_ratio 的元素数量 > 0 则判定截断

    Args:
        frame_rgb: RGB 格式的图像数组 (H, W, 3)
        border_width_ratio: 边界宽度比例，用于判定"触及边界"的边距（默认 0.05）
        min_area_ratio: 最小元素面积占比，低于此值的边界元素忽略（默认 0.01）

    Returns:
        DetectorResult，truncated_elements 为截断元素数量
    """
    h, w = frame_rgb.shape[:2]
    img_area = w * h
    margin_x = int(w * border_width_ratio)
    margin_y = int(h * border_width_ratio)

    # 取四角像素估算背景色（各取 5x5 区域均值，抗噪）
    corner_size = max(3, min(h, w) // 20)
    corners = [
        frame_rgb[:corner_size, :corner_size],
        frame_rgb[:corner_size, -corner_size:],
        frame_rgb[-corner_size:, :corner_size],
        frame_rgb[-corner_size:, -corner_size:],
    ]
    bg_color = np.mean([c.mean(axis=(0, 1)) for c in corners], axis=0).astype(np.float32)

    # 计算与背景色的色彩差异
    diff = np.sqrt(np.sum((frame_rgb.astype(np.float32) - bg_color) ** 2, axis=2))

    # 二值化：差异 > 30 的区域为前景
    _, fg_mask = cv2.threshold(diff.astype(np.float32), 30, 255, cv2.THRESH_BINARY)
    fg_mask = fg_mask.astype(np.uint8)

    # 形态学闭合 + 膨胀，连通断裂区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    # 找轮廓
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    truncated_elements = 0
    edge_bboxes: List[Dict[str, int]] = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area_ratio * img_area:
            continue

        # 检查是否触及四条边界
        touches_left = x <= margin_x
        touches_top = y <= margin_y
        touches_right = (x + cw) >= (w - margin_x)
        touches_bottom = (y + ch) >= (h - margin_y)

        # 至少触及两条不同方向的边界（如左+上、右+下等）才判定为截断
        edge_count = sum([touches_left, touches_top, touches_right, touches_bottom])
        if edge_count >= 2:
            truncated_elements += 1
            edge_bboxes.append({"x": x, "y": y, "w": cw, "h": ch})

    is_anomaly = truncated_elements > 0
    confidence = min(truncated_elements * 0.3, 1.0) if is_anomaly else 0.0

    return DetectorResult(
        detector_name="truncation",
        is_anomaly=is_anomaly,
        confidence=round(confidence, 4),
        details={
            "truncated_elements": truncated_elements,
            "edge_bboxes": edge_bboxes,
        },
    )


def detect_blank_regions(
    frame_rgb: np.ndarray,
    max_area_ratio: float = 0.40,
    std_threshold: float = 10,
    letterbox_aware: bool = True,
) -> DetectorResult:
    """
    空白区域检测（支持 letterbox padding 感知）。

    原理：
    1. 灰度化 → 计算局部标准差图（用 cv2.blur 计算均值和均值的平方差）
    2. 二值化：标准差 < std_threshold 的区域为"近似纯色"
    3. cv2.connectedComponentsWithStats() 分析连通域
    4. 最大连通域面积占比 > max_area_ratio 则判定异常空白
    5. letterbox_aware=True 时，检查图像四边是否有 letterbox padding，排除 padding 区域

    Args:
        frame_rgb: RGB 格式的图像数组 (H, W, 3)
        max_area_ratio: 最大空白区域面积占比阈值（默认 0.40）
        std_threshold: 局部标准差阈值，低于此值视为近似纯色（默认 10）
        letterbox_aware: 是否启用 letterbox padding 感知（默认 True）

    Returns:
        DetectorResult，max_blank_ratio 为最大空白区域占比
    """
    h, w = frame_rgb.shape[:2]
    img_area = w * h
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 计算局部标准差图：std = sqrt(E[X^2] - E[X]^2)
    block_size = max(15, min(h, w) // 20)
    if block_size % 2 == 0:
        block_size += 1

    mean_gray = cv2.blur(gray, (block_size, block_size))
    mean_sq = cv2.blur(gray ** 2, (block_size, block_size))
    variance = np.clip(mean_sq - mean_gray ** 2, 0, None)
    local_std = np.sqrt(variance)

    # 二值化：标准差 < std_threshold 的区域为纯色
    uniform_mask = (local_std < std_threshold).astype(np.uint8) * 255

    # letterbox 感知：检测并排除四边的 letterbox padding
    is_letterbox = False
    if letterbox_aware:
        is_letterbox, uniform_mask = _remove_letterbox_padding(
            frame_rgb, uniform_mask, h, w
        )

    # 连通域分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        uniform_mask, connectivity=8
    )

    max_blank_ratio = 0.0  # type: float
    max_blank_bbox: Dict[str, int] = {}

    # 跳过背景标签 0
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ratio = float(area) / img_area
        if ratio > max_blank_ratio:
            max_blank_ratio = ratio
            max_blank_bbox = {
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
            }

    is_anomaly = bool(max_blank_ratio > max_area_ratio)
    confidence = round(float(min(max_blank_ratio, 1.0)), 4) if is_anomaly else 0.0

    return DetectorResult(
        detector_name="blank_region",
        is_anomaly=is_anomaly,
        confidence=confidence,
        details={
            "max_blank_ratio": round(float(max_blank_ratio), 4),
            "blank_region_bbox": max_blank_bbox,
            "is_letterbox_padding": bool(is_letterbox),
        },
    )


def _remove_letterbox_padding(
    frame_rgb: np.ndarray,
    uniform_mask: np.ndarray,
    h: int,
    w: int,
) -> Tuple[bool, np.ndarray]:
    """
    检测并移除 letterbox padding 区域。

    通过检查图像四角颜色一致性来判断是否存在 letterbox padding。
    如果四角中有 3 个以上颜色接近（差异 < 15），则判定为 letterbox 图像，
    并将 padding 区域从 uniform_mask 中排除。

    Args:
        frame_rgb: RGB 图像数组
        uniform_mask: 当前纯色 mask
        h: 图像高度
        w: 图像宽度

    Returns:
        (is_letterbox, updated_mask) 元组
    """
    corner_size = max(5, min(h, w) // 30)
    corners = [
        frame_rgb[:corner_size, :corner_size],
        frame_rgb[:corner_size, -corner_size:],
        frame_rgb[-corner_size:, :corner_size],
        frame_rgb[-corner_size:, -corner_size:],
    ]
    corner_means = [c.mean(axis=(0, 1)) for c in corners]

    # 计算四角之间的颜色差异
    similar_count = 0
    ref_color = corner_means[0]
    for cm in corner_means[1:]:
        if np.linalg.norm(cm - ref_color) < 15:
            similar_count += 1

    if similar_count < 3:
        return False, uniform_mask

    # 判定为 letterbox，用背景色生成 mask 并从 uniform_mask 中减去
    bg_color = ref_color.astype(np.float32)
    diff = np.sqrt(np.sum((frame_rgb.astype(np.float32) - bg_color) ** 2, axis=2))
    padding_mask = (diff < 20).astype(np.uint8) * 255

    # 从 uniform_mask 中排除 padding 区域
    updated_mask = cv2.bitwise_and(uniform_mask, cv2.bitwise_not(padding_mask))

    return True, updated_mask


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def analyze_cv_quality(
    image: Image.Image,
    config: dict = None,
) -> CVAnalysisResult:
    """
    对图片进行传统 CV 质量检测分析。

    仿照 ui_detector.py 的 analyze_ui_layout() 签名，接受 PIL Image 和可选配置。
    依次调用模糊、曝光、截断、空白区域四个检测器，并计算综合质量评分。

    Args:
        image: PIL Image 对象
        config: 可选的检测器配置字典（从 YAML detectors 节读取），
                为 None 时使用各检测器的默认参数

    Returns:
        CVAnalysisResult，包含各检测器结果和综合质量评分
    """
    frame_rgb = np.array(image.convert("RGB"))

    # 解析各检测器配置
    blur_cfg = (config or {}).get("blur", {})
    exposure_cfg = (config or {}).get("exposure", {})
    truncation_cfg = (config or {}).get("truncation", {})
    blank_cfg = (config or {}).get("blank_region", {})

    blur_enabled = blur_cfg.get("enabled", True)
    exposure_enabled = exposure_cfg.get("enabled", True)
    truncation_enabled = truncation_cfg.get("enabled", True)
    blank_enabled = blank_cfg.get("enabled", True)

    result = CVAnalysisResult()

    # --- 1. 模糊检测 ---
    if blur_enabled:
        result.blur = detect_blur(
            frame_rgb,
            threshold=blur_cfg.get("laplacian_variance_threshold", 100.0),
        )

    # --- 2. 曝光异常检测 ---
    if exposure_enabled:
        result.exposure = detect_exposure(
            frame_rgb,
            over_brightness=exposure_cfg.get("overexposure_brightness", 240),
            under_brightness=exposure_cfg.get("underexposure_brightness", 15),
            over_ratio=exposure_cfg.get("overexposure_ratio", 0.95),
            under_ratio=exposure_cfg.get("underexposure_ratio", 0.05),
        )

    # --- 3. 截断检测 ---
    if truncation_enabled:
        result.truncation = detect_truncation(
            frame_rgb,
            border_width_ratio=truncation_cfg.get("border_width_ratio", 0.05),
            min_area_ratio=truncation_cfg.get("min_edge_element_area_ratio", 0.01),
        )

    # --- 4. 空白区域检测 ---
    if blank_enabled:
        result.blank_region = detect_blank_regions(
            frame_rgb,
            max_area_ratio=blank_cfg.get("max_area_ratio", 0.40),
            std_threshold=blank_cfg.get("color_std_threshold", 10),
        )

    # --- 5. 综合质量评分 ---
    result.overall_quality_score = _compute_quality_score(result)

    return result


def _compute_quality_score(result: CVAnalysisResult) -> float:
    """
    根据各检测器结果计算综合质量评分 (0.0 ~ 1.0)。

    评分策略：
    - 基础分 1.0
    - 每个异常检测器按 confidence 加权扣分
    - 权重：blur=0.25, exposure=0.30, truncation=0.25, blank_region=0.20

    Args:
        result: CVAnalysisResult，各检测器已填充

    Returns:
        综合质量评分，范围 [0.0, 1.0]
    """
    weights = {
        "blur": 0.25,
        "exposure": 0.30,
        "truncation": 0.25,
        "blank_region": 0.20,
    }

    total_penalty = 0.0
    detectors = [
        ("blur", result.blur),
        ("exposure", result.exposure),
        ("truncation", result.truncation),
        ("blank_region", result.blank_region),
    ]

    for name, det_result in detectors:
        if det_result is not None and det_result.is_anomaly:
            total_penalty += weights.get(name, 0.0) * det_result.confidence

    return round(max(1.0 - total_penalty, 0.0), 4)


# ---------------------------------------------------------------------------
# 独立 CLI
# ---------------------------------------------------------------------------

def main():
    """命令行入口

    支持用法：
        python -m src.cv_detectors input.jpg
        python -m src.cv_detectors input.jpg --config config/default.yaml
        python -m src.cv_detectors input_dir/ --output report.json
    """
    parser = argparse.ArgumentParser(
        description="CV 质量检测器 — 分析图像中的模糊/曝光/截断/空白区域等质量问题"
    )
    parser.add_argument("input", type=str, help="输入图片路径或目录")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 报告路径")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML 配置文件路径（读取 detectors 节）")
    args = parser.parse_args()

    # 加载配置
    config = None
    if args.config:
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
            config = full_config.get("detectors", None)
            print(f"已加载配置: {args.config}")
        except ImportError:
            print("警告: 未安装 PyYAML，忽略 --config 参数")
        except Exception as e:
            print(f"警告: 加载配置文件失败: {e}")

    # 收集图片路径
    input_path = args.input
    image_paths = []

    if os.path.isfile(input_path):
        image_paths = [input_path]
    elif os.path.isdir(input_path):
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for f in sorted(os.listdir(input_path)):
            if os.path.splitext(f)[1].lower() in exts:
                image_paths.append(os.path.join(input_path, f))
    else:
        print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)

    if not image_paths:
        print("错误: 未找到可处理的图片文件")
        sys.exit(1)

    # 逐张分析
    all_results = {}
    for path in image_paths:
        print(f"分析: {path}")
        img = Image.open(path).convert("RGB")
        result = analyze_cv_quality(img, config=config)
        all_results[path] = result.to_dict()

        d = result.to_dict()
        parts = []
        if d["blur"] and d["blur"]["is_anomaly"]:
            parts.append(f"模糊(conf={d['blur']['confidence']:.2f})")
        if d["exposure"] and d["exposure"]["is_anomaly"]:
            parts.append(f"{d['exposure']['details']['type']}(conf={d['exposure']['confidence']:.2f})")
        if d["truncation"] and d["truncation"]["is_anomaly"]:
            parts.append(f"截断(元素={d['truncation']['details']['truncated_elements']})")
        if d["blank_region"] and d["blank_region"]["is_anomaly"]:
            parts.append(f"空白(占比={d['blank_region']['details']['max_blank_ratio']:.2%})")

        if parts:
            print(f"  ⚠ 异常: {' | '.join(parts)}")
        else:
            print(f"  ✓ 未检测到异常")
        print(f"  质量评分: {d['overall_quality_score']:.3f}")

    # 输出报告
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n报告保存至: {args.output}")

    # 汇总统计
    total = len(all_results)
    anomaly_count = sum(
        1 for r in all_results.values()
        if any([
            (r.get("blur") or {}).get("is_anomaly", False),
            (r.get("exposure") or {}).get("is_anomaly", False),
            (r.get("truncation") or {}).get("is_anomaly", False),
            (r.get("blank_region") or {}).get("is_anomaly", False),
        ])
    )
    avg_score = np.mean([r["overall_quality_score"] for r in all_results.values()])
    print(f"\n{'=' * 50}")
    print(f"总计分析: {total} 张")
    print(f"检测到异常: {anomaly_count} 张 ({100 * anomaly_count / total:.0f}%)")
    print(f"平均质量评分: {avg_score:.3f}")


if __name__ == "__main__":
    main()
