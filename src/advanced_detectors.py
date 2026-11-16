"""
高级异常检测模块：静止帧检测、Crash 弹窗检测、元素缺失检测。

功能：
  1. 静止帧检测（相邻帧灰度直方图相关性分析）
  2. Crash 弹窗检测（模板匹配优先 + 规则兜底）
  3. 元素缺失检测（基于模板匹配的 UI 元素完整性校验）
  4. 统一入口 analyze_advanced()

用法：
    from src.advanced_detectors import analyze_advanced

    result = analyze_advanced(image_or_frames, is_video=False)
    print(result.to_dict())
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from src.cv_detectors import DetectorResult

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass
class AdvancedAnalysisResult:
    """高级分析综合结果"""
    still_frames: Optional[DetectorResult] = None
    crash: Optional[DetectorResult] = None
    missing_elements: Optional[DetectorResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "still_frames": self.still_frames.to_dict() if self.still_frames else None,
            "crash": self.crash.to_dict() if self.crash else None,
            "missing_elements": (
                self.missing_elements.to_dict() if self.missing_elements else None
            ),
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _pil_to_rgb(frame: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """将 PIL Image 或 numpy 数组统一转为 RGB numpy 数组。"""
    if isinstance(frame, np.ndarray):
        return frame
    return np.array(frame.convert("RGB"))


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB → 灰度。"""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _calc_hist(gray: np.ndarray) -> np.ndarray:
    """计算灰度直方图（256 bins，范围 0-256）。"""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist


def _list_template_images(template_dir: str) -> List[str]:
    """列出模板目录中所有支持的图片文件路径。"""
    if not os.path.isdir(template_dir):
        return []
    paths: List[str] = []
    for fname in sorted(os.listdir(template_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            paths.append(os.path.join(template_dir, fname))
    return paths


def _safe_imread(path: str) -> Optional[np.ndarray]:
    """安全读取图片，失败返回 None。"""
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 检测器 1：静止帧检测
# ---------------------------------------------------------------------------

def detect_still_frames(
    frames: List[Union[Image.Image, np.ndarray]],
    hist_correlation_threshold: float = 0.95,
    min_still_frames: int = 5,
) -> DetectorResult:
    """
    检测视频帧列表中是否存在连续静止帧。

    原理：逐帧转灰度 → 计算灰度直方图 → 用 cv2.compareHist 计算相邻帧
    直方图相关性，相关性 > threshold 的帧对视为"静止"。统计最长连续
    静止段，若帧数 >= min_still_frames 则判定为异常。

    Args:
        frames: 帧列表（PIL Image 或 numpy 数组）
        hist_correlation_threshold: 直方图相关性阈值，默认 0.95
        min_still_frames: 判定异常所需的最小连续静止帧数，默认 5

    Returns:
        DetectorResult，包含静止帧检测结果
    """
    details: Dict[str, Any] = {
        "max_consecutive_still": 0,
        "still_segments": [],
        "threshold": hist_correlation_threshold,
    }

    # 边界情况：帧数不足
    if frames is None or len(frames) < 2:
        return DetectorResult(
            detector_name="still_frames",
            is_anomaly=False,
            confidence=0.0,
            details=details,
        )

    # 逐帧计算灰度直方图
    histograms: List[np.ndarray] = []
    for f in frames:
        rgb = _pil_to_rgb(f)
        gray = _to_gray(rgb)
        histograms.append(_calc_hist(gray))

    # 计算相邻帧直方图相关性
    correlations: List[float] = []
    for i in range(len(histograms) - 1):
        corr = cv2.compareHist(
            histograms[i], histograms[i + 1], cv2.HISTCMP_CORREL
        )
        correlations.append(float(corr))

    # 标记静止帧对（相关性 > threshold）
    still_flags = [c > hist_correlation_threshold for c in correlations]

    # 统计连续静止段
    segments: List[Dict[str, int]] = []
    max_consecutive = 0
    current_start: Optional[int] = None
    current_count = 0

    for i, flag in enumerate(still_flags):
        if flag:
            if current_start is None:
                current_start = i
            current_count += 1
        else:
            if current_start is not None:
                # 连续静止帧数 = current_count + 1（帧对数 + 1 = 帧数）
                frame_count = current_count + 1
                if frame_count >= 2:
                    segments.append({
                        "start": current_start,
                        "end": current_start + current_count,
                    })
                max_consecutive = max(max_consecutive, frame_count)
                current_start = None
                current_count = 0

    # 处理末尾段
    if current_start is not None:
        frame_count = current_count + 1
        if frame_count >= 2:
            segments.append({
                "start": current_start,
                "end": current_start + current_count,
            })
        max_consecutive = max(max_consecutive, frame_count)

    details["max_consecutive_still"] = max_consecutive
    details["still_segments"] = segments

    is_anomaly = max_consecutive >= min_still_frames
    # 置信度：连续静止帧数越多越确信
    if is_anomaly:
        confidence = min(1.0, max_consecutive / (min_still_frames * 3))
    else:
        confidence = 0.0

    return DetectorResult(
        detector_name="still_frames",
        is_anomaly=is_anomaly,
        confidence=round(confidence, 4),
        details=details,
    )


# ---------------------------------------------------------------------------
# 检测器 2：Crash 弹窗检测
# ---------------------------------------------------------------------------

def _multi_scale_template_match(
    image_bgr: np.ndarray,
    template_bgr: np.ndarray,
    scale_range: tuple = (0.5, 2.0),
    scale_steps: int = 15,
) -> tuple:
    """
    多尺度模板匹配，返回 (最大匹配值, 最佳匹配位置, 最佳缩放比例)。

    Args:
        image_bgr: BGR 格式待搜索图像
        template_bgr: BGR 格式模板图像
        scale_range: 缩放范围 (min_scale, max_scale)
        scale_steps: 缩放步数

    Returns:
        (max_val, max_loc, best_scale)
    """
    gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_tpl = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    max_val = -1.0
    max_loc = (0, 0)
    best_scale = 1.0

    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    for scale in scales:
        h, w = gray_tpl.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w <= 0 or new_h <= 0:
            continue
        if new_w > gray_img.shape[1] or new_h > gray_img.shape[0]:
            continue

        resized = cv2.resize(gray_tpl, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(gray_img, resized, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(result)
        if mv > max_val:
            max_val = mv
            max_loc = ml
            best_scale = scale

    return max_val, max_loc, best_scale


def _rule_based_crash_detect(
    frame_rgb: np.ndarray,
) -> tuple:
    """
    基于规则的 Crash 弹窗兜底检测。

    在画面中心 60% 区域做边缘检测 → 找矩形轮廓 → 检查是否有
    小面积高亮度矩形（疑似弹窗）被大面积低亮度区域（深色背景）包围。

    Args:
        frame_rgb: RGB 格式图像

    Returns:
        (match_score, location_dict_or_None)
    """
    h, w = frame_rgb.shape[:2]
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    # 中心 60% 区域
    cx1 = int(w * 0.2)
    cy1 = int(h * 0.2)
    cx2 = int(w * 0.8)
    cy2 = int(h * 0.8)
    roi = gray[cy1:cy2, cx1:cx2]

    # 边缘检测
    edges = cv2.Canny(roi, 50, 150)

    # 形态学闭操作，连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 找轮廓
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = roi.shape[0] * roi.shape[1]
    best_score = 0.0
    best_loc: Optional[Dict[str, int]] = None

    # 整体画面平均亮度
    overall_mean = float(np.mean(gray))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.01 or area > roi_area * 0.4:
            continue

        # 近似矩形
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) < 4:
            continue

        x, y, rw, rh = cv2.boundingRect(approx)
        aspect = rw / rh if rh > 0 else 0

        # 弹窗一般接近正方形或横向矩形（宽高比 0.5~3.0）
        if aspect < 0.5 or aspect > 3.0:
            continue

        # 弹窗区域亮度
        dialog_roi = roi[y:y + rh, x:x + rw]
        dialog_brightness = float(np.mean(dialog_roi))

        # 弹窗应该比背景亮
        brightness_diff = dialog_brightness - overall_mean
        if brightness_diff < 15:
            continue

        # 评分：亮度差 + 矩形度
        rect_area = rw * rh
        solidity = area / rect_area if rect_area > 0 else 0
        score = min(1.0, (brightness_diff / 100.0) * 0.6 + solidity * 0.4)

        if score > best_score:
            best_score = score
            best_loc = {
                "x": cx1 + x,
                "y": cy1 + y,
                "w": rw,
                "h": rh,
            }

    return best_score, best_loc


def detect_crash_dialog(
    frame_rgb: np.ndarray,
    template_dir: Optional[str] = None,
    match_threshold: float = 0.7,
) -> DetectorResult:
    """
    检测画面中是否存在 Crash 弹窗。

    优先策略：如果 template_dir 存在且包含图片文件，使用多尺度模板匹配。
    兜底策略：无模板时，基于规则检测画面中心区域的高对比度矩形弹窗特征。

    Args:
        frame_rgb: RGB 格式的当前帧图像（numpy 数组）
        template_dir: Crash 弹窗模板图片目录路径，可选
        match_threshold: 匹配阈值，默认 0.7

    Returns:
        DetectorResult，包含 Crash 弹窗检测结果
    """
    details: Dict[str, Any] = {
        "method": "rule",
        "match_score": 0.0,
        "location": None,
    }

    # 优先策略：模板匹配
    template_files = _list_template_images(template_dir) if template_dir else []
    if template_files:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        best_score = -1.0
        best_loc = None

        for tpl_path in template_files:
            tpl = _safe_imread(tpl_path)
            if tpl is None:
                warnings.warn(f"无法读取模板图片，已跳过: {tpl_path}")
                continue
            score, loc, _ = _multi_scale_template_match(frame_bgr, tpl)
            if score > best_score:
                best_score = score
                best_loc = loc

        if best_score >= 0:
            details["method"] = "template"
            details["match_score"] = round(float(best_score), 4)
            if best_loc is not None:
                details["location"] = {"x": int(best_loc[0]), "y": int(best_loc[1])}

            is_anomaly = best_score > match_threshold
            confidence = round(float(best_score), 4)
            return DetectorResult(
                detector_name="crash_dialog",
                is_anomaly=is_anomaly,
                confidence=confidence,
                details=details,
            )

    # 兜底策略：规则检测
    score, loc = _rule_based_crash_detect(frame_rgb)
    details["method"] = "rule"
    details["match_score"] = round(float(score), 4)
    details["location"] = loc

    # 规则检测置信度打折（规则比较粗糙）
    is_anomaly = score > 0.5
    confidence = round(float(score) * 0.6, 4)  # 规则检测降低置信度

    return DetectorResult(
        detector_name="crash_dialog",
        is_anomaly=is_anomaly,
        confidence=confidence,
        details=details,
    )


# ---------------------------------------------------------------------------
# 检测器 3：元素缺失检测
# ---------------------------------------------------------------------------

def detect_missing_elements(
    frame_rgb: np.ndarray,
    template_dir: Optional[str] = None,
    match_threshold: float = 0.6,
) -> DetectorResult:
    """
    基于模板匹配检测画面中是否缺少预期的 UI 元素。

    遍历模板目录中的每个模板图片，对每个模板做 cv2.matchTemplate
    （TM_CCOEFF_NORMED），最大匹配值 < match_threshold 的模板视为
    "缺失元素"。

    Args:
        frame_rgb: RGB 格式的当前帧图像（numpy 数组）
        template_dir: 参考模板图片目录路径，可选
        match_threshold: 匹配阈值，低于此值视为缺失，默认 0.6

    Returns:
        DetectorResult，包含元素缺失检测结果
    """
    details: Dict[str, Any] = {
        "total_templates": 0,
        "missing_templates": [],
        "matched_templates": [],
        "match_scores": {},
    }

    # 模板目录不存在时优雅降级
    if not template_dir or not os.path.isdir(template_dir):
        details["message"] = "template_dir not found, skipped"
        return DetectorResult(
            detector_name="missing_elements",
            is_anomaly=False,
            confidence=0.0,
            details=details,
        )

    template_files = _list_template_images(template_dir)
    if not template_files:
        details["message"] = "template_dir is empty, skipped"
        return DetectorResult(
            detector_name="missing_elements",
            is_anomaly=False,
            confidence=0.0,
            details=details,
        )

    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    details["total_templates"] = len(template_files)

    missing: List[str] = []
    matched: List[str] = []
    scores: Dict[str, float] = {}

    for tpl_path in template_files:
        tpl = _safe_imread(tpl_path)
        if tpl is None:
            warnings.warn(f"无法读取模板图片，已跳过: {tpl_path}")
            continue

        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

        # 模板不能大于搜索图像
        if tpl_gray.shape[0] > frame_gray.shape[0] or tpl_gray.shape[1] > frame_gray.shape[1]:
            warnings.warn(f"模板尺寸大于搜索图像，已跳过: {tpl_path}")
            continue

        result = cv2.matchTemplate(frame_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        fname = os.path.basename(tpl_path)
        scores[fname] = round(float(max_val), 4)

        if max_val < match_threshold:
            missing.append(fname)
        else:
            matched.append(fname)

    details["missing_templates"] = missing
    details["matched_templates"] = matched
    details["match_scores"] = scores

    is_anomaly = len(missing) > 0
    # 置信度：缺失越多越确信
    if is_anomaly:
        confidence = min(1.0, len(missing) / max(1, details["total_templates"]))
    else:
        confidence = 1.0  # 全部匹配，很确信没有缺失

    return DetectorResult(
        detector_name="missing_elements",
        is_anomaly=is_anomaly,
        confidence=round(confidence, 4),
        details=details,
    )


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def analyze_advanced(
    image_or_frames: Union[Image.Image, List[Union[Image.Image, np.ndarray]]],
    is_video: bool = False,
    config: Optional[dict] = None,
) -> AdvancedAnalysisResult:
    """
    高级检测统一入口。

    对视频输入执行静止帧检测；对图片/视频帧执行 Crash 弹窗检测和
    元素缺失检测。

    Args:
        image_or_frames: 单张图片（PIL Image）或视频帧列表（List[PIL Image]）
        is_video: 是否为视频输入
        config: 可选配置字典，支持以下键：
            - hist_correlation_threshold (float): 静止帧直方图相关性阈值
            - min_still_frames (int): 最小连续静止帧数
            - crash_template_dir (str): Crash 弹窗模板目录
            - crash_match_threshold (float): Crash 匹配阈值
            - element_template_dir (str): 元素模板目录
            - element_match_threshold (float): 元素匹配阈值

    Returns:
        AdvancedAnalysisResult，包含所有高级检测结果
    """
    cfg = config or {}
    result = AdvancedAnalysisResult()

    # --- 视频：静止帧检测 ---
    if is_video and isinstance(image_or_frames, list) and len(image_or_frames) >= 2:
        result.still_frames = detect_still_frames(
            frames=image_or_frames,
            hist_correlation_threshold=cfg.get("hist_correlation_threshold", 0.95),
            min_still_frames=cfg.get("min_still_frames", 5),
        )

    # --- 图片/视频帧：Crash 弹窗检测 + 元素缺失检测 ---
    # 选取代表性帧（视频取第一帧，单图直接用）
    if isinstance(image_or_frames, list):
        if len(image_or_frames) == 0:
            return result
        representative = image_or_frames[0]
    else:
        representative = image_or_frames

    frame_rgb = _pil_to_rgb(representative)

    # Crash 弹窗检测
    result.crash = detect_crash_dialog(
        frame_rgb=frame_rgb,
        template_dir=cfg.get("crash_template_dir"),
        match_threshold=cfg.get("crash_match_threshold", 0.7),
    )

    # 元素缺失检测
    result.missing_elements = detect_missing_elements(
        frame_rgb=frame_rgb,
        template_dir=cfg.get("element_template_dir"),
        match_threshold=cfg.get("element_match_threshold", 0.6),
    )

    return result
