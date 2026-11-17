"""
UI 布局异常检测模块：对 UI 截图进行结构化分析，检测布局错位、元素重叠、文本重叠。

功能：
  1. 布局结构分析（主体元素检测、对齐分析、间距分析）
  2. 布局错位检测（元素偏离对齐网格、间距不一致）
  3. 元素重叠检测（主体元素相交分析、重叠区域定位）
  4. 文本重叠检测（文本行区域相交分析）
  5. 综合报告生成（含异常定位可视化）

设计思路（降低误报）：
  - 先检测"主体元素"（卡片、导航栏、按钮等大块结构），排除文本字符级噪声
  - 对齐检测只对同一行/列的成对元素做统计偏差判断
  - 重叠检测过滤包含关系（一个元素完全包含另一个并不是布局异常）
  - 文本检测使用 MSER + 形态学，专门检测小字区域

用法：
    from src.ui_detector import analyze_ui_layout

    result = analyze_ui_layout(image)
    print(result.to_dict())
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =========================================================================
# 常量
# =========================================================================

# 元素检测参数
_MIN_ELEMENT_AREA_RATIO = 0.0008   # 最小元素面积占全图比例（224x400 => ~72px²，约 8x9）
_MAX_ASPECT_RATIO = 12.0           # 最大宽高比
_MIN_ELEMENT_DIM = 6               # 最小元素边长

# 对齐检测参数
_ALIGNMENT_ISSUE_THRESHOLD = 0.25  # 对齐异常检出阈值（问题占比超过此值才标记有错位）

# 重叠检测参数
_MIN_OVERLAP_IOU = 0.05            # 最小重叠 IoU
_MIN_OVERLAP_AREA_RATIO = 0.001    # 最小重叠面积占全图比例

# 文本检测参数
_MIN_TEXT_AREA = 18
_MAX_TEXT_HEIGHT_RATIO = 0.2

# 健康度评分参数
_HEALTH_MIS_PENALTY = 0.10
_HEALTH_OVERLAP_PENALTY = 0.15
_HEALTH_TEXT_PENALTY = 0.20
_HEALTH_MAX_PENALTY = 0.80


# =========================================================================
# 数据结构
# =========================================================================


@dataclass
class BBox:
    """边界框"""
    x: int
    y: int
    w: int
    h: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def iou_with(self, other: "BBox") -> float:
        """IoU"""
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        iw = min(self.right, other.right) - ix
        ih = min(self.bottom, other.bottom) - iy
        if iw <= 0 or ih <= 0:
            return 0.0
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def intersection_area_with(self, other: "BBox") -> int:
        """相交面积"""
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        iw = min(self.right, other.right) - ix
        ih = min(self.bottom, other.bottom) - iy
        if iw <= 0 or ih <= 0:
            return 0
        return iw * ih

    def contains(self, other: "BBox") -> bool:
        """是否完全包含 other"""
        return (self.x <= other.x and self.y <= other.y
                and self.right >= other.right and self.bottom >= other.bottom)

    def is_same_row(self, other: "BBox", tolerance: float = 0.6) -> bool:
        """同行判断：垂直中心在高度容忍范围内"""
        avg_h = (self.h + other.h) / 2
        return abs(self.cy - other.cy) < avg_h * tolerance

    def is_same_col(self, other: "BBox", tolerance: float = 0.6) -> bool:
        """同列判断"""
        avg_w = (self.w + other.w) / 2
        return abs(self.cx - other.cx) < avg_w * tolerance


@dataclass
class DetectedElement:
    """检测到的 UI 元素"""
    bbox: BBox
    element_type: str = "rect"   # "rect" | "icon" | "text_block"
    color: Tuple[int, int, int] = (0, 0, 0)
    confidence: float = 1.0


@dataclass
class AlignmentIssue:
    """对齐问题"""
    element_a: BBox
    element_b: BBox
    issue_type: str    # "horizontal_misalignment" | "vertical_misalignment" | "margin_inconsistency"
    offset: int
    description: str = ""


@dataclass
class OverlapIssue:
    """重叠问题"""
    element_a: BBox
    element_b: BBox
    overlap_area: int
    overlap_bbox: BBox
    iou: float
    description: str = ""


@dataclass
class TextOverlapIssue:
    """文本重叠"""
    text_region_a: BBox
    text_region_b: BBox
    overlap_area: int
    description: str = ""


@dataclass
class UIAnalysisResult:
    """综合分析结果"""
    has_misalignment: bool = False
    has_overlap: bool = False
    has_text_overlap: bool = False
    misalignment_details: List[AlignmentIssue] = field(default_factory=list)
    overlap_details: List[OverlapIssue] = field(default_factory=list)
    text_overlap_details: List[TextOverlapIssue] = field(default_factory=list)
    elements_detected: int = 0
    text_regions_detected: int = 0
    layout_health_score: float = 1.0
    image_size: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        def _bb(b: BBox) -> Dict:
            return {"x": b.x, "y": b.y, "w": b.w, "h": b.h}
        return {
            "has_misalignment": self.has_misalignment,
            "has_overlap": self.has_overlap,
            "has_text_overlap": self.has_text_overlap,
            "elements_detected": self.elements_detected,
            "text_regions_detected": self.text_regions_detected,
            "layout_health_score": self.layout_health_score,
            "image_size": list(self.image_size),
            "misalignment_details": [
                {"issue_type": a.issue_type, "element_a": _bb(a.element_a),
                 "element_b": _bb(a.element_b), "offset": a.offset,
                 "description": a.description}
                for a in self.misalignment_details
            ],
            "overlap_details": [
                {"element_a": _bb(o.element_a), "element_b": _bb(o.element_b),
                 "overlap_area": o.overlap_area, "overlap_bbox": _bb(o.overlap_bbox),
                 "iou": round(o.iou, 3), "description": o.description}
                for o in self.overlap_details
            ],
            "text_overlap_details": [
                {"text_region_a": _bb(t.text_region_a),
                 "text_region_b": _bb(t.text_region_b),
                 "overlap_area": t.overlap_area, "description": t.description}
                for t in self.text_overlap_details
            ],
        }


# =========================================================================
# 元素检测
# =========================================================================


def _detect_main_elements(
    img: np.ndarray,
    min_area: int = 0,
) -> List[DetectedElement]:
    """
    检测 UI 元素——所有可见的结构组件（含子元素）。

    策略：
    - 保留边缘检测和色彩检测两种结果，合并后去重
    - 不去除"包含"关系——卡片内的按钮/标签/图标保留为独立元素
    - 仅移除 IoU > 0.8 的高度重叠的重复框
    """
    h, w = img.shape[:2]
    if min_area == 0:
        min_area = max(int(w * h * _MIN_ELEMENT_AREA_RATIO), 30)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- 方法 A: 边缘检测 ---
    edges = cv2.Canny(gray, 20, 80)
    close_kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(closed, dilate_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edge_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area or cw < _MIN_ELEMENT_DIM or ch < _MIN_ELEMENT_DIM:
            continue
        if cw > w * 0.97 and ch > h * 0.97:
            continue
        aspect = cw / max(ch, 1)
        if aspect > _MAX_ASPECT_RATIO or ch / max(cw, 1) > _MAX_ASPECT_RATIO:
            continue
        edge_boxes.append(BBox(x, y, cw, ch))

    # --- 方法 B: 色彩差异检测（检测纯色区域）---
    corners = img[[0, 0, h-1, h-1], [0, w-1, 0, w-1]]
    bg_color = np.median(corners.reshape(-1, 3), axis=0)
    diff = np.max(np.abs(img.astype(np.int16) - bg_color.astype(np.int16)), axis=2).astype(np.uint8)

    _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    fg_kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, fg_kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, fg_kernel)

    contours2, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_boxes = []
    for cnt in contours2:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw * ch < min_area or cw < _MIN_ELEMENT_DIM or ch < _MIN_ELEMENT_DIM:
            continue
        if cw > w * 0.97 and ch > h * 0.97:
            continue
        color_boxes.append(BBox(x, y, cw, ch))

    # --- 合并去重（仅排除 IoU > 0.8 的高度重叠）---
    all_boxes = edge_boxes + color_boxes
    all_boxes.sort(key=lambda b: b.area, reverse=True)

    unique = []
    for b in all_boxes:
        is_dup = False
        for u in unique:
            iou = b.iou_with(u)
            if iou > 0.8:  # 只有几乎完全重叠才去重
                is_dup = True
                break
        if not is_dup:
            unique.append(b)

    # 转为 DetectedElement
    elements = []
    for b in unique:
        elem_type = "rect"
        aspect = b.w / max(b.h, 1)
        if b.w < 30 and b.h < 30:
            elem_type = "icon"
        elif aspect > 3 and b.h < 25:
            elem_type = "text_block"

        region = img[b.y:b.y + b.h, b.x:b.x + b.w]
        avg_color = tuple(int(c) for c in np.mean(region, axis=(0, 1))) if region.size > 0 else (128, 128, 128)
        elements.append(DetectedElement(bbox=b, element_type=elem_type, color=avg_color))

    elements.sort(key=lambda e: e.bbox.area, reverse=True)
    return elements


# =========================================================================
# 文本区域检测
# =========================================================================


def _detect_text_regions(
    img: np.ndarray,
    min_text_area: int = 0,
    return_unmerged: bool = False,
) -> List[BBox]:
    """
    检测 UI 中所有文本行/块区域。

    结合形态学方法和 MSER 两种策略：

    策略 A（形态学）:
      - 灰度化 → Sobel 垂直边缘（文本边缘密集） → 阈值 → 水平膨胀连接字符

    策略 B（MSER）:
      - MSER 检测稳定极值区域 → 按形状/面积过滤 → 只保留文本尺度区域

    合并两种结果后去重。

    Args:
        img: RGB numpy array
        min_text_area: 最小文本面积

    Returns:
        文本区域 BBox 列表
    """
    h, w = img.shape[:2]
    if min_text_area == 0:
        min_text_area = _MIN_TEXT_AREA

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ============= 策略 A: 形态学文本检测 =============
    # 1. Sobel 垂直梯度（文本笔画产生密集垂直边缘）
    sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, binary_x = cv2.threshold(sobel_x, 30, 255, cv2.THRESH_BINARY)

    # 2. 水平膨胀连接同一行字符
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    morphed = cv2.morphologyEx(binary_x, cv2.MORPH_CLOSE, h_kernel, iterations=2)

    # 3. 垂直膨胀连接多行文本
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, v_kernel, iterations=1)

    # 4. 查找轮廓
    contours_a, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions_a = []
    for cnt in contours_a:
        x, y, tw, th = cv2.boundingRect(cnt)
        area = tw * th
        if area < min_text_area or th > h * _MAX_TEXT_HEIGHT_RATIO or th < 4:
            continue
        if tw / max(th, 1) > 18:
            continue
        regions_a.append(BBox(x, y, tw, th))

    # ============= 策略 B: MSER 文本检测 =============
    try:
        mser = cv2.MSER_create()
        regions_mser, _ = mser.detectRegions(gray)
        regions_b = []
        for region in regions_mser:
            if len(region) < 3:
                continue
            x, y, tw, th = cv2.boundingRect(region)
            area = tw * th
            if area < min_text_area or th > h * _MAX_TEXT_HEIGHT_RATIO:
                continue
            # 按 MSER 结果的包络矩形合并
            regions_b.append(BBox(x, y, tw, th))
    except Exception:
        regions_b = []

    # ============= 合并 =============
    all_regions = regions_a + regions_b

    # 如果需要未合并的原始区域，先返回
    if return_unmerged:
        # 仅去重（不合并）
        unique_raw = []
        for r in sorted(all_regions, key=lambda x: x.area, reverse=True):
            is_dup = False
            for u in unique_raw:
                if r.iou_with(u) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique_raw.append(r)
        return unique_raw

    # 去重（IoU > 0.6 合并）
    unique = []
    for r in sorted(all_regions, key=lambda x: x.area, reverse=True):
        is_dup = False
        for u in unique:
            if r.iou_with(u) > 0.6 or u.contains(r):
                is_dup = True
                break
        if not is_dup:
            unique.append(r)

    # 合并邻近文本区域（同一行或相邻行的文本）
    unique.sort(key=lambda r: (r.y, r.x))
    merged = _merge_nearby_regions(unique, dx_threshold=8, dy_threshold=6)

    return merged


def _merge_nearby_regions(
    regions: List[BBox],
    dx_threshold: int = 8,
    dy_threshold: int = 6,
) -> List[BBox]:
    """合并邻近的文本区域"""
    if not regions:
        return []

    merged = []
    used = [False] * len(regions)

    for i, a in enumerate(regions):
        if used[i]:
            continue
        combined = BBox(a.x, a.y, a.w, a.h)
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j, b in enumerate(regions):
                if used[j]:
                    continue
                y_overlap = (combined.y < b.bottom and combined.bottom > b.y)
                x_gap = max(0, b.x - combined.right) if b.x > combined.x else max(0, combined.x - b.right)
                if y_overlap and x_gap < dx_threshold:
                    nx = min(combined.x, b.x)
                    ny = min(combined.y, b.y)
                    nr = max(combined.right, b.right)
                    nb = max(combined.bottom, b.bottom)
                    combined = BBox(nx, ny, nr - nx, nb - ny)
                    used[j] = True
                    changed = True
            # 额外检查：y 距离接近也合并
            if not changed:
                for j, b in enumerate(regions):
                    if used[j]:
                        continue
                    if abs(combined.cy - b.cy) < dy_threshold:
                        nx = min(combined.x, b.x)
                        ny = min(combined.y, b.y)
                        nr = max(combined.right, b.right)
                        nb = max(combined.bottom, b.bottom)
                        combined = BBox(nx, ny, nr - nx, nb - ny)
                        used[j] = True
                        changed = True
                        break

        merged.append(combined)

    return merged


# =========================================================================
# 对齐异常检测
# =========================================================================


def _detect_alignment_issues(
    elements: List[DetectedElement],
    tolerance: int = 10,
    image_width: int = 0,
    image_height: int = 0,
) -> Tuple[List[AlignmentIssue], bool]:
    """
    检测对齐异常。

    策略：
    1. 只检测面积 > 100px² 的元素
    2. 按尺寸分层分组（避免将背景元素和图标对比）
    3. 同行/同列内用平均值检测偏移

    Returns:
        (issues_list, has_significant_issues)
    """
    issues = []

    # 保留面积 > 100px² 的元素
    all_boxes = [
        e.bbox for e in elements
        if e.bbox.area > 100
    ]

    if len(all_boxes) < 2:
        return [], False

    # 按面积分层：将面积在同一数量级的元素放一起比较
    # 分 3 层：大 (top 30%)、中 (middle 40%)、小 (bottom 30%)
    sorted_boxes = sorted(all_boxes, key=lambda b: b.area)
    n = len(sorted_boxes)
    if n >= 6:
        split1 = n // 3
        split2 = 2 * n // 3
        layers = [sorted_boxes[:split1], sorted_boxes[split1:split2], sorted_boxes[split2:]]
    else:
        layers = [sorted_boxes]  # 不足 6 个不分层

    total_pairs = 0
    misaligned_pairs = 0

    # 对每层分别做对齐检测
    for main_boxes in layers:
        if len(main_boxes) < 2:
            continue

        # === 1. 同行水平对齐 ===
        row_groups = _group_by_row(main_boxes, tolerance=0.5)
        for row in row_groups:
            if len(row) < 2:
                continue
            row.sort(key=lambda r: r.x)

            for attr_name, attr_field in [("顶部(y)", "y"), ("底部(bottom)", "bottom"),
                                           ("垂直中心(cy)", "cy")]:
                vals = [getattr(b, attr_field) for b in row]
                mean_v = float(np.mean(vals))
                max_dev = max(abs(v - mean_v) for v in vals)
                if max_dev > tolerance:
                    worst = max(range(len(row)), key=lambda i: abs(vals[i] - mean_v))
                    b = row[worst]
                    diff = abs(vals[worst] - mean_v)
                    total_pairs += 1
                    if diff > tolerance:
                        misaligned_pairs += 1
                        issues.append(AlignmentIssue(
                            element_a=b, element_b=row[0] if len(row) > 1 else row[(worst+1)%len(row)],
                            issue_type="horizontal_misalignment",
                            offset=int(diff),
                            description=f"元素 {attr_name} 偏移 {diff:.0f}px (行均值 {mean_v:.0f})"
                        ))

        # === 2. 同列垂直对齐 ===
        col_groups = _group_by_col(main_boxes, tolerance=0.5)
        for col in col_groups:
            if len(col) < 2:
                continue
            col.sort(key=lambda r: r.y)
            vals_x = [b.cx for b in col]
            mean_x = float(np.mean(vals_x))
            max_dev_x = max(abs(v - mean_x) for v in vals_x)
            if max_dev_x > tolerance:
                worst = max(range(len(col)), key=lambda i: abs(vals_x[i] - mean_x))
                b = col[worst]
                diff = abs(vals_x[worst] - mean_x)
                total_pairs += 1
                if diff > tolerance:
                    misaligned_pairs += 1
                    issues.append(AlignmentIssue(
                        element_a=b, element_b=col[0] if len(col) > 1 else col[(worst+1)%len(col)],
                        issue_type="vertical_misalignment",
                        offset=int(diff),
                        description=f"元素水平中心偏移 {diff:.0f}px (列均值 {mean_x:.0f})"
                    ))

        # === 3. 水平间距一致性 ===
        for row in row_groups:
            if len(row) < 3:
                continue
            row.sort(key=lambda r: r.x)
            gaps = [row[k + 1].x - row[k].right for k in range(len(row) - 1)]
            if gaps:
                mean_gap = float(np.mean(gaps))
                for k, gap in enumerate(gaps):
                    total_pairs += 1
                    if abs(gap - mean_gap) > max(tolerance, 8):
                        misaligned_pairs += 1
                        issues.append(AlignmentIssue(
                            element_a=row[k + 1], element_b=row[k],
                            issue_type="margin_inconsistency",
                            offset=int(abs(gap - mean_gap)),
                            description=f"水平间距 {gap}px 偏离均值 {mean_gap:.0f}px"
                        ))

        # === 4. 垂直间距一致性 ===
        for col in col_groups:
            if len(col) < 3:
                continue
            col.sort(key=lambda r: r.y)
            gaps = [col[k + 1].y - col[k].bottom for k in range(len(col) - 1)]
            if gaps:
                mean_gap = float(np.mean(gaps))
                for k, gap in enumerate(gaps):
                    total_pairs += 1
                    if abs(gap - mean_gap) > max(tolerance, 8):
                        misaligned_pairs += 1
                        issues.append(AlignmentIssue(
                            element_a=col[k + 1], element_b=col[k],
                            issue_type="margin_inconsistency",
                            offset=int(abs(gap - mean_gap)),
                            description=f"垂直间距 {gap}px 偏离均值 {mean_gap:.0f}px"
                        ))

    # 判断是否有显著问题
    # 仅当存在多处大偏移时才标记
    has_significant = False
    if issues:
        large_offsets = sum(1 for i in issues if i.offset > tolerance + 8)
        has_significant = large_offsets >= 2

    return issues, has_significant


def _group_by_row(boxes: List[BBox], tolerance: float = 0.7) -> List[List[BBox]]:
    """将边界框按行分组"""
    groups = []
    used = [False] * len(boxes)
    for i, a in enumerate(boxes):
        if used[i]:
            continue
        row = [a]
        used[i] = True
        for j, b in enumerate(boxes):
            if used[j]:
                continue
            if a.is_same_row(b, tolerance):
                row.append(b)
                used[j] = True
        if len(row) > 1:
            groups.append(row)
        else:
            # 单元素行也保留，但后续只用列分组
            groups.append(row)
    return groups


def _group_by_col(boxes: List[BBox], tolerance: float = 0.7) -> List[List[BBox]]:
    """将边界框按列分组"""
    groups = []
    used = [False] * len(boxes)
    for i, a in enumerate(boxes):
        if used[i]:
            continue
        col = [a]
        used[i] = True
        for j, b in enumerate(boxes):
            if used[j]:
                continue
            if a.is_same_col(b, tolerance):
                col.append(b)
                used[j] = True
        if len(col) > 1:
            groups.append(col)
    return groups


# =========================================================================
# 元素重叠检测
# =========================================================================


def _detect_element_overlaps(
    elements: List[DetectedElement],
    min_iou: float = 0.0,
    image_area: int = 0,
) -> Tuple[List[OverlapIssue], bool]:
    """
    检测元素重叠。

    改进策略：
    1. 过滤包含关系（A 完全包含 B 不是重叠异常）
    2. 过滤相邻布局区段重叠（如 header 与 tab bar 设计上相邻）
       → 两个元素宽度接近全宽且 y 方向相邻 → 忽略
    3. 区分"覆盖"（小元素被大元素遮挡）和"相交"
    """
    issues = []
    if len(elements) < 2:
        return [], False

    if min_iou == 0:
        min_iou = _MIN_OVERLAP_IOU
    min_overlap_area = max(int(image_area * _MIN_OVERLAP_AREA_RATIO), 30)

    boxes = [e.bbox for e in elements]
    img_w = max(b.right for b in boxes) if boxes else 0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]

            # 过滤包含关系（一个完全在另一个里面）
            if a.contains(b) or b.contains(a):
                continue

            # 过滤相邻布局区段的重叠（header 与 tab、card 与 list 等）
            # 特征：两个元素宽度相同且都接近全宽 → 垂直堆叠的布局区段
            both_wide = (a.w > img_w * 0.7 and b.w > img_w * 0.7)
            same_h_position = (abs(a.x - b.x) < 5 and abs(a.w - b.w) < 5)
            if both_wide and same_h_position:
                continue  # 垂直堆叠的相邻布局区段，不视为重叠

            iou = a.iou_with(b)
            if iou < min_iou:
                continue

            ix = max(a.x, b.x)
            iy = max(a.y, b.y)
            iw = min(a.right, b.right) - ix
            ih = min(a.bottom, b.bottom) - iy
            overlap_area = iw * ih if iw > 0 and ih > 0 else 0

            if overlap_area < min_overlap_area:
                continue

            # 判断严重程度
            smaller = a if a.area <= b.area else b
            larger = b if a.area <= b.area else a
            cover_ratio = overlap_area / smaller.area

            if cover_ratio > 0.7:
                desc = (f"小元素({smaller.w}x{smaller.h}) "
                        f"被大元素({larger.w}x{larger.h}) "
                        f"覆盖 {cover_ratio:.0%}")
            else:
                desc = (f"元素({a.w}x{a.h}) 与 ({b.w}x{b.h}) "
                        f"相交 {overlap_area}px² IoU={iou:.2f}")

            issues.append(OverlapIssue(
                element_a=a, element_b=b,
                overlap_area=overlap_area,
                overlap_bbox=BBox(ix, iy, iw, ih),
                iou=iou, description=desc,
            ))

    issues.sort(key=lambda o: o.overlap_area, reverse=True)
    has_significant = len(issues) > 0 and issues[0].overlap_area > max(min_overlap_area * 3, 100)
    return issues, has_significant


# =========================================================================
# 文本重叠检测
# =========================================================================


def _detect_text_overlaps(
    text_regions: List[BBox],
    unmerged_regions: List[BBox],
    min_overlap: int = 15,
    image_area: int = 0,
) -> Tuple[List[TextOverlapIssue], bool]:
    """
    检测文本重叠。

    策略：
    1. 使用未合并的原始文本区域（unmerged_regions）做 BBox 重叠检测
       — 这些区域是 MSER 和 Sobel 文本的原始结果（每个字符/单词片段）
    2. 两两检查重叠，寻找跨文本行的覆盖

    Args:
        text_regions: 合并后的完整文本行区域
        unmerged_regions: 未合并的原始文本片段
        min_overlap: 最小重叠面积
        image_area: 图像总面积（用于自适应阈值）

    Returns:
        (issues_list, has_significant)
    """
    issues = []

    if image_area > 0 and min_overlap == 15:
        min_overlap = max(int(image_area * 0.0005), 15)

    # 使用合并后的文本区块做重叠检测
    # 注意：对合成数据（文字直接覆盖渲染）效果有限，
    # 因为渲染会覆盖而非叠加像素，且邻近文本会被合并为同一区域。
    # 对真实 UI 截图（文本区域有明显边界框）效果较好。
    regions = text_regions if len(text_regions) > 1 else unmerged_regions

    if len(regions) < 2:
        return [], False

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            a, b = regions[i], regions[j]
            if a.contains(b) or b.contains(a):
                continue
            overlap = a.intersection_area_with(b)
            if overlap < min_overlap:
                continue
            issues.append(TextOverlapIssue(
                text_region_a=a, text_region_b=b,
                overlap_area=overlap,
                description=(f"文本片段({a.w}x{a.h}) 与 ({b.w}x{b.h}) "
                            f"重叠 {overlap}px²"),
            ))

    issues.sort(key=lambda o: o.overlap_area, reverse=True)
    has_significant = len(issues) > 0
    return issues, has_significant


# =========================================================================
# 健康度评分
# =========================================================================


def _compute_health_score(
    has_misalignment: bool,
    has_overlap: bool,
    has_text_overlap: bool,
    mis_count: int,
    overlap_count: int,
    text_count: int,
) -> float:
    """
    布局健康度评分 (0.0 - 1.0)。

    规则：
    - 起始 1.0，每类异常按 severity 扣分
    - 多处异常累加扣分（有上限）
    - 多类异常叠加扣分
    """
    penalty = 0.0

    if has_misalignment:
        penalty += min(mis_count * 0.06, 0.30)

    if has_overlap:
        penalty += min(overlap_count * 0.08, 0.35)

    if has_text_overlap:
        penalty += min(text_count * 0.10, 0.40)

    # 多类异常同时存在时额外扣分
    anomaly_types = sum([has_misalignment, has_overlap, has_text_overlap])
    if anomaly_types >= 2:
        penalty += 0.05
    if anomaly_types >= 3:
        penalty += 0.10

    penalty = min(penalty, _HEALTH_MAX_PENALTY)
    return round(max(0.0, 1.0 - penalty), 4)


# =========================================================================
# 主分析入口
# =========================================================================


def analyze_ui_layout(
    image: Image.Image,
    enable_misalignment: bool = True,
    enable_overlap: bool = True,
    enable_text_overlap: bool = True,
    tolerance: int = 10,
    min_overlap_iou: float = 0.05,
) -> UIAnalysisResult:
    """
    对 UI 截图进行完整的布局异常分析。

    流程：
    1. 主体元素检测（边缘检测 + 色彩检测，合并去重）
    2. 文本区域检测（Sobel + MSER 双策略）
    3. 对齐异常分析（按尺寸分层 + 统计偏差）
    4. 元素重叠分析（过滤布局相邻区段）
    5. 文本重叠分析（合并文本区域 BBox 相交）
    6. 综合健康度评分

    Args:
        image: PIL RGB 图像
        enable_misalignment: 是否检测布局错位
        enable_overlap: 是否检测元素重叠
        enable_text_overlap: 是否检测文本重叠
        tolerance: 对齐检测容差（像素），越大越宽松（默认 10）
        min_overlap_iou: 最小重叠 IoU 阈值（默认 0.05）

    Returns:
        UIAnalysisResult
    """
    img_rgb = np.array(image.convert("RGB"))
    h, w = img_rgb.shape[:2]
    img_area = w * h

    result = UIAnalysisResult(image_size=(w, h))

    # --- 1. 主体元素检测 ---
    elements = _detect_main_elements(img_rgb)
    result.elements_detected = len(elements)

    # --- 2. 文本区域检测 ---
    text_regions = _detect_text_regions(img_rgb)
    unmerged_text_regions = _detect_text_regions(img_rgb, return_unmerged=True)
    result.text_regions_detected = len(text_regions)

    # --- 3. 布局错位分析 ---
    if enable_misalignment and len(elements) >= 2:
        align_issues, has_align = _detect_alignment_issues(
            elements, tolerance=tolerance,
            image_width=w, image_height=h,
        )
        result.misalignment_details = align_issues
        result.has_misalignment = has_align

    # --- 4. 元素重叠分析 ---
    if enable_overlap and len(elements) >= 2:
        overlap_issues, has_overlap = _detect_element_overlaps(
            elements, min_iou=min_overlap_iou,
            image_area=img_area,
        )
        result.overlap_details = overlap_issues
        result.has_overlap = has_overlap

    # --- 5. 文本重叠分析 ---
    if enable_text_overlap and len(unmerged_text_regions) >= 2:
        txt_issues, has_txt = _detect_text_overlaps(
            text_regions, unmerged_text_regions,
            image_area=img_area,
        )
        result.text_overlap_details = txt_issues
        result.has_text_overlap = has_txt

    # --- 6. 健康度评分 ---
    result.layout_health_score = _compute_health_score(
        has_misalignment=result.has_misalignment,
        has_overlap=result.has_overlap,
        has_text_overlap=result.has_text_overlap,
        mis_count=len(result.misalignment_details),
        overlap_count=len(result.overlap_details),
        text_count=len(result.text_overlap_details),
    )

    return result


# =========================================================================
# 可视化
# =========================================================================


def draw_detection_result(
    image: Image.Image,
    result: UIAnalysisResult,
) -> Image.Image:
    """
    绘制检测结果标注图。

    颜色约定：
    - 红色半透明 = 元素重叠区域
    - 橙色边框 = 错位元素
    - 紫色高亮 = 文本重叠区域
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    # 重叠区域（红色）
    for od in result.overlap_details:
        ob = od.overlap_bbox
        draw.rectangle(
            [ob.x, ob.y, ob.right, ob.bottom],
            fill=(255, 0, 0, 60), outline=(255, 0, 0, 200), width=2,
        )
        draw.text((ob.x, max(0, ob.y - 12)),
                  f"重叠 IoU={od.iou:.2f}", fill=(255, 0, 0))

    # 错位元素（橙色）
    for ad in result.misalignment_details:
        a = ad.element_a
        draw.rectangle(
            [a.x, a.y, a.right, a.bottom],
            outline=(255, 165, 0, 200), width=2,
        )
        draw.text((a.x, max(0, a.y - 12)),
                  f"偏移 {ad.offset}px", fill=(255, 165, 0))

    # 文本重叠（紫色）
    for td in result.text_overlap_details:
        a, b = td.text_region_a, td.text_region_b
        ix = max(a.x, b.x); iy = max(a.y, b.y)
        ir = min(a.right, b.right); ib = min(a.bottom, b.bottom)
        draw.rectangle(
            [ix, iy, ir, ib],
            fill=(128, 0, 128, 80), outline=(128, 0, 128, 200), width=2,
        )
        draw.text((ix, max(0, iy - 12)), f"文本重叠", fill=(128, 0, 128))

    # 健康度标签
    score = result.layout_health_score
    color = (0, 200, 0) if score >= 0.8 else ((255, 165, 0) if score >= 0.5 else (255, 0, 0))
    draw.text((5, 5), f"健康度: {score:.2f}", fill=color)

    if result.has_misalignment:
        draw.text((5, 20), f"布局错位: {len(result.misalignment_details)} 处", fill=(255, 165, 0))
    if result.has_overlap:
        draw.text((5, 35), f"元素重叠: {len(result.overlap_details)} 处", fill=(255, 0, 0))
    if result.has_text_overlap:
        draw.text((5, 50), f"文本重叠: {len(result.text_overlap_details)} 处", fill=(128, 0, 128))

    return img


# =========================================================================
# 批量分析
# =========================================================================


def analyze_ui_layout_batch(
    image_paths: List[str],
    **kwargs,
) -> Dict[str, UIAnalysisResult]:
    """批量分析"""
    results = {}
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            results[path] = analyze_ui_layout(img, **kwargs)
        except Exception as e:
            print(f"警告: 分析 {path} 失败: {e}")
    return results


# =========================================================================
# CLI 入口
# =========================================================================


def main():
    """命令行入口"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="UI 布局异常检测 — 分析 UI 截图中的布局问题"
    )
    parser.add_argument("input", type=str, help="输入图片路径或目录")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 JSON 报告路径")
    parser.add_argument("--visualize", "-v", type=str, default=None, help="可视化结果保存路径/目录")
    parser.add_argument("--tolerance", "-t", type=int, default=10, help="对齐检测容差（默认 10）")
    parser.add_argument("--min-iou", type=float, default=0.05, help="最小重叠 IoU（默认 0.05）")
    parser.add_argument("--no-misalignment", action="store_true", help="跳过错位检测")
    parser.add_argument("--no-overlap", action="store_true", help="跳过重叠检测")
    parser.add_argument("--no-text-overlap", action="store_true", help="跳过文本重叠检测")
    args = parser.parse_args()

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

    all_results = {}
    for path in image_paths:
        print(f"分析: {path}")
        img = Image.open(path).convert("RGB")
        result = analyze_ui_layout(
            img,
            enable_misalignment=not args.no_misalignment,
            enable_overlap=not args.no_overlap,
            enable_text_overlap=not args.no_text_overlap,
            tolerance=args.tolerance,
            min_overlap_iou=args.min_iou,
        )
        all_results[path] = result.to_dict()
        d = result.to_dict()
        print(f"  主体元素: {d['elements_detected']} | 文本区域: {d['text_regions_detected']} | "
              f"健康度: {d['layout_health_score']:.3f}")
        if d['has_misalignment']:
            print(f"  ⚠ 布局错位: {len(d['misalignment_details'])} 处")
        if d['has_overlap']:
            print(f"  ⚠ 元素重叠: {len(d['overlap_details'])} 处")
        if d['has_text_overlap']:
            print(f"  ⚠ 文本重叠: {len(d['text_overlap_details'])} 处")
        if not d['has_misalignment'] and not d['has_overlap'] and not d['has_text_overlap']:
            print(f"  ✓ 未检测到异常")

        if args.visualize:
            vis = draw_detection_result(img, result)
            vis_path = args.visualize
            if len(image_paths) > 1:
                base = os.path.splitext(os.path.basename(path))[0]
                os.makedirs(vis_path, exist_ok=True)
                vis_path = os.path.join(vis_path, f"{base}_annotated.png")
            vis.save(vis_path)
            print(f"  可视化: {vis_path}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n报告保存至: {args.output}")

    total = len(all_results)
    with_mis = sum(1 for r in all_results.values() if r["has_misalignment"])
    with_ovr = sum(1 for r in all_results.values() if r["has_overlap"])
    with_txt = sum(1 for r in all_results.values() if r["has_text_overlap"])
    avg_score = np.mean([r["layout_health_score"] for r in all_results.values()])
    print(f"\n{'='*50}")
    print(f"总计分析: {total} 张")
    print(f"布局错位: {with_mis} 张 ({100*with_mis/total:.0f}%)")
    print(f"元素重叠: {with_ovr} 张 ({100*with_ovr/total:.0f}%)")
    print(f"文本重叠: {with_txt} 张 ({100*with_txt/total:.0f}%)")
    print(f"平均健康度: {avg_score:.3f}")


if __name__ == "__main__":
    main()
