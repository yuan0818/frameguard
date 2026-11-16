"""
生成 UI 布局异常数据：布局错位、元素重叠、文本重叠。

用法:
    python scripts/generate_ui_data.py --count 500 --output data/ui_exception
    python scripts/generate_ui_data.py --count 500 --output data/ui_normal --mode normal

生成的 UI 截图模拟移动 App 页面（9:16 宽高比），通过 Letterbox 填充为 224×224。
"""

import os
import sys
import argparse
import random
import csv
from typing import List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 尝试加载系统字体
FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            FONT = ImageFont.truetype(fp, size=14)
            break
        except Exception:
            pass
if FONT is None:
    FONT = ImageFont.load_default()

# ==============================================================
# UI 组件定义
# ==============================================================

@dataclass
class Rect:
    x: int; y: int; w: int; h: int
    color: Tuple[int, int, int]
    label: str = ""
    text_color: Tuple[int, int, int] = (255, 255, 255)
    corner_radius: int = 4
    border_color: Optional[Tuple[int, int, int]] = None


def draw_rounded_rect(draw, rect: Rect):
    """绘制圆角矩形"""
    r = rect.corner_radius
    color = rect.color
    draw.rounded_rectangle(
        [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
        radius=r, fill=color,
        outline=rect.border_color, width=1 if rect.border_color else 0,
    )


def draw_text(draw, rect: Rect):
    """在矩形中居中绘制文本"""
    if not rect.label:
        return
    try:
        bbox = draw.textbbox((0, 0), rect.label, font=FONT)
    except Exception:
        return
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = rect.x + (rect.w - tw) // 2
    ty = rect.y + (rect.h - th) // 2
    draw.text((tx, ty), rect.label, fill=rect.text_color, font=FONT)


# ==============================================================
# UI 布局模板
# ==============================================================

class UILayoutTemplate:
    """生成一张完整的移动 App UI 截图"""

    # App 配色主题
    THEMES = [
        {"bg": (245, 245, 248), "header": (44, 123, 229), "btn": (44, 123, 229),
         "card": (255, 255, 255), "text": (30, 30, 30), "subtext": (140, 140, 150),
         "accent": (255, 149, 0), "danger": (255, 59, 48)},
        {"bg": (248, 248, 248), "header": (34, 139, 34), "btn": (34, 139, 34),
         "card": (255, 255, 255), "text": (20, 20, 20), "subtext": (120, 120, 130),
         "accent": (255, 165, 0), "danger": (220, 20, 60)},
        {"bg": (250, 250, 252), "header": (88, 86, 214), "btn": (88, 86, 214),
         "card": (255, 255, 255), "text": (25, 25, 25), "subtext": (130, 130, 140),
         "accent": (255, 193, 7), "danger": (255, 69, 58)},
    ]

    def __init__(self, canvas_w: int = 224, canvas_h: int = 400):
        self.cw = canvas_w   # canvas width
        self.ch = canvas_h   # canvas height (9:16 mobile)
        self.theme = random.choice(self.THEMES)

        # 布局参数
        self.header_h = 48
        self.tab_h = 36
        self.margin = 10
        self.card_spacing = 8
        self.bottom_nav_h = 44

    def _make_rects(self) -> List[Rect]:
        """生成正常 UI 布局的组件列表"""
        t = self.theme
        cw, ch = self.cw, self.ch
        mg = self.margin
        rects = []

        # --- Status bar ---
        rects.append(Rect(0, 0, cw, 20, (0, 0, 0, 0), "", border_color=None))

        # --- Header ---
        rects.append(Rect(0, 20, cw, self.header_h - 20, t["header"],
                          "App Title", (255, 255, 255), corner_radius=0))

        # --- Tab bar ---
        tab_y = self.header_h + 4
        tab_w = cw // 3
        tabs = ["推荐", "关注", "热门"]
        for i, name in enumerate(tabs):
            active = (i == 0)
            color = t["btn"] if active else (220, 220, 225)
            tcolor = (255, 255, 255) if active else t["subtext"]
            rects.append(Rect(i * tab_w, tab_y, tab_w - 1, self.tab_h,
                              color, name, tcolor, corner_radius=0))

        # --- Content cards ---
        content_y = tab_y + self.tab_h + mg
        card_w = cw - 2 * mg
        card_h = random.randint(50, 80)

        card_configs = [
            ("商品推荐", t["accent"], "立即查看  >"),
            ("最新消息", t["btn"], "了解更多  >"),
            ("热门活动", t["danger"], "参与活动  >"),
        ]

        for title, badge_color, action_text in card_configs:
            # Card background
            rects.append(Rect(mg, content_y, card_w, card_h, t["card"],
                              "", border_color=(220, 220, 228)))
            # Badge/icon area
            rects.append(Rect(mg + 8, content_y + 10, 40, card_h - 20,
                              badge_color, "", corner_radius=6))
            # Title text
            rects.append(Rect(mg + 58, content_y + 8, card_w - 80, 20,
                              (0, 0, 0, 0), title, t["text"], corner_radius=0))
            # Subtitle
            rects.append(Rect(mg + 58, content_y + 30, card_w - 80, 16,
                              (0, 0, 0, 0), "描述文字内容", t["subtext"], corner_radius=0))
            # Action button
            rects.append(Rect(cw - mg - 70, content_y + card_h - 28, 60, 22,
                              t["btn"], action_text, (255, 255, 255), corner_radius=10))
            content_y += card_h + self.card_spacing

        # --- List items ---
        for i in range(3):
            item_h = 40
            rects.append(Rect(mg, content_y, card_w, item_h, t["card"],
                              "", border_color=(220, 220, 228)))
            icon_size = 24
            rects.append(Rect(mg + 8, content_y + 8, icon_size, icon_size,
                              (100 + i * 30, 150, 200), "", corner_radius=12))
            rects.append(Rect(mg + 40, content_y + 4, 80, 18,
                              (0, 0, 0, 0), f"列表项 {i+1}", t["text"], corner_radius=0))
            rects.append(Rect(mg + 40, content_y + 22, 50, 14,
                              (0, 0, 0, 0), "副标题", t["subtext"], corner_radius=0))
            content_y += item_h + 2

        # --- Bottom navigation ---
        rects.append(Rect(0, ch - self.bottom_nav_h, cw, self.bottom_nav_h,
                          (248, 248, 250), "", border_color=(220, 220, 228)))
        nav_items = ["首页", "发现", "消息", "我的"]
        nav_w = cw // 4
        for i, name in enumerate(nav_items):
            active = (i == 0)
            c = t["btn"] if active else t["subtext"]
            rects.append(Rect(i * nav_w, ch - self.bottom_nav_h + 10,
                              nav_w, 26, (0, 0, 0, 0), name, c, corner_radius=0))

        return rects

    def render_normal(self) -> Image.Image:
        """渲染正常 UI 布局"""
        return self._render(self._make_rects())

    def render_misalignment(self, severity: float = 0.5) -> Image.Image:
        """
        渲染布局错位：元素偏离正确位置。

        错位类型:
          - 卡片水平/垂直偏移
          - 按钮错位
          - 列表项不对齐
          - Tab 高亮偏移
        """
        rects = self._make_rects()
        n = len(rects)

        # 随机选取部分元素做偏移
        shift_indices = random.sample(
            range(2, n),  # 避开 status bar 和 header
            k=random.randint(3, max(3, int(n * 0.3 * severity)))
        )
        max_shift = int(15 * severity)

        for idx in shift_indices:
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            # 保证不出画布
            rects[idx].x = max(0, min(self.cw - rects[idx].w // 2, rects[idx].x + dx))
            rects[idx].y = max(0, min(self.ch - rects[idx].h // 2, rects[idx].y + dy))

        return self._render(rects)

    def render_overlap(self, severity: float = 0.5) -> Image.Image:
        """
        渲染元素重叠：两个或多个元素发生重叠。

        策略：选取内容区域的元素，将其移动到与其他元素重叠的位置。
        """
        rects = self._make_rects()

        # 选取内容区域的 card 和 list item（跳过 status bar、header、tab、bottom nav）
        content_rects = [(i, r) for i, r in enumerate(rects)
                         if r.y > self.header_h + self.tab_h
                         and r.y < self.ch - self.bottom_nav_h
                         and r.color != (0, 0, 0, 0)]  # 排除透明占位

        if len(content_rects) < 2:
            return self._render(rects)

        num_overlaps = random.randint(1, max(1, int(3 * severity)))
        for _ in range(num_overlaps):
            # 选两个元素
            a_idx, a_rect = random.choice(content_rects)
            b_idx, b_rect = random.choice(content_rects)
            if a_idx == b_idx:
                continue

            # 把 a 移到 b 上，制造重叠
            overlap_ratio = random.uniform(0.2, 0.6)
            rects[a_idx].x = b_rect.x + int(b_rect.w * (1 - overlap_ratio))
            rects[a_idx].y = b_rect.y + int(b_rect.h * (1 - overlap_ratio))

        return self._render(rects)

    def render_text_overlap(self, severity: float = 0.5) -> Image.Image:
        """
        渲染文本重叠：文本元素相互覆盖导致不可读。

        策略：
          - 把多个文本 label 集中到同一位置
          - 或者把文本区域移到其他文本上方
        """
        rects = self._make_rects()

        # 找所有带 label 的元素
        text_rects = [(i, r) for i, r in enumerate(rects)
                      if r.label and r.color == (0, 0, 0, 0)]  # 透明文本占位

        if len(text_rects) < 2:
            return self._render(rects)

        # 制造文本扎堆
        num_clusters = random.randint(1, max(1, int(3 * severity)))
        for _ in range(num_clusters):
            cluster = random.sample(text_rects, min(3, len(text_rects)))
            # 以第一个为锚点
            anchor_x, anchor_y = cluster[0][1].x, cluster[0][1].y
            for idx, r in cluster[1:]:
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-3, 3)
                rects[idx].x = anchor_x + offset_x
                rects[idx].y = anchor_y + offset_y

        # 额外：把一些文本移到其他文本组件上方
        for i, (idx, r) in enumerate(text_rects):
            if random.random() < 0.3 * severity:
                # 找附近的另一个文本
                other = random.choice(text_rects)
                if other[0] != idx:
                    rects[idx].x = other[1].x + random.randint(-8, 8)
                    rects[idx].y = other[1].y + random.randint(-4, 4)

        return self._render(rects)

    def _render(self, rects: List[Rect]) -> Image.Image:
        """渲染所有组件到画布"""
        img = Image.new("RGB", (self.cw, self.ch), self.theme["bg"])
        draw = ImageDraw.Draw(img)

        for rect in rects:
            # 跳过透明占位
            if rect.color == (0, 0, 0, 0) and not rect.label:
                continue

            if rect.color != (0, 0, 0, 0):
                draw_rounded_rect(draw, rect)

            if rect.label:
                draw_text(draw, rect)

        return img


# ==============================================================
# 主流程
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="生成 UI 布局异常数据"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=500,
        help="每类生成数量（默认 500）",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/ui_exception",
        help="输出目录",
    )
    parser.add_argument(
        "--mode", "-m", type=str, default="all",
        choices=["all", "normal", "misalignment", "overlap", "text_overlap"],
        help="生成模式",
    )
    parser.add_argument(
        "--canvas-width", type=int, default=224,
    )
    parser.add_argument(
        "--canvas-height", type=int, default=400,
    )
    parser.add_argument(
        "--severity", "-s", type=float, default=0.7,
        help="异常严重度 0-1",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    modes = {
        "normal": "ui_normal",
        "misalignment": "ui_layout_misalignment",
        "overlap": "ui_element_overlap",
        "text_overlap": "ui_text_overlap",
    }

    if args.mode == "all":
        active_modes = list(modes.items())
    else:
        active_modes = [(args.mode, modes[args.mode])]

    csv_rows = []
    generated = 0

    for mode_name, class_name in active_modes:
        print(f"生成 {class_name} ...")

        for i in range(args.count):
            template = UILayoutTemplate(args.canvas_width, args.canvas_height)

            if mode_name == "normal":
                img = template.render_normal()
            elif mode_name == "misalignment":
                img = template.render_misalignment(args.severity)
            elif mode_name == "overlap":
                img = template.render_overlap(args.severity)
            elif mode_name == "text_overlap":
                img = template.render_text_overlap(args.severity)
            else:
                continue

            fname = f"ui_{class_name}_{i:04d}.png"
            img.save(os.path.join(args.output, fname))
            csv_rows.append([fname, class_name])
            generated += 1

        print(f"  → {args.count} 张")

    # 写入 CSV
    csv_path = os.path.join(args.output, "data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "exception_type"])
        writer.writerows(csv_rows)

    print(f"\n{'='*50}")
    print(f"完成! 共 {generated} 张 UI 截图 → {os.path.abspath(args.output)}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
