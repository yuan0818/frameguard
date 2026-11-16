"""
对 normal 图片批量施加 6 类异常 corruption，生成带标签的训练数据。

用法:
    # 从 normal 目录生成所有异常数据
    python scripts/generate_corruptions.py --input data/normal --output data/exception

    # 控制每张原图生成多少变体
    python scripts/generate_corruptions.py --input data/normal --output data/exception --variants 3

    # 仅生成指定类别
    python scripts/generate_corruptions.py --input data/normal --output data/exception --classes black_screen,mosaic
"""

import os
import sys
import argparse
import random
import csv
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# 尝试导入 OpenCV（可选，用于更高质量的 corruption）
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ==============================================================
# 6 类异常 Corruption 生成器
# ==============================================================

def corrupt_screen_corruption(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    花屏/屏幕撕裂: 彩色横条纹 + 随机色块 + 撕裂线。

    策略:
      - 50% 概率: 彩虹色横条纹（模拟显示器花屏）
      - 30% 概率: 随机彩色矩形块
      - 20% 概率: 水平撕裂 + 偏移
    """
    h, w = img.shape[:2]
    result = img.copy().astype(np.float32)
    mode = random.choice(["stripes", "blocks", "tear", "mixed"])

    if mode in ("stripes", "mixed"):
        # 彩色横条纹
        stripe_height = random.randint(2, 12)
        for y in range(0, h, stripe_height):
            if random.random() < 0.6 * severity:
                color = np.random.randint(0, 256, 3).astype(np.float32)
                alpha = random.uniform(0.3, 0.9) * severity
                end_y = min(y + stripe_height, h)
                result[y:end_y, :] = result[y:end_y, :] * (1 - alpha) + color * alpha

    if mode in ("blocks", "mixed"):
        # 随机色块
        for _ in range(int(8 * severity)):
            bx = random.randint(0, w - 1)
            by = random.randint(0, h - 1)
            bw = random.randint(20, max(20, w // 3))
            bh = random.randint(10, max(10, h // 4))
            color = np.random.randint(0, 256, 3).astype(np.float32)
            alpha = random.uniform(0.4, 1.0)
            y1, y2 = max(0, by), min(h, by + bh)
            x1, x2 = max(0, bx), min(w, bx + bw)
            result[y1:y2, x1:x2] = result[y1:y2, x1:x2] * (1 - alpha) + color * alpha

    if mode == "tear":
        # 水平撕裂：随机行偏移
        tear_y = random.randint(h // 4, 3 * h // 4)
        shift = random.randint(20, w // 4) * random.choice([-1, 1])
        tear_h = random.randint(2, 8)
        result[tear_y:tear_y + tear_h, max(0, -shift):min(w, w - shift)] = \
            img[tear_y:tear_y + tear_h, max(0, shift):min(w, w + shift)]
        # 撕裂线颜色
        result[tear_y:tear_y + tear_h, :max(0, shift)] = [255, 0, 255]
        result[tear_y:tear_y + tear_h, min(w, w + shift):] = [0, 255, 0]

    result = np.clip(result, 0, 255).astype(np.uint8)
    # 加少量噪点增强真实感
    noise = (np.random.randn(*result.shape) * 8 * severity).astype(np.int16)
    result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result


def corrupt_black_screen(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    黑屏: 整体极暗（0-15 像素值），模拟黑屏但仍有微弱信号。

    策略:
      - 全图压制到接近 0
      - 保留极少量原图结构（模拟背光漏光）
      - 加微弱噪点
    """
    h, w = img.shape[:2]

    # 基础：极暗（保留 0-3% 的原图信息）
    darkness = random.uniform(0.0, 0.03) * severity
    result = (img.astype(np.float32) * darkness)

    # 模拟背光不均匀: 边缘稍亮
    y_grid, x_grid = np.ogrid[:h, :w]
    vignette = 1 + 0.05 * severity * (
        ((x_grid - w / 2) / (w / 2)) ** 2 + ((y_grid - h / 2) / (h / 2)) ** 2
    )
    result = result * vignette[:, :, np.newaxis]

    # 随机暗色斑块
    for _ in range(random.randint(0, 5)):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        radius = random.randint(5, 30)
        patch_val = random.uniform(2, 15) * severity
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = (dist < radius).astype(np.float32)
        for c in range(3):
            result[:, :, c] = result[:, :, c] * (1 - mask * 0.3) + patch_val * mask * 0.3

    # 微弱噪点（模拟传感器噪声）
    noise = np.random.randn(h, w, 3) * 3 * severity
    result = np.clip(result + noise, 0, 255).astype(np.uint8)
    return result


def corrupt_green_screen(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    绿屏: 整体偏绿，模拟显示器的绿色死机画面。

    策略:
      - 绿色通道大幅增强，R/B 大幅削弱
      - 少量原图结构残留（模拟故障过渡期）
      - 绿色色调有随机变化
    """
    h, w = img.shape[:2]
    result = img.astype(np.float32)

    # 绿色基调
    green_base = random.randint(100, 220)
    green_var = random.randint(10, 40)
    r_factor = random.uniform(0.0, 0.1)
    b_factor = random.uniform(0.0, 0.1)

    result[:, :, 0] *= r_factor  # R 削弱
    result[:, :, 1] = green_base + np.random.randn(h, w) * green_var  # G 主导
    result[:, :, 2] *= b_factor  # B 削弱

    # 保留少量原图亮度作为残影
    gray = np.mean(img, axis=2, keepdims=True)
    ghost_alpha = random.uniform(0.02, 0.15)
    result[:, :, 1] = result[:, :, 1] * (1 - ghost_alpha) + gray[:, :, 0] * ghost_alpha

    # 绿色条纹变化
    for y in range(0, h, random.randint(4, 16)):
        if random.random() < 0.3:
            shift = random.uniform(-20, 20)
            end_y = min(y + random.randint(2, 8), h)
            result[y:end_y, :, 1] = np.clip(result[y:end_y, :, 1] + shift, 0, 255)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def corrupt_blocking_artifact(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    块效应 (Blocking Artifact): 模拟 JPEG/视频编码的 8×8 块状伪影。

    策略:
      - 在 8×8 块边界引入像素值跳变
      - 每块内部加 DCT-like 量化噪声
      - 可选的色度子采样效果
    """
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    block_size = random.choice([4, 8, 16])

    # 方法 1: 逐块添加轻微色偏，制造块边界
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            ey = min(by + block_size, h)
            ex = min(bx + block_size, w)
            if random.random() < 0.3 * severity:
                shift = np.random.randn(3) * 12 * severity
                result[by:ey, bx:ex] += shift

    # 方法 2: 块边界增强（模拟 blocking 最显著的特征）
    border_intensity = 25 * severity
    for by in range(block_size, h, block_size):
        if by < h:
            result[by:min(by + 1, h), :] += border_intensity
    for bx in range(block_size, w, block_size):
        if bx < w:
            result[:, bx:min(bx + 1, w)] += border_intensity

    # 方法 3: JPEG 压缩模拟
    if HAS_CV2 and random.random() < 0.5:
        # 用极低质量 JPEG 编码再解码
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(5 + (1 - severity) * 20)]
        _, enc = cv2.imencode('.jpg', img, encode_params)
        jpeg_result = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if jpeg_result is not None:
            result = cv2.cvtColor(jpeg_result, cv2.COLOR_BGR2RGB).astype(np.float32)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def corrupt_mosaic(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    马赛克: 像素化效果，模拟被打码或极低分辨率放大。

    策略:
      - 下采样到极小尺寸再最近邻放大
      - 块大小随机变化
      - 可选部分区域马赛克（模拟局部打码）
    """
    h, w = img.shape[:2]
    result = img.copy()

    # 马赛克块大小
    mosaic_size = random.randint(
        max(4, int(8 * severity)),
        max(8, int(32 * severity + 4))
    )

    if random.random() < 0.4:
        # 全图马赛克
        small_h, small_w = max(1, h // mosaic_size), max(1, w // mosaic_size)
        small = cv2.resize(result, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # 局部马赛克（1-3 个矩形区域）
        result = result.astype(np.float32)
        for _ in range(random.randint(1, 3)):
            mw = random.randint(w // 5, w // 2)
            mh = random.randint(h // 5, h // 2)
            mx = random.randint(0, max(0, w - mw))
            my = random.randint(0, max(0, h - mh))

            region = result[my:my + mh, mx:mx + mw].astype(np.uint8)
            small_h2 = max(1, mh // mosaic_size)
            small_w2 = max(1, mw // mosaic_size)
            small = cv2.resize(region, (small_w2, small_h2), interpolation=cv2.INTER_LINEAR)
            mosaic_region = cv2.resize(small, (mw, mh), interpolation=cv2.INTER_NEAREST)
            result[my:my + mh, mx:mx + mw] = mosaic_region.astype(np.float32)

        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def corrupt_color_space_error(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
    """
    色彩空间出错: 通道错乱、色相偏移、饱和度异常。

    策略:
      - 30%: RGB 通道随机置换 (如 R↔B, 模拟 YUV/RGB 混淆)
      - 30%: 色相大幅度旋转
      - 20%: 单通道过曝或欠曝
      - 20%: 混合多种色彩异常
    """
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    mode = random.choice(["swap", "hue", "channel_extreme", "mixed"])

    if mode == "swap":
        # 通道置换
        order = list(range(3))
        random.shuffle(order)
        # 确保不是原始顺序
        if order == [0, 1, 2]:
            order = [2, 0, 1]
        result = result[:, :, order]

        # 额外：某个通道乘以异常系数
        channel_to_boost = random.randint(0, 2)
        result[:, :, channel_to_boost] *= random.uniform(0.1, 2.5)

    elif mode == "hue":
        # 在 HSV 空间大幅旋转色相
        if HAS_CV2:
            img_uint8 = img.astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_shift = random.randint(30, 150) * random.choice([-1, 1])
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            # 饱和度也可能异常
            if random.random() < 0.5:
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.3, 3.0), 0, 255)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        else:
            # PIL fallback
            pil_img = Image.fromarray(img)
            # 用增强模拟
            r = ImageEnhance.Color(pil_img).enhance(random.uniform(0.1, 0.5))
            result = np.array(r).astype(np.float32)

    elif mode == "channel_extreme":
        # 某个通道极端异常
        for c in range(3):
            factor = random.choice([0.05, 0.1, 3.0, 5.0])
            if factor > 1:
                result[:, :, c] = np.clip(result[:, :, c] * factor + random.uniform(0, 60), 0, 255)
            else:
                result[:, :, c] *= factor

    elif mode == "mixed":
        # 混合多种异常
        result[:, :, [0, 2]] = result[:, :, [2, 0]]  # R↔B
        result[:, :, 1] = np.clip(result[:, :, 1] * random.uniform(0.3, 2.0), 0, 255)  # G 异常
        if random.random() < 0.5:
            result = np.clip(result + np.random.randint(-30, 30, 3), 0, 255)

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ==============================================================
# 生成器注册表
# ==============================================================

CORRUPTION_REGISTRY: Dict[str, dict] = {
    "screen_corruption": {
        "class_id": 1,
        "fn": corrupt_screen_corruption,
        "description": "花屏/屏幕撕裂",
    },
    "black_screen": {
        "class_id": 2,
        "fn": corrupt_black_screen,
        "description": "黑屏",
    },
    "green_screen": {
        "class_id": 3,
        "fn": corrupt_green_screen,
        "description": "绿屏",
    },
    "blocking_artifact": {
        "class_id": 4,
        "fn": corrupt_blocking_artifact,
        "description": "块效应",
    },
    "mosaic": {
        "class_id": 5,
        "fn": corrupt_mosaic,
        "description": "马赛克",
    },
    "color_space_error": {
        "class_id": 6,
        "fn": corrupt_color_space_error,
        "description": "色彩空间出错",
    },
}


# ==============================================================
# 主流程
# ==============================================================

def load_image(path: str) -> np.ndarray:
    """加载图片为 RGB numpy array"""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(img: np.ndarray, path: str):
    """保存 numpy array 为图片"""
    Image.fromarray(img).save(path)


def process_image(
    img_path: str,
    output_dir: str,
    classes: List[str],
    variants: int,
    base_severity: float,
) -> List[Tuple[str, str]]:
    """
    对一张 normal 图片生成多种 corruption 变体。

    Args:
        img_path: 原图路径
        output_dir: 输出目录
        classes: 要生成的类别列表
        variants: 每类生成的变体数量
        base_severity: 基础严重度 [0, 1]

    Returns:
        [(filename, exception_type), ...] 用于生成 CSV
    """
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path)
    records = []

    for class_name in classes:
        info = CORRUPTION_REGISTRY[class_name]
        fn = info["fn"]

        for v in range(variants):
            # 每个变体随机严重度，增加多样性
            severity = random.uniform(0.3, 1.0) * base_severity
            # 对于某些类别，确保最小严重度
            if class_name in ("black_screen", "green_screen"):
                severity = max(0.6, severity)

            corrupted = fn(img, severity)

            out_name = f"{base_name}_{class_name}_v{v:02d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            save_image(corrupted, out_path)

            records.append((out_name, class_name))

    return records


def main():
    parser = argparse.ArgumentParser(
        description="对 normal 图片批量施加异常 corruption，生成带标签训练数据"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="normal 图片目录路径",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="异常图片输出目录路径",
    )
    parser.add_argument(
        "--classes", "-c", type=str, default=None,
        help="要生成的类别，逗号分隔。默认全部 6 类。可选: " +
             ", ".join(CORRUPTION_REGISTRY.keys()),
    )
    parser.add_argument(
        "--variants", "-v", type=int, default=3,
        help="每张原图每种 corruption 的变体数（默认 3）",
    )
    parser.add_argument(
        "--severity", "-s", type=float, default=0.7,
        help="基础 corruption 严重度 0-1（默认 0.7）",
    )
    parser.add_argument(
        "--image-ext", type=str, default=".jpg,.jpeg,.png,.bmp,.tiff,.webp",
        help="支持的图片扩展名（逗号分隔）",
    )
    parser.add_argument(
        "--csv-name", type=str, default="data.csv",
        help="输出的 CSV 文件名（默认 data.csv）",
    )
    args = parser.parse_args()

    # 解析要生成的类别
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",")]
        for c in classes:
            if c not in CORRUPTION_REGISTRY:
                print(f"错误: 未知类别 '{c}'。可选: {list(CORRUPTION_REGISTRY.keys())}")
                sys.exit(1)
    else:
        classes = list(CORRUPTION_REGISTRY.keys())

    # 支持的图片扩展名
    image_exts = {ext.strip().lower() for ext in args.image_ext.split(",")}

    # 扫描输入目录
    input_dir = args.input
    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)

    image_files = []
    for f in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_exts and not f.startswith("."):
            image_files.append(os.path.join(input_dir, f))

    if not image_files:
        print(f"错误: 在 {input_dir} 中未找到支持的图片文件")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 处理图片
    total = len(image_files) * len(classes) * args.variants
    print(f"输入图片: {len(image_files)} 张")
    print(f"生成类别: {classes}")
    print(f"每类变体: {args.variants} 种")
    print(f"预计生成: {total} 张异常图片")
    print(f"输出目录: {args.output}")
    print("-" * 50)

    all_records = []
    done = 0

    for img_path in image_files:
        try:
            records = process_image(
                img_path, args.output, classes,
                variants=args.variants,
                base_severity=args.severity,
            )
            all_records.extend(records)
            done += 1
            if done % 10 == 0 or done == len(image_files):
                print(f"  进度: {done}/{len(image_files)} 张原图完成 ({len(all_records)} 张输出)")
        except Exception as e:
            print(f"  警告: 处理 {img_path} 时出错: {e}，跳过")

    # 写入 CSV
    csv_path = os.path.join(args.output, args.csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "exception_type"])
        writer.writerows(all_records)

    # 统计
    class_counts = {}
    for _, cls in all_records:
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n{'='*50}")
    print(f"生成完成！")
    print(f"总计: {len(all_records)} 张异常图片")
    print(f"各类别分布:")
    for cls in classes:
        info = CORRUPTION_REGISTRY[cls]
        count = class_counts.get(cls, 0)
        print(f"  {cls:<22} (ID:{info['class_id']}) -> {count:>5} 张  [{info['description']}]")
    print(f"\nCSV 文件: {csv_path}")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()
