"""
从 Google Open Images V7 下载指定数量的正常图片。

用法:
    # 下载 500 张图片到 data/normal/
    python scripts/download_openimages.py --count 500 --output data/normal

    # 指定分辨率下限（过滤小图）
    python scripts/download_openimages.py --count 500 --min-size 300 --output data/normal

工作原理:
    1. 从 Open Images 下载图片 ID 列表
    2. 随机抽取指定数量
    3. 多线程从 Google S3 并行下载
"""

import os
import sys
import argparse
import random
import csv
import io
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np
from PIL import Image

# 修复 macOS Python 的 SSL 证书问题
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    try:
        SSL_CONTEXT = ssl.create_default_context()
    except Exception:
        SSL_CONTEXT = ssl._create_unverified_context()

# Open Images V7 训练集图片 ID 列表（约 1.9M 个 ID，~18MB）
IMAGE_ID_URL = "https://storage.googleapis.com/openimages/v7/oidv7-train-images-with-labels.csv"
# 也可以用验证集（更小，~41K ID）
VAL_IMAGE_ID_URL = "https://storage.googleapis.com/openimages/v6/oidv6-val-images-with-labels.csv"

# 图片下载 URL 模板（Google S3）
IMAGE_URL_TEMPLATE = "https://s3.amazonaws.com/open-images-dataset/train/{image_id}.jpg"


def download_csv(url: str) -> list:
    """下载并解析 Open Images CSV，提取图片 URL 列表"""
    print(f"正在下载图片 ID 列表...")
    print(f"  {url}")

    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = urlopen(req, timeout=60, context=SSL_CONTEXT).read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(data))
    image_ids = []
    for row in reader:
        image_id = row.get("ImageID", "")
        if image_id:
            image_ids.append(image_id)

    print(f"  共 {len(image_ids):,} 个图片 ID")
    return image_ids


def download_single_image(image_id: str, output_dir: str, min_size: int = 0, timeout: int = 15) -> bool:
    """
    下载单张图片。

    Returns:
        True 如果下载成功且满足尺寸要求
    """
    url = IMAGE_URL_TEMPLATE.format(image_id=image_id)
    out_path = os.path.join(output_dir, f"{image_id}.jpg")

    # 跳过已存在的
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urlopen(req, timeout=timeout, context=SSL_CONTEXT).read()

        if len(data) < 2048:
            return False

        # 检查最小尺寸
        if min_size > 0:
            try:
                img = Image.open(io.BytesIO(data))
                w, h = img.size
                if w < min_size or h < min_size:
                    return False
            except Exception:
                return False

        # 保存
        with open(out_path, "wb") as f:
            f.write(data)
        return True

    except (URLError, OSError, Exception):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="从 Google Open Images V7 下载正常图片"
    )
    parser.add_argument(
        "--count", "-n", type=int, default=500,
        help="下载图片数量（默认 500）",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/normal",
        help="输出目录（默认 data/normal）",
    )
    parser.add_argument(
        "--min-size", "-s", type=int, default=224,
        help="图片最小边长（默认 224，过滤小图）",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=8,
        help="并行下载线程数（默认 8）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--timeout", type=int, default=20,
        help="单张图片下载超时（秒）",
    )
    parser.add_argument(
        "--val", action="store_true",
        help="使用验证集 ID 列表（更快，但仅 41K 个 ID）",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # --- Step 1: 获取图片 ID 列表 ---
    csv_url = VAL_IMAGE_ID_URL if args.val else IMAGE_ID_URL
    all_ids = download_csv(csv_url)

    # --- Step 2: 随机抽取 ---
    # 多抽取一些，补偿下载失败
    sample_count = min(args.count * 2, len(all_ids))
    sampled_ids = random.sample(all_ids, sample_count)
    print(f"\n随机抽取 {sample_count} 个 ID（目标 {args.count} 张，补偿失败）")

    # --- Step 3: 多线程下载 ---
    print(f"\n开始下载（{args.workers} 线程，最小边长 {args.min_size}px）...")
    success = 0
    failed = 0
    skipped_small = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_single_image, img_id, args.output, args.min_size, args.timeout
            ): img_id
            for img_id in sampled_ids
        }

        for i, future in enumerate(as_completed(futures)):
            img_id = futures[future]
            try:
                ok = future.result()
            except Exception:
                ok = False

            if ok:
                success += 1
            else:
                failed += 1

            # 进度
            if (i + 1) % 50 == 0 or (i + 1) == len(sampled_ids):
                print(f"  进度: {i+1}/{len(sampled_ids)}  "
                      f"成功={success} 失败={failed}  "
                      f"({100*success/(i+1):.0f}% 成功率)")

            # 达到目标数量，提前退出
            if success >= args.count:
                # 取消剩余任务
                for f in futures:
                    f.cancel()
                break

    # --- Step 4: 清理（删除失败/空文件） ---
    for f in os.listdir(args.output):
        fpath = os.path.join(args.output, f)
        if os.path.getsize(fpath) < 2048:
            os.remove(fpath)

    final_count = len([
        f for f in os.listdir(args.output)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\n{'='*50}")
    print(f"下载完成！")
    print(f"  实际获得: {final_count} 张有效图片")
    print(f"  保存至:   {os.path.abspath(args.output)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
