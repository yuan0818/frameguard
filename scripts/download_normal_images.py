"""
从多个来源下载正常图片，用于训练数据准备。

用法:
    # 从 COCO val2017 下载 500 张（推荐，最快）
    python scripts/download_normal_images.py --source coco --count 500 --output test/dataset

    # 从 HuggingFace 下载
    python scripts/download_normal_images.py --source hf --count 500 --output test/dataset

    # 从 DIV2K 下载
    python scripts/download_normal_images.py --source div2k --output test/dataset

来源:
    coco   - COCO 2017 val (1GB, 5000 张, CC-BY)
    hf     - HuggingFace 通用图片数据集 (CC-BY/CC0)
    div2k  - DIV2K 高清图片 (800 张, 学术用途)
"""

import os
import sys
import argparse
import random
import io
import ssl
import zipfile
import tarfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request, urlretrieve

from PIL import Image

# --- SSL Fix ---
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    try:
        SSL_CTX = ssl.create_default_context()
    except Exception:
        SSL_CTX = ssl._create_unverified_context()

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"


# ==============================================================
# Source: COCO 2017 val (最推荐)
# ==============================================================
def download_coco(output_dir: str, count: int, min_size: int, seed: int):
    """从 COCO val2017 下载图片"""
    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(output_dir, "_val2017.zip")
    extract_dir = os.path.join(output_dir, "_val2017_extract")

    print(f"下载 COCO val2017 (~1GB)...")
    print(f"  {url}")

    # 下载 zip
    if not os.path.exists(zip_path):
        _download_file(url, zip_path, desc="  val2017.zip")
    else:
        print(f"  使用已缓存: {zip_path}")

    # 解压
    if not os.path.exists(extract_dir):
        print(f"  解压中...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        print(f"  使用已解压: {extract_dir}")

    # 收集所有图片
    all_images = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, f))

    print(f"  共 {len(all_images)} 张图片")

    # 随机选择 + 过滤小图
    random.seed(seed)
    random.shuffle(all_images)

    selected = 0
    for img_path in all_images:
        if selected >= count:
            break
        try:
            img = Image.open(img_path)
            w, h = img.size
            if w >= min_size and h >= min_size:
                dst = os.path.join(output_dir, os.path.basename(img_path))
                shutil.copy(img_path, dst)
                selected += 1
        except Exception:
            pass

    # 清理临时文件
    shutil.rmtree(extract_dir)
    os.remove(zip_path)

    return selected


# ==============================================================
# Source: HuggingFace (使用 datasets 库)
# ==============================================================
def download_huggingface(output_dir: str, count: int, seed: int):
    """从 HuggingFace 下载图片（多种公开数据集）"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("需要安装 datasets 库: pip install datasets")
        return 0

    # 尝试多个公开图片数据集
    datasets_to_try = [
        ("zh-plus/COCO-30", "train"),       # COCO 子集
        ("polinaeterna/coco-2017-val", "train"),  # COCO val
    ]

    selected = 0
    for ds_name, split in datasets_to_try:
        if selected >= count:
            break
        try:
            print(f"  尝试: {ds_name}")
            ds = load_dataset(ds_name, split=split, streaming=True)
            ds = ds.shuffle(seed=seed)

            for item in ds:
                if selected >= count:
                    break
                try:
                    img = item.get("image") or item.get("img") or item.get("jpg")
                    if img is None:
                        continue
                    if hasattr(img, "save"):
                        img.save(os.path.join(output_dir, f"hf_{selected:05d}.jpg"))
                    else:
                        Image.fromarray(img).save(os.path.join(output_dir, f"hf_{selected:05d}.jpg"))
                    selected += 1
                    if selected % 50 == 0:
                        print(f"    已下载 {selected}/{count}")
                except Exception:
                    continue

            if selected > 0:
                return selected
        except Exception as e:
            print(f"    失败: {e}")
            continue

    return selected


# ==============================================================
# Source: DIV2K
# ==============================================================
def download_div2k(output_dir: str, min_size: int, seed: int):
    """下载 DIV2K 训练集（800 张高清图，~5GB）"""
    # 尝试多个镜像
    urls = [
        "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "https://cv.snu.ac.kr/research/EDSR/DIV2K.tar",
    ]
    zip_path = os.path.join(output_dir, "_div2k.zip")
    extract_dir = os.path.join(output_dir, "_div2k_extract")

    downloaded = False
    for url in urls:
        try:
            print(f"下载 DIV2K...")
            print(f"  {url}")
            _download_file(url, zip_path, desc="  DIV2K")
            downloaded = True
            break
        except Exception as e:
            print(f"  失败: {e}")
            continue

    if not downloaded:
        print("所有 DIV2K 镜像不可用")
        return 0

    # 解压
    print(f"  解压中...")
    os.makedirs(extract_dir, exist_ok=True)

    if zip_path.endswith(".zip"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    elif zip_path.endswith(".tar"):
        with tarfile.open(zip_path, "r") as tf:
            tf.extractall(extract_dir)

    # 移动图片
    count = 0
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                src = os.path.join(root, f)
                dst = os.path.join(output_dir, f)
                shutil.move(src, dst)
                count += 1

    # 清理
    shutil.rmtree(extract_dir)
    os.remove(zip_path)

    print(f"  共获得 {count} 张 DIV2K 图片")
    return count


# ==============================================================
# 工具函数
# ==============================================================
def _download_file(url: str, dest: str, desc: str = "", timeout: int = 300):
    """带进度条的下载"""
    from urllib.request import urlopen, Request

    req = Request(url, headers={"User-Agent": USER_AGENT})
    response = urlopen(req, timeout=timeout, context=SSL_CTX)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    block_size = 8192

    with open(dest, "wb") as f:
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r{desc} {pct:.0f}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

    print()


# ==============================================================
# Main
# ==============================================================
def main():
    parser = argparse.ArgumentParser(
        description="下载正常图片（多来源支持）"
    )
    parser.add_argument(
        "--source", "-s", type=str, default="coco",
        choices=["coco", "hf", "div2k"],
        help="数据来源: coco (推荐), hf (HuggingFace), div2k (高清)",
    )
    parser.add_argument(
        "--count", "-n", type=int, default=500,
        help="下载数量（coco/hf 模式有效）",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="test/dataset",
        help="输出目录",
    )
    parser.add_argument(
        "--min-size", type=int, default=224,
        help="最小图片边长",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_abs = os.path.abspath(args.output)

    print(f"{'='*55}")
    print(f"下载正常图片")
    print(f"  来源: {args.source}")
    print(f"  目标: {args.count if args.source != 'div2k' else '全部'} 张")
    print(f"  输出: {output_abs}")
    print(f"{'='*55}")

    if args.source == "coco":
        count = download_coco(args.output, args.count, args.min_size, args.seed)
    elif args.source == "hf":
        count = download_huggingface(args.output, args.count, args.seed)
    elif args.source == "div2k":
        count = download_div2k(args.output, args.min_size, args.seed)
    else:
        count = 0

    # 统计
    final = len([
        f for f in os.listdir(args.output)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.startswith("_")
    ])

    print(f"\n{'='*55}")
    print(f"完成! 共 {final} 张图片 → {output_abs}")
    print(f"{'='*55}")
    print(f"\n下一步:")
    print(f"  python scripts/generate_corruptions.py \\")
    print(f"      --input {output_abs} \\")
    print(f"      --output data/exception \\")
    print(f"      --variants 3")


if __name__ == "__main__":
    main()
