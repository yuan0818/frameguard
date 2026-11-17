"""
推理模块：对图片、视频、目录进行质量分类推理。

支持 UI 布局异常检测（后处理）：
  - 当 CNN 分类结果为 UI 相关类别（7-9）时，
    自动调用 ui_detector 进行结构化分析
  - 提供异常定位和健康度评分

用法:
    # 单张图片推理
    python -m src.predict --input image.jpg --checkpoint outputs/checkpoints/best_model.pth

    # 视频推理
    python -m src.predict --input video.mp4 --checkpoint outputs/checkpoints/best_model.pth

    # 批量目录推理
    python -m src.predict --input test_dir/ --checkpoint outputs/checkpoints/best_model.pth

    # 启用 UI 布局深度分析
    python -m src.predict --input ui_screenshot.png --checkpoint best_model.pth --ui-analysis
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.ui_detector import analyze_ui_layout, draw_detection_result
    HAS_UI_DETECTOR = True
except ImportError:
    HAS_UI_DETECTOR = False

try:
    from src.cv_detectors import analyze_cv_quality
    HAS_CV_DETECTOR = True
except ImportError:
    HAS_CV_DETECTOR = False

try:
    from src.advanced_detectors import analyze_advanced
    HAS_ADVANCED_DETECTOR = True
except ImportError:
    HAS_ADVANCED_DETECTOR = False

from src.config import load_config, get_default_config, config_for_checkpoint
from src.model import ImageQualityCNN, create_model
from src.video_utils import (
    is_image_file,
    is_video_file,
    get_video_info,
    compute_frame_indices,
    extract_single_frame,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def build_inference_transform(
    image_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    use_letterbox: bool = False,
) -> transforms.Compose:
    """构建推理用图像变换"""
    if use_letterbox:
        from .dataset import LetterboxResize
        fill_rgb = tuple(int(m * 255) for m in mean)
        resize_op = LetterboxResize(image_size, fill=fill_rgb)
    else:
        resize_op = transforms.Resize(image_size)
    return transforms.Compose([
        resize_op,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def detector_configs(config, template_dir: Optional[str] = None) -> Tuple[dict, dict]:
    """
    从配置的 detectors 节构建 (cv_config, advanced_config)。

    cv_config 直接传给 analyze_cv_quality（各检测器的 enabled 与阈值生效）。
    advanced_config 是 analyze_advanced 使用的扁平键。相对模板路径按项目根目录解析；
    template_dir 给定时覆盖配置，约定其下有 crash/ 与 elements/ 两个子目录。
    """
    detectors = getattr(config, "detectors", None)
    if hasattr(detectors, "to_dict"):
        detectors = detectors.to_dict()
    detectors = detectors or {}

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def resolve(path):
        if not path:
            return None
        return path if os.path.isabs(path) else os.path.join(project_root, path)

    still = detectors.get("still_frame") or {}
    crash = detectors.get("crash") or {}
    missing = detectors.get("missing_element") or {}

    advanced = {
        "crash_template_dir": resolve(crash.get("template_dir")),
        "element_template_dir": resolve(missing.get("template_dir")),
    }
    if template_dir:
        advanced["crash_template_dir"] = os.path.join(template_dir, "crash")
        advanced["element_template_dir"] = os.path.join(template_dir, "elements")
    for target, section, key in (
        ("hist_correlation_threshold", still, "hist_correlation_threshold"),
        ("min_still_frames", still, "min_still_frames"),
        ("crash_match_threshold", crash, "match_threshold"),
        ("element_match_threshold", missing, "match_threshold"),
    ):
        if key in section:
            advanced[target] = section[key]

    return detectors, advanced


def load_model(checkpoint_path: str, config, device: torch.device) -> ImageQualityCNN:
    """
    加载训练好的模型。

    Args:
        checkpoint_path: 模型权重路径
        config: 配置对象
        device: 计算设备

    Returns:
        加载好权重的模型（eval 模式）
    """
    backbone = getattr(config.model, "backbone", None)
    freeze_backbone = getattr(config.model, "freeze_backbone", True)

    model = create_model(
        num_classes=config.model.num_classes,
        dropout_rate=config.model.dropout_rate,
        classifier_dropout=config.model.classifier_dropout,
        input_channels=config.model.input_channels,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        pretrained_path=checkpoint_path,
    )
    model = model.to(device)
    model.eval()
    return model


def predict_single_image(
    image: Union[str, Image.Image],
    model: ImageQualityCNN,
    transform: transforms.Compose,
    device: torch.device,
    class_names: Dict[int, str],
    enable_ui_analysis: bool = False,
    enable_cv_analysis: bool = False,
    cv_config: Optional[dict] = None,
    enable_advanced_analysis: bool = False,
    advanced_config: Optional[dict] = None,
) -> Dict:
    """
    对单张图片进行推理。

    Args:
        image: 图片路径 或 PIL Image 对象
        model: 模型
        transform: 图像变换
        device: 设备
        class_names: 类别 ID → 名称映射
        enable_ui_analysis: 是否启用 UI 布局深度分析（需要 ui_detector 模块）
        enable_cv_analysis: 是否启用传统 CV 质量检测（需要 cv_detectors 模块）
        cv_config: 配置的 detectors 节（None 时使用检测器默认参数）
        enable_advanced_analysis: 是否对这张图片做 Crash 弹窗/元素缺失检测
        advanced_config: analyze_advanced 的配置（见 detector_configs）

    Returns:
        {"class_id": int, "class_name": str, "confidence": float,
         "probabilities": {...}, "ui_analysis": {...} 或 None,
         "cv_analysis": {...} 或 None, "advanced_analysis": {...} 或 None}
    """
    # 加载图片
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    # 变换
    tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()

    # 构建概率字典
    prob_dict = {
        class_names.get(i, f"class_{i}"): probs[i].item()
        for i in range(len(probs))
    }

    # UI 布局分析（可选）
    ui_analysis = None
    if enable_ui_analysis and HAS_UI_DETECTOR:
        # 对任何分类结果都进行 UI 分析，但仅在 UI 异常时详细输出
        try:
            if isinstance(image, str):
                ui_img = Image.open(image).convert("RGB")
            else:
                ui_img = image.convert("RGB")

            ui_result = analyze_ui_layout(ui_img)
            ui_dict = ui_result.to_dict()
            ui_analysis = {
                "layout_health_score": ui_dict["layout_health_score"],
                "elements_detected": ui_dict["elements_detected"],
                "text_regions": ui_dict["text_regions_detected"],
                "has_misalignment": ui_dict["has_misalignment"],
                "has_element_overlap": ui_dict["has_overlap"],
                "has_text_overlap": ui_dict["has_text_overlap"],
                "misalignment_count": len(ui_dict["misalignment_details"]),
                "overlap_count": len(ui_dict["overlap_details"]),
                "text_overlap_count": len(ui_dict["text_overlap_details"]),
            }
            # 补充异常定位信息
            if ui_dict["misalignment_details"]:
                ui_analysis["misalignment_details"] = ui_dict["misalignment_details"][:5]
            if ui_dict["overlap_details"]:
                ui_analysis["overlap_details"] = ui_dict["overlap_details"][:5]
            if ui_dict["text_overlap_details"]:
                ui_analysis["text_overlap_details"] = ui_dict["text_overlap_details"][:5]
        except Exception as e:
            ui_analysis = {"error": str(e)}

    # CV 质量检测（可选）
    cv_analysis = None
    if enable_cv_analysis and HAS_CV_DETECTOR:
        try:
            cv_result = analyze_cv_quality(img, config=cv_config)
            cv_analysis = cv_result.to_dict()
        except Exception as e:
            cv_analysis = {"error": str(e)}

    # 高级检测（可选，单张图片：Crash 弹窗 / 元素缺失）
    advanced_analysis = None
    if enable_advanced_analysis and HAS_ADVANCED_DETECTOR:
        try:
            advanced_analysis = analyze_advanced(img, is_video=False, config=advanced_config).to_dict()
        except Exception as e:
            advanced_analysis = {"error": str(e)}

    return {
        "class_id": pred_class,
        "class_name": class_names.get(pred_class, f"class_{pred_class}"),
        "confidence": confidence,
        "probabilities": prob_dict,
        "ui_analysis": ui_analysis,
        "cv_analysis": cv_analysis,
        "advanced_analysis": advanced_analysis,
    }


def predict_image(
    image_path: str,
    model: ImageQualityCNN,
    transform: transforms.Compose,
    device: torch.device,
    class_names: Dict[int, str],
    enable_ui_analysis: bool = False,
    enable_cv_analysis: bool = False,
    cv_config: Optional[dict] = None,
    enable_advanced_analysis: bool = False,
    advanced_config: Optional[dict] = None,
) -> Dict:
    """
    对单张图片文件进行推理。
    """
    result = predict_single_image(image_path, model, transform, device, class_names,
                                  enable_ui_analysis=enable_ui_analysis,
                                  enable_cv_analysis=enable_cv_analysis,
                                  cv_config=cv_config,
                                  enable_advanced_analysis=enable_advanced_analysis,
                                  advanced_config=advanced_config)
    result["file_path"] = image_path
    return result


def predict_video(
    video_path: str,
    model: ImageQualityCNN,
    transform: transforms.Compose,
    device: torch.device,
    class_names: Dict[int, str],
    max_frames: int = 300,
    frame_interval_sec: float = 1.0,
    enable_ui_analysis: bool = False,
    enable_cv_analysis: bool = False,
    enable_advanced_analysis: bool = False,
    cv_config: Optional[dict] = None,
    advanced_config: Optional[dict] = None,
) -> Dict:
    """
    对视频进行逐帧推理，汇总结果。

    汇总策略：
    - 统计每帧的预测类别，返回多数投票结果
    - 报告各类别帧数占比

    Returns:
        {
            "file_path": str,
            "total_frames": int,
            "overall_class_id": int,
            "overall_class_name": str,
            "overall_confidence": float,
            "frame_results": List[Dict],
            "class_distribution": Dict[str, float],
            "advanced_analysis": {...} 或 None,
        }
    """
    fps, total_frames, duration = get_video_info(video_path)
    frame_indices = compute_frame_indices(total_frames, fps, max_frames, frame_interval_sec)

    frame_results = []
    class_counter = Counter()
    pil_frames: List[Image.Image] = []

    for idx in tqdm(frame_indices, desc=f"Processing {os.path.basename(video_path)}", leave=False):
        frame = extract_single_frame(video_path, idx)
        if frame is None:
            continue

        # 收集 PIL 帧用于高级分析
        if enable_advanced_analysis and HAS_ADVANCED_DETECTOR:
            if isinstance(frame, Image.Image):
                pil_frames.append(frame.copy())
            else:
                pil_frames.append(Image.fromarray(frame).convert("RGB"))

        result = predict_single_image(frame, model, transform, device, class_names,
                                      enable_ui_analysis=enable_ui_analysis,
                                      enable_cv_analysis=enable_cv_analysis,
                                      cv_config=cv_config)
        result["frame_index"] = idx
        result["timestamp"] = idx / fps if fps > 0 else 0.0
        frame_results.append(result)
        class_counter[result["class_name"]] += 1

    # 多数投票
    if class_counter:
        most_common = class_counter.most_common(1)[0]
        overall_class_name = most_common[0]
        overall_confidence = most_common[1] / len(frame_results)

        # 查找对应的 class_id
        overall_class_id = None
        for cid, cname in class_names.items():
            if cname == overall_class_name:
                overall_class_id = cid
                break
    else:
        overall_class_name = "unknown"
        overall_class_id = -1
        overall_confidence = 0.0

    # 类别分布（占比）
    class_distribution = {
        name: count / len(frame_results) if frame_results else 0.0
        for name, count in class_counter.items()
    }

    # 高级分析（视频级）
    advanced_analysis = None
    if enable_advanced_analysis and HAS_ADVANCED_DETECTOR and pil_frames:
        try:
            adv_result = analyze_advanced(pil_frames, is_video=True, config=advanced_config)
            advanced_analysis = adv_result.to_dict()
        except Exception as e:
            advanced_analysis = {"error": str(e)}

    return {
        "file_path": video_path,
        "total_frames": len(frame_results),
        "overall_class_id": overall_class_id,
        "overall_class_name": overall_class_name,
        "overall_confidence": overall_confidence,
        "frame_results": frame_results,
        "class_distribution": class_distribution,
        "advanced_analysis": advanced_analysis,
    }


def predict_directory(
    dir_path: str,
    model: ImageQualityCNN,
    transform: transforms.Compose,
    device: torch.device,
    class_names: Dict[int, str],
    max_frames: int = 300,
    frame_interval_sec: float = 1.0,
    recursive: bool = True,
    enable_ui_analysis: bool = False,
    enable_cv_analysis: bool = False,
    enable_advanced_analysis: bool = False,
    cv_config: Optional[dict] = None,
    advanced_config: Optional[dict] = None,
) -> List[Dict]:
    """
    对目录中的所有图片和视频进行批量推理。

    Args:
        enable_ui_analysis: 是否启用 UI 布局深度分析
        enable_cv_analysis: 是否启用传统 CV 质量检测
        enable_advanced_analysis: 是否启用高级检测

    Returns:
        推理结果列表
    """
    results = []

    # 收集所有文件
    all_files = []
    if recursive:
        for root, _, files in os.walk(dir_path):
            for f in files:
                all_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dir_path):
            fp = os.path.join(dir_path, f)
            if os.path.isfile(fp):
                all_files.append(fp)

    # 分类处理
    for file_path in tqdm(all_files, desc="Processing directory"):
        try:
            if is_image_file(file_path):
                result = predict_image(file_path, model, transform, device, class_names,
                                       enable_ui_analysis=enable_ui_analysis,
                                       enable_cv_analysis=enable_cv_analysis,
                                       cv_config=cv_config,
                                       enable_advanced_analysis=enable_advanced_analysis,
                                       advanced_config=advanced_config)
                result["source_type"] = "image"
                results.append(result)
            elif is_video_file(file_path):
                result = predict_video(
                    file_path, model, transform, device, class_names,
                    max_frames=max_frames,
                    frame_interval_sec=frame_interval_sec,
                    enable_ui_analysis=enable_ui_analysis,
                    enable_cv_analysis=enable_cv_analysis,
                    enable_advanced_analysis=enable_advanced_analysis,
                    cv_config=cv_config,
                    advanced_config=advanced_config,
                )
                result["source_type"] = "video"
                results.append(result)
        except Exception as e:
            print(f"警告: 处理文件 {file_path} 时出错: {e}")

    return results


def print_single_result(result: Dict):
    """格式化打印单条推理结果"""
    print(f"  类别: {result['class_name']} (ID: {result['class_id']})")
    print(f"  置信度: {result['confidence']:.4f}")
    if "probabilities" in result:
        print(f"  各类概率:")
        probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
        for name, prob in probs[:3]:
            print(f"    {name}: {prob:.4f}")

    # UI 分析结果（如果有）
    ui = result.get("ui_analysis")
    if ui:
        if "error" in ui:
            print(f"  UI 布局分析: 错误 - {ui['error']}")
        else:
            print(f"  UI 布局分析:")
            print(f"    健康度评分: {ui['layout_health_score']:.3f}")
            print(f"    检测元素: {ui['elements_detected']} | 文本区域: {ui['text_regions']}")
            if ui["has_misalignment"]:
                print(f"    ⚠ 布局错位: {ui['misalignment_count']} 处")
                if ui.get("misalignment_details"):
                    for m in ui["misalignment_details"][:2]:
                        print(f"      - {m['description']}")
            if ui["has_element_overlap"]:
                print(f"    ⚠ 元素重叠: {ui['overlap_count']} 处")
            if ui["has_text_overlap"]:
                print(f"    ⚠ 文本重叠: {ui['text_overlap_count']} 处")

    # CV 分析结果（如果有）
    cv = result.get("cv_analysis")
    if cv:
        if "error" in cv:
            print(f"  CV 质量检测: 错误 - {cv['error']}")
        else:
            print(f"  CV 质量检测:")
            if cv.get("overall_quality_score") is not None:
                print(f"    综合质量评分: {cv['overall_quality_score']:.3f}")
            blur = cv.get("blur")
            if blur:
                status = "⚠ 模糊" if blur["is_anomaly"] else "正常"
                print(f"    模糊检测: {status} (置信度: {blur['confidence']:.3f})")
            exposure = cv.get("exposure")
            if exposure:
                exp_type = exposure["details"].get("type", "normal")
                status = f"⚠ {exp_type}" if exposure["is_anomaly"] else "正常"
                print(f"    曝光检测: {status} (亮度: {exposure['details'].get('mean_brightness', 0):.1f})")
            truncation = cv.get("truncation")
            if truncation:
                status = f"⚠ {truncation['details'].get('truncated_elements', 0)} 处截断" if truncation["is_anomaly"] else "正常"
                print(f"    截断检测: {status}")
            blank = cv.get("blank_region")
            if blank:
                status = f"⚠ 空白占比 {blank['details'].get('max_blank_ratio', 0):.1%}" if blank["is_anomaly"] else "正常"
                print(f"    空白检测: {status}")

    # 高级分析结果（如果有）
    adv = result.get("advanced_analysis")
    if adv:
        if "error" in adv:
            print(f"  高级检测: 错误 - {adv['error']}")
        else:
            print(f"  高级检测:")
            still = adv.get("still_frames")
            if still:
                status = f"⚠ 连续 {still['details'].get('max_consecutive_still', 0)} 帧静止" if still["is_anomaly"] else "正常"
                print(f"    静止帧检测: {status}")
            crash = adv.get("crash")
            if crash:
                status = "⚠ 检测到崩溃弹窗" if crash["is_anomaly"] else "正常"
                print(f"    Crash 检测: {status}")
            missing = adv.get("missing_elements")
            if missing:
                status = f"⚠ 缺失 {len(missing['details'].get('missing_templates', []))} 个元素" if missing["is_anomaly"] else "正常"
                print(f"    元素缺失检测: {status}")


def print_video_result(result: Dict):
    """格式化打印视频推理结果"""
    print(f"  总帧数: {result['total_frames']}")
    print(f"  整体判断: {result['overall_class_name']} (置信度: {result['overall_confidence']:.4f})")
    print(f"  类别帧数分布:")
    for name, ratio in sorted(result["class_distribution"].items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {ratio:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="图片/视频质量检测 CNN 分类器 - 推理"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入路径（图片、视频或目录）",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="视频最大提取帧数（默认 300）",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=1.0,
        help="视频帧采样间隔（秒，默认 1.0）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="结果输出 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--ui-analysis",
        action="store_true",
        default=False,
        help="启用 UI 布局深度分析（对 UI 截图调用 ui_detector 进行结构化分析）",
    )
    parser.add_argument(
        "--cv-analysis",
        action="store_true",
        default=False,
        help="启用传统 CV 质量检测（模糊/过曝/截断/空白区域）",
    )
    parser.add_argument(
        "--advanced-analysis",
        action="store_true",
        default=False,
        help="启用高级检测（静止帧/Crash/元素缺失，需配合模板）",
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        default=None,
        help="检测模板目录，需包含 crash/ 与 elements/ 子目录；覆盖配置中的 template_dir",
    )
    args = parser.parse_args()

    # 加载配置
    config = config_for_checkpoint(args.checkpoint, args.config)

    # 设备
    device = get_device()
    print(f"设备: {device}")

    # 类别名称映射
    class_names = {}
    cn_config = config.model.class_names
    if hasattr(cn_config, '_data'):
        cn_config = cn_config._data
    if isinstance(cn_config, dict):
        class_names = {int(k) if not isinstance(k, int) else k: v for k, v in cn_config.items()}

    print(f"类别映射: {class_names}")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    # 构建变换
    use_letterbox = getattr(config.data, "letterbox", False)
    transform = build_inference_transform(
        image_size=tuple(config.data.image_size),
        mean=tuple(config.data.mean),
        std=tuple(config.data.std),
        use_letterbox=use_letterbox,
    )

    # 检测器配置（detectors 节 + --template-dir）
    cv_config, advanced_config = detector_configs(config, args.template_dir)

    input_path = args.input

    print(f"\n{'='*60}")
    print(f"推理输入: {input_path}")
    print(f"{'='*60}")

    results = None

    if os.path.isfile(input_path):
        if is_image_file(input_path):
            print("\n[单张图片推理]")
            result = predict_image(input_path, model, transform, device, class_names,
                                   enable_ui_analysis=args.ui_analysis,
                                   enable_cv_analysis=args.cv_analysis,
                                   cv_config=cv_config,
                                   enable_advanced_analysis=args.advanced_analysis,
                                   advanced_config=advanced_config)
            print_single_result(result)
            results = result

        elif is_video_file(input_path):
            print("\n[视频推理]")
            result = predict_video(
                input_path, model, transform, device, class_names,
                max_frames=args.max_frames,
                frame_interval_sec=args.frame_interval,
                enable_ui_analysis=args.ui_analysis,
                enable_cv_analysis=args.cv_analysis,
                enable_advanced_analysis=args.advanced_analysis,
                cv_config=cv_config,
                advanced_config=advanced_config,
            )
            print_video_result(result)
            results = result

        else:
            print(f"错误: 不支持的文件格式: {input_path}")
            sys.exit(1)

    elif os.path.isdir(input_path):
        print("\n[批量目录推理]")
        results_list = predict_directory(
            input_path, model, transform, device, class_names,
            max_frames=args.max_frames,
            frame_interval_sec=args.frame_interval,
            enable_ui_analysis=args.ui_analysis,
            enable_cv_analysis=args.cv_analysis,
            enable_advanced_analysis=args.advanced_analysis,
            cv_config=cv_config,
            advanced_config=advanced_config,
        )

        # 汇总统计
        image_count = sum(1 for r in results_list if r.get("source_type") == "image")
        video_count = sum(1 for r in results_list if r.get("source_type") == "video")
        print(f"\n共处理 {len(results_list)} 个文件 (图片: {image_count}, 视频: {video_count})")

        # 打印每个文件的结果
        print(f"\n{'='*60}")
        print("详细结果:")
        print(f"{'='*60}")
        for r in results_list:
            fpath = r.get("file_path", "unknown")
            print(f"\n--- {os.path.basename(fpath)} ---")
            print(f"  路径: {fpath}")
            if r.get("source_type") == "video":
                print_video_result(r)
            else:
                print_single_result(r)

        results = results_list

    else:
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)

    # 输出 JSON
    if args.output and results is not None:
        import json

        def convert_for_json(obj):
            """递归转换对象为 JSON 可序列化格式"""
            if isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (torch.Tensor,)):
                return obj.tolist()
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            else:
                return obj

        results_json = convert_for_json(results)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
