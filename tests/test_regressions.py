"""Regression tests for preprocessing, label matching, detector wiring and held-out evaluation."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from src.advanced_detectors import _rule_based_crash_detect, analyze_advanced
from src.config import get_default_config
from src.dataset import (ImageQualityDataset, LetterboxResize, SampleRecord,
                         heldout_overlap, scan_exception_samples)
from src.model import ImageQualityCNN, PretrainedBackbone
from src.predict import detector_configs

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = [".jpg", ".png"]
VIDEO_EXTS = [".mp4"]


def _exception_dir(tmp_path, files, csv_text):
    root = tmp_path / "exception"
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    (root / "data.csv").write_text(csv_text, encoding="utf-8")
    return root


def _scan(root):
    samples = scan_exception_samples(str(root), "data.csv", IMAGE_EXTS, VIDEO_EXTS)
    return {os.path.relpath(s.image_path, root).replace(os.sep, "/"): (s.label, s.group_id)
            for s in samples}


def test_letterbox_training_sees_the_whole_frame():
    aug = get_default_config().data.augmentation.to_dict()
    assert aug["random_resized_crop"]["enable"]
    dataset = ImageQualityDataset([], is_train=True, augmentation_config=aug, use_letterbox=True)
    first = dataset.transform.transforms[0]
    assert isinstance(first, LetterboxResize)

    # A tall screenshot with a red header and a blue footer: both must still be in frame.
    tall = np.full((400, 224, 3), 128, dtype=np.uint8)
    tall[:40] = [255, 0, 0]
    tall[-40:] = [0, 0, 255]
    boxed = np.array(first(Image.fromarray(tall)))
    assert boxed.shape == (224, 224, 3)
    assert (boxed[..., 0] > 200).any() and (boxed[..., 2] > 200).any()


def test_source_id_survives_csv_names_without_extension(tmp_path):
    root = _exception_dir(tmp_path, ["renamed_a.jpg"],
                          "file_name,exception_type,source_id\nrenamed_a,mosaic,photo_7\n")
    assert _scan(root) == {"renamed_a.jpg": (5, "photo_7")}


def test_same_basename_in_subfolders_keeps_its_own_row(tmp_path):
    csv_text = ("file_name,exception_type,source_id\n"
                "a/frame.jpg,black_screen,src_a\n"
                "b/frame.jpg,mosaic,src_b\n")
    root = _exception_dir(tmp_path, ["a/frame.jpg", "b/frame.jpg", "c/frame.jpg"], csv_text)
    # c/frame.jpg has no row of its own and the basename is ambiguous, so it stays unlabeled.
    assert _scan(root) == {"a/frame.jpg": (2, "src_a"), "b/frame.jpg": (5, "src_b")}


def test_frozen_backbone_batchnorm_stats_do_not_drift():
    model = PretrainedBackbone("resnet18", num_classes=7, freeze_backbone=True, imagenet_init=False)
    model.train()
    assert not model.backbone.training and model.classifier.training
    before = model.backbone.bn1.running_mean.clone()
    model(torch.randn(4, 3, 64, 64))
    assert torch.equal(model.backbone.bn1.running_mean, before)


def test_detector_config_is_built_from_the_detectors_section(tmp_path):
    config = get_default_config()
    cv_config, advanced = detector_configs(config)
    assert cv_config["blur"]["laplacian_variance_threshold"] == 100.0
    assert advanced["min_still_frames"] == 5
    assert advanced["crash_match_threshold"] == 0.7
    assert os.path.isabs(advanced["crash_template_dir"])

    _, advanced = detector_configs(config, str(tmp_path))
    assert advanced["crash_template_dir"] == str(tmp_path / "crash")
    assert advanced["element_template_dir"] == str(tmp_path / "elements")


def test_cli_honors_detector_config_and_runs_advanced_on_images(tmp_path):
    cfg = get_default_config().to_dict()
    cfg["detectors"]["blur"]["enabled"] = False
    torch.save({"model_state_dict": ImageQualityCNN().state_dict(), "config": cfg}, tmp_path / "model.pth")
    Image.fromarray(np.full((64, 80, 3), 120, dtype=np.uint8)).save(tmp_path / "sample.png")

    result = subprocess.run(
        [sys.executable, "-m", "src.predict", "-i", str(tmp_path / "sample.png"),
         "-c", str(tmp_path / "model.pth"), "--cv-analysis", "--advanced-analysis",
         "--template-dir", str(tmp_path / "templates"), "-o", str(tmp_path / "out.json")],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr

    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["cv_analysis"]["blur"] is None
    assert out["cv_analysis"]["exposure"] is not None
    assert out["advanced_analysis"]["crash"] is not None
    assert out["advanced_analysis"]["still_frames"] is None


def test_rule_based_crash_uses_rgb_luminance():
    # Red dialog on a blue background is brighter in RGB luminance and darker if channels are swapped.
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[..., 2] = 255
    frame[90:150, 120:200] = [255, 0, 0]
    _, location = _rule_based_crash_detect(frame)
    assert location is not None


def test_video_crash_check_looks_past_the_first_frame():
    plain = np.full((240, 320, 3), 40, dtype=np.uint8)
    dialog = plain.copy()
    dialog[90:150, 120:200] = 230
    result = analyze_advanced([plain, plain, dialog], is_video=True)
    assert result.crash.is_anomaly
    assert result.crash.details["sampled_frame_position"] == 2


def test_heldout_overlap_flags_sources_seen_in_training():
    manifest = {"train_groups": ["photo_1"], "val_groups": ["photo_2"]}
    seen = SampleRecord("image", 5, image_path="/x/photo_1_mosaic_v00.jpg")
    fresh = SampleRecord("image", 0, image_path="/x/other.jpg")
    assert heldout_overlap([seen, fresh], manifest) == ["photo_1"]
    assert heldout_overlap([fresh], manifest) == []
