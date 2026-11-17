from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from PIL import Image
from src.config import get_default_config, config_for_checkpoint
from src.model import ImageQualityCNN, create_model
from src.dataset import SampleRecord, split_by_source, source_group
from scripts.generate_synthetic_data import generate


def test_source_variants_and_video_frames_stay_together():
    samples = []
    for i in range(10):
        samples += [SampleRecord("image", 0, image_path=f"/normal/photo_{i}.jpg"),
                    SampleRecord("image", 1, image_path=f"/exception/photo_{i}_screen_corruption_v00.jpg"),
                    SampleRecord("image", 2, image_path=f"/exception/photo_{i}_black_screen_v01.jpg")]
    samples += [SampleRecord("video_frame", 0, video_path="clip.mp4", frame_index=i) for i in range(20)]
    tr, va = split_by_source(samples)
    assert not {source_group(s) for s in tr} & {source_group(s) for s in va}
    assert len(tr) + len(va) == len(samples)
    assert [id(s) for s in va] == [id(s) for s in split_by_source(samples)[1]]


def test_explicit_source_and_small_data():
    samples = [SampleRecord("image", 0, image_path="a.jpg", group_id="same"),
               SampleRecord("image", 1, image_path="renamed.jpg", group_id="same")]
    with pytest.raises(ValueError, match="independent"):
        split_by_source(samples)


def test_generator_never_overwrites(tmp_path):
    (tmp_path / "valuable.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        generate(tmp_path)
    assert (tmp_path / "valuable.txt").read_text() == "keep"


def test_forward_backward_and_class_default():
    assert get_default_config().model.num_classes == 7
    model = ImageQualityCNN()
    y = model(torch.randn(2, 3, 64, 64))
    loss = torch.nn.functional.cross_entropy(y, torch.tensor([0, 6]))
    loss.backward()
    assert y.shape == (2, 7)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    p = tmp_path_factory.mktemp("image")
    cfg = get_default_config().to_dict()
    torch.save({"model_state_dict": ImageQualityCNN().state_dict(), "config": cfg}, p / "model.pth")
    Image.fromarray(np.full((64, 80, 3), 120, dtype=np.uint8)).save(p / "sample.png")
    return p


def test_strict_checkpoint_mismatch(checkpoint):
    with pytest.raises(ValueError, match="architecture/classes"):
        create_model(num_classes=10, pretrained_path=str(checkpoint / "model.pth"))
    assert config_for_checkpoint(checkpoint / "model.pth").model.num_classes == 7


def test_cli_has_one_entry_and_accepts_cv(checkpoint):
    result = subprocess.run([sys.executable, "-m", "src.predict", "-i", str(checkpoint / "sample.png"),
                             "-c", str(checkpoint / "model.pth"), "--cv-analysis", "-o", str(checkpoint / "out.json")],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("加载模型:") == 1
    assert (checkpoint / "out.json").exists()


def test_legacy_evaluation_requires_explicit_opt_in(checkpoint):
    from src.evaluate import evaluate
    with pytest.raises(ValueError, match="provenance"):
        evaluate(str(checkpoint / "model.pth"), get_default_config(), False)


def test_export_metadata_records_preprocessing(tmp_path):
    import json
    from scripts.export_model import export_info
    cfg = get_default_config()
    cfg.data.letterbox = True
    info = json.loads(Path(export_info(cfg, str(tmp_path))).read_text())
    assert info["letterbox"] is True
    assert info["num_classes"] == 7
