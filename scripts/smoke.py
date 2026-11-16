"""Run a self-contained CPU pipeline smoke test in temporary storage."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import yaml

ROOT = Path(__file__).resolve().parents[1]

def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"},
                            capture_output=True, text=True, timeout=240)
    if result.returncode:
        raise RuntimeError(f"{args}:\n{result.stdout}\n{result.stderr}")
    print("PASS", " ".join(map(str, args[:3])), flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="frameguard-smoke-") as temp:
        tmp = Path(temp)
        cfg = yaml.safe_load((ROOT / "config/demo.yaml").read_text())
        cfg["data"]["root_dir"] = str(tmp / "data")
        cfg["output"]["checkpoint_dir"] = str(tmp / "checkpoints")
        cfg["output"]["log_dir"] = str(tmp / "logs")
        config = tmp / "config.yaml"
        config.write_text(yaml.safe_dump(cfg))
        run("scripts/generate_synthetic_data.py", "--out", tmp / "data", "--sources", 8)
        run("-m", "src.train", "--config", config, "--device", "cpu")
        checkpoint = tmp / "checkpoints/best_model.pth"
        run("-m", "src.evaluate", "--checkpoint", checkpoint, "--no-cm")
        run("-m", "src.predict", "-i", tmp / "data/normal/demo_0000.png", "-c", checkpoint,
            "--cv-analysis", "-o", tmp / "prediction.json")
        run("scripts/export_model.py", "--checkpoint", checkpoint, "--format", "pytorch", "--output-dir", tmp / "export")
        assert json.loads((tmp / "prediction.json").read_text())
        assert (tmp / "export/model.pth").exists()
    print("PASS FrameGuard train/evaluate/predict/export; temporary artifacts removed")


if __name__ == "__main__":
    main()
