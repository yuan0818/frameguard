"""Generate a small reproducible demo dataset without replacing existing data."""
from pathlib import Path
import argparse
import csv
import random
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_corruptions import CORRUPTION_REGISTRY


def generate(out, sources=20, seed=42, prefix="demo"):
    out = Path(out)
    if sources < 2:
        raise ValueError("Need at least two independent sources")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    random.seed(seed)
    np.random.seed(seed)
    normal, exception = out / "normal", out / "exception"
    normal.mkdir(parents=True, exist_ok=True)
    exception.mkdir(parents=True, exist_ok=True)
    rows = []
    yy, xx = np.mgrid[:224, :224]
    for i in range(sources):
        phase = np.random.uniform(0, 6, 3)
        img = np.stack([128 + 70 * np.sin(xx / (13 + i) + v) * np.cos(yy / (17 + i)) for v in phase], axis=-1)
        img = np.clip(img + np.random.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        stem = f"{prefix}_{i:04d}"
        Image.fromarray(img).save(normal / f"{stem}.png")
        for name, spec in CORRUPTION_REGISTRY.items():
            filename = f"{stem}_{name}_v00.png"
            Image.fromarray(spec["fn"](img.copy(), 0.8)).save(exception / filename)
            rows.append((filename, name, stem))
    with (exception / "data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "exception_type", "source_id"])
        writer.writerows(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--sources", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="demo", help="source name prefix; use a different one for held-out data")
    args = ap.parse_args()
    print(generate(args.out, args.sources, args.seed, args.prefix))


if __name__ == "__main__":
    main()
