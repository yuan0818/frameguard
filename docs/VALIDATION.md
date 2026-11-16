# Validation — 2026-11-16

Two runs are recorded. The container run is the later one and covers the
current code; the macOS run predates the regression fixes.

## Linux container run (current code)

- Podman, `python:3.14-slim` image, Linux arm64, CPU only.
- Python 3.14.10, PyTorch 2.12.0+cpu, torchvision 0.27.0+cpu, OpenCV 4.13.0.
- Clean install with the same commands as `.github/workflows/ci.yml`, on a
  copy of the working tree (base commit b599b0b plus the regression fixes).
- `python -m pytest -q`: **17 passed**.
- `python scripts/smoke.py`: **passed**. Covers generation, one CPU training
  epoch, grouped validation, evaluation of a separately generated held-out set
  with `--data-root`, prediction with --cv-analysis, and PyTorch export.
- Regression tests added in this round: letterbox applied before the random
  crop in training, label/source_id matching from one CSV row, duplicate
  basenames in subfolders, frozen-backbone BatchNorm statistics, detector
  config and `--template-dir` wiring, advanced analysis on single images,
  crash check beyond the first video frame, RGB luminance in the rule-based
  crash check, and held-out source-overlap detection.

## macOS run (before the regression fixes)

- macOS ARM64, Python 3.14.4, PyTorch 2.12.0, NumPy 2.4.6.
- Dependencies installed into a new venv without system-site-packages. The
  resolved closure is recorded in requirements-lock-macos-py314.txt.
- `pip check`: passed. Main requirements use constraints.txt for direct versions.
- `python -m pytest -q`: **8 passed** (the suite at that commit).
- `python scripts/smoke.py`: **passed**, using temporary data/weights/config.
- Source syntax and candidate-file credential-pattern scan completed.
- Local dataset at that time: 7,600 training / 1,900 validation samples; 400 / 100
  original source groups, with **zero group overlap**.
- Existing local seven-class best_model.pth loaded with the default config;
  --cv-analysis completed once with no duplicate-argument failure.

No datasets, recordings, model weights, virtual environments or IDE settings
are part of the Git source release.

## Not established by these runs

- GitHub-hosted CI execution, Linux x86_64, Windows runtime or CUDA behavior.
- Independent real-recording quality, full retraining, or production latency.
- Optional labeling UI / ONNX / external dataset download integration.
- Template-based crash and missing-element detection with real templates.

The historical quality scores described in MODEL_CARD.md remain legacy
validation evidence, not new independent test results. Existing user data and
weights were retained; smoke outputs were isolated in temporary directories.
