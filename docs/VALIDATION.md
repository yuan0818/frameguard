# Validation — 2026-11-16

## Environment

- macOS ARM64, Python 3.14.4, PyTorch 2.12.0, NumPy 2.4.6.
- Dependencies installed into a new venv without system-site-packages.
- Both projects' dev dependencies were installed together; project-specific
  resolved closures are recorded in requirements-lock-macos-py314.txt.
- `pip check`: passed. Main requirements use constraints.txt for direct versions.

## Executed checks

- `python -m pytest -q`: **8 passed**.
- `python scripts/smoke.py`: **passed**, using temporary data/weights/config.
- Source syntax and candidate-file credential-pattern scan completed.
- No datasets, recordings, model weights, virtual environments or IDE settings
  are part of the Git source release.

- Smoke covers generation, one CPU training epoch, grouped validation,
  prediction with --cv-analysis, and PyTorch export.
- Current local dataset: 7,600 training / 1,900 validation samples; 400 / 100
  original source groups, with **zero group overlap**.
- Existing local seven-class best_model.pth loads with the default config;
  --cv-analysis completes once with no duplicate-argument failure.
- Regression tests include grouped variants/video frames, strict class mismatch,
  generator overwrite protection, forward/backward, CLI and export metadata.

## Not established by this run

- GitHub-hosted CI execution, Linux/Windows runtime or CUDA behavior.
- Independent real-recording quality, full retraining, or production latency.
- Optional labeling UI / ONNX / external dataset download integration.

The historical quality scores described in MODEL_CARD.md remain legacy
validation evidence, not new independent test results. Existing user data and
weights were retained; new smoke outputs were isolated in temporary directories.
