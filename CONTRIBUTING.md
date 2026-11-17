# Contributing

Use Python 3.14 and an isolated virtual environment. Install
`requirements-dev.txt`, then run `python -m pytest -q` and `python scripts/smoke.py`.
Tests create temporary inputs and weights; no downloaded datasets are needed.

Keep a source recording/image and all its derived samples in one partition.
Never report validation used for tuning as an independent test benchmark.
Include a regression test for changes to preprocessing, labels, checkpoint
compatibility, or inference. Do not commit recordings, datasets, credentials,
local environments, model weights or generated outputs. Describe data sources
and evaluation limitations when sharing results.
