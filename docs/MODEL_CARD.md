# Model card — experimental v0.1

- Intended use: research and prototyping of media quality checks.
- Not established: production reliability, device-level latency, independent real-world accuracy.
- Source release: code, configs, synthetic smoke generator and tests only; no pretrained weights or datasets.
- Quality metrics: validation is used for selection/tuning. An untouched test set and fixed thresholds are required for final reporting.
- Reproducibility: retain the exact config, seed, source grouping, data version and checkpoint together. GPU and cross-platform bitwise reproducibility are not guaranteed.

## Legacy artifacts

The existing local 7-class image checkpoint records 96.16% validation accuracy.
Its training and validation variants share source images; the old test folder
also duplicates normal training inputs. This is not an independent benchmark.
New training groups by source and stores split metadata. Retraining is required
before a new evaluation claim. A new split applied to an old model does not undo
its prior exposure to those sources.
