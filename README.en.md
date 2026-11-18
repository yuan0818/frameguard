# FrameGuard · Experimental v0.1

[简体中文](README.md) | English

An experimental PyTorch project for classifying the visual quality of images and
video. It includes a residual CNN, optional torchvision backbones, training,
evaluation, image/video inference and weight export. The code is released under
the [MIT](LICENSE) license.

## What it detects

FrameGuard decides whether an image or a video frame shows one of the
**technical picture faults** below. It does not understand what the picture
shows, and it does not process audio.

### CNN classification

Each image or sampled frame gets one class and a softmax probability for every
class (single-label; the fault is not localised). The default model has 7
classes:

| ID | Class | Meaning |
|---|---|---|
| 0 | `normal` | None of the faults below |
| 1 | `screen_corruption` | Corrupted picture: coloured stripes, random blocks, tearing |
| 2 | `black_screen` | Picture almost entirely black |
| 3 | `green_screen` | Picture flooded with green |
| 4 | `blocking_artifact` | Square block artifacts from compression |
| 5 | `mosaic` | Pixelation of the whole frame or part of it |
| 6 | `color_space_error` | Swapped channels, hue shift, abnormal saturation |

`config/ui.yaml` adds 3 UI classes: 7 `ui_layout_misalignment`,
8 `ui_element_overlap` and 9 `ui_text_overlap`. They need your own data and a
10-class model trained on it.

- **Input**: images (jpg, png, bmp, tiff, webp), videos (mp4, avi, mov, mkv, flv,
  wmv, webm) or a directory (processed recursively). The model takes 224×224 RGB;
  non-square pictures can be letterboxed.
- **Video**: by default one frame per second, at most 300 frames
  (`--frame-interval`, `--max-frames`). Each frame is classified and the overall
  class is a majority vote. `overall_confidence` is the share of frames with that
  class, not a model confidence. A short fault can be outvoted by normal frames, so
  check the per-frame results and `class_distribution` as well.
- **Training data**: fault samples are synthesised from normal images by
  `scripts/generate_corruptions.py`. Accuracy on real device or encoder faults has
  not been established on an independent test set; see "Limitations" below.

### Optional rule-based checks

These checks are OpenCV rules. Their thresholds come from the `detectors` section
of the config, they are independent of the CNN class, and they have not been
calibrated against labelled data.

| Flag | Checks | Method |
|---|---|---|
| `--cv-analysis` | Blur, over/under-exposure, edge truncation, large blank areas | Laplacian variance; brightness histogram; foreground regions touching two or more borders; connected low-variance regions |
| `--ui-analysis` | UI misalignment, element overlap, text overlap | Elements from edges and colour regions, text from Sobel + MSER; parameters tuned for screenshots around 224×400 |
| `--advanced-analysis` | Frozen frames (video only), crash dialogs, missing UI elements | Grey-histogram correlation between consecutive sampled frames; template matching, falling back to a bright-rectangle rule in the centre of the frame when there are no templates; element template matching, skipped without templates |

### Not supported

- **Audio**: the soundtrack is never read, so silence, clipping or audio/video
  sync problems are not detected.
- **Picture content**: no recognition of objects, people, scenes, the meaning of
  text, or what a video is about.
- **Timing faults**: apart from frozen frames, no detection of stutter, dropped
  frames or frame-rate problems.
- **Multi-label and localisation**: one CNN class per frame, no fault regions.

## Installation and checks

Work from the repository root with Python 3.14. `constraints.txt` pins the direct
dependency versions used for validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke.py
```

On Windows, activate with `.venv\Scripts\activate`. The smoke run generates data
in a temporary directory, trains for one epoch, evaluates (on the validation split
and on a separately generated held-out set), predicts and exports. It needs no
downloads and does not overwrite existing models. CI runs the same checks on Linux
CPU; see the [validation record](docs/VALIDATION.md) for what actually ran.

## Minimal training example

```bash
python scripts/generate_synthetic_data.py --out data/demo --sources 20
python -m src.train --config config/demo.yaml --device cpu
python -m src.evaluate --checkpoint outputs/demo/checkpoints/best_model.pth --no-cm
python -m src.predict -i data/demo/normal/demo_0000.png -c outputs/demo/checkpoints/best_model.pth --cv-analysis -o outputs/demo/prediction.json
python scripts/export_model.py --checkpoint outputs/demo/checkpoints/best_model.pth --format pytorch --output-dir outputs/demo/export
```

The generator refuses to write into a non-empty directory. The example only
exercises the pipeline; it says nothing about real detection quality. For real
training use `config/default.yaml`; epochs, paths, augmentation and backbone are
all configurable. The default CNN has about 6.6M parameters: 5 residual
convolution blocks, a spatial reduction layer and a classifier head. Non-square
images can be letterboxed; training and inference must use the same setting.

## Your own data

Put normal images/videos in `data/normal/` and faulty ones in `data/exception/`,
with a `data/exception/data.csv`:

```csv
file_name,exception_type,source_id
photo_black_screen_v00.jpg,black_screen,photo
photo_mosaic_v00.jpg,mosaic,photo
```

Here `source_id=photo` refers to the normal original `photo.jpg`. All variants of
one original and all frames of one video stay in the same partition. File names
produced by the project's generators are recognised automatically; custom or
renamed derivatives must have a source_id. Grouped splitting does not keep class
proportions, so check the class counts printed before training. See
[DATA.md](docs/DATA.md) for the full rules and data licensing notes.

## Inference and model files

`src.predict` accepts a single image, a video or a directory. `--ui-analysis`,
`--cv-analysis` and `--advanced-analysis` are optional. Template-based checks need
your own templates: set `template_dir` in the config's `detectors` section, or
point `--template-dir` at a directory containing `crash/` and `elements/`
subdirectories. Detector thresholds come from the config's `detectors` section.
New checkpoints store their config and the train/validation source groups, and
the CLI uses that saved config by default; `--config` overrides it. Weights must
match the architecture exactly, and a class-count mismatch fails with a clear error.

The `pytorch` export is covered by the smoke run. `torchscript` and `onnx` are
optional, experimental formats; ONNX needs onnx/onnxscript installed separately,
and compatibility with every runtime is not guaranteed. Export metadata records the
classes, input size, normalisation and letterbox setting. A bare state_dict must be
loaded with the same config it was trained with.

## Limitations and legacy weights

The repository ships no data and no pretrained weights. An existing local 7-class
checkpoint still works for inference, but its historical validation set was split
by sample and shares source images with training, so its 96.16% is not an
independent test result. The existing `test/dataset` is not an independent test set
either.

Evaluation refuses legacy checkpoints that have no split record by default.
`--allow-legacy-evaluation` is for diagnostics only and does not restore
independence. Retrain from scratch with the grouped split, and keep a separate test
set that plays no part in model selection or tuning. A held-out set uses the same
`normal/` + `exception/data.csv` layout as the training data and is evaluated with
`python -m src.evaluate --checkpoint <ckpt> --data-root <held-out dir>`. It fails
if its source groups overlap training/validation (matched by source name, so renamed
duplicates are not caught). Read the [model card](docs/MODEL_CARD.md).
