# FrameGuard · Experimental v0.1

简体中文 | [English](README.en.md)

基于 PyTorch 的图片/视频画面质量分类实验项目。包含残差 CNN、可选 torchvision
backbone、训练、评估、图片/视频推理以及权重导出。代码采用 [MIT](LICENSE)。

## 能力范围

FrameGuard 判断一张图片或一个视频帧是否存在下列**画面技术故障**。它不理解
画面内容，也不处理音频。

### CNN 分类

每张图片或每个采样帧输出一个类别和各类别的 softmax 概率（单标签，不定位
故障位置）。默认 7 类：

| ID | 类别 | 含义 |
|---|---|---|
| 0 | `normal` | 无下列故障 |
| 1 | `screen_corruption` | 花屏：彩色横纹、随机色块、画面撕裂 |
| 2 | `black_screen` | 黑屏：画面几乎全黑 |
| 3 | `green_screen` | 绿屏：画面整体被绿色覆盖 |
| 4 | `blocking_artifact` | 块效应：编码压缩产生的方块伪影 |
| 5 | `mosaic` | 马赛克：全图或局部像素化 |
| 6 | `color_space_error` | 色彩空间出错：通道错乱、色相偏移、饱和度异常 |

`config/ui.yaml` 另加 3 个 UI 类别：7 `ui_layout_misalignment`（布局错位）、
8 `ui_element_overlap`（元素重叠）、9 `ui_text_overlap`（文本重叠），需自行
准备数据并训练 10 类权重。

- **输入**：图片（jpg、png、bmp、tiff、webp）、视频（mp4、avi、mov、mkv、flv、
  wmv、webm）或目录（递归处理）。模型输入为 224×224 RGB，非方形画面可选 letterbox。
- **视频**：默认每秒取一帧、最多 300 帧（`--frame-interval`、`--max-frames`），
  逐帧分类后按多数投票给出整体类别。`overall_confidence` 是该类别的帧数占比，
  不是模型置信度；短暂故障可能被正常帧淹没，应同时查看逐帧结果和
  `class_distribution`。
- **训练数据**：异常样本由 `scripts/generate_corruptions.py` 在正常图片上合成。
  在真实设备或编码故障上的准确率尚未经独立测试验证，见下文“结果限制”。

### 可选规则检测

以下检测基于 OpenCV 规则，阈值读取配置的 `detectors` 节，与 CNN 类别相互独立，
未经标注数据校准。

| 参数 | 检测项 | 方法 |
|---|---|---|
| `--cv-analysis` | 模糊、过曝/欠曝、边缘截断、大面积空白 | Laplacian 方差；亮度直方图；触及两条以上边界的前景区域；局部低方差连通域 |
| `--ui-analysis` | UI 布局错位、元素重叠、文本重叠 | 边缘与色块检测元素，Sobel + MSER 检测文本；参数按约 224×400 的截图设定 |
| `--advanced-analysis` | 静止帧（仅视频）、Crash 弹窗、UI 元素缺失 | 相邻采样帧灰度直方图相关性；模板匹配，无模板时用画面中心亮矩形规则兜底；元素模板匹配，无模板时跳过 |

### 不支持

- **音频**：不读取音轨，不能检测无声、爆音、音画不同步等问题。
- **画面内容**：不识别物体、人物、场景、文字含义或视频主题。
- **时间类故障**：除静止帧外，不检测卡顿、掉帧或帧率异常。
- **多标签与定位**：每帧只给出一个 CNN 类别，不输出故障区域。

## 安装与验证

在仓库根目录操作，使用 Python 3.14。`constraints.txt` 固定本次验证的直接依赖版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/smoke.py
```

Windows 激活命令为 `.venv\Scripts\activate`。冒烟在临时目录生成数据、训练
一轮、评估（验证划分和另行生成的留出集）、推理和导出，不需要下载数据，也不
覆盖已有模型。CI 在 Linux CPU 上执行相同检查；实际执行结果见 [验证记录](docs/VALIDATION.md)。

## 最小训练示例

```bash
python scripts/generate_synthetic_data.py --out data/demo --sources 20
python -m src.train --config config/demo.yaml --device cpu
python -m src.evaluate --checkpoint outputs/demo/checkpoints/best_model.pth --no-cm
python -m src.predict -i data/demo/normal/demo_0000.png -c outputs/demo/checkpoints/best_model.pth --cv-analysis -o outputs/demo/prediction.json
python scripts/export_model.py --checkpoint outputs/demo/checkpoints/best_model.pth --format pytorch --output-dir outputs/demo/export
```

生成器拒绝覆盖非空目录。示例只验证管线，不代表实际检测质量。正式训练可使用
`config/default.yaml`；训练轮数、路径、增强和 backbone 均可配置。默认 CNN
约 660 万参数，使用 5 个残差卷积块、空间特征压缩和分类头。非方形图像可配置
letterbox，训练与推理必须保持一致。

## 自有数据

把正常图片/视频放在 `data/normal/`，异常数据放在 `data/exception/`，并提供
`data/exception/data.csv`：

```csv
file_name,exception_type,source_id
photo_black_screen_v00.jpg,black_screen,photo
photo_mosaic_v00.jpg,mosaic,photo
```

这里 `source_id=photo` 对应正常原图 `photo.jpg`。同一原图的所有变体、同一
视频的所有帧会留在同一分区。项目生成器的文件名也能自动识别来源；自定义或
重命名派生数据必须填写 source_id。分组划分不保证类别比例，训练前应检查输出
的类别计数。详细规则与数据授权说明见 [DATA.md](docs/DATA.md)。

## 推理和模型文件

`src.predict` 支持单张图片、视频或目录。可选 `--ui-analysis`、`--cv-analysis`、
`--advanced-analysis`；高级模板检测需自行提供模板（配置 `detectors` 节的
`template_dir`，或 `--template-dir` 指向含 `crash/`、`elements/` 子目录的目录）。
检测阈值读取配置的 `detectors` 节。新 checkpoint 保存配置和
训练/验证 source groups，CLI 默认读取其中的配置；`--config` 可显式覆盖。
权重严格匹配结构，类数不匹配会明确报错。

`pytorch` 导出经冒烟验证；`torchscript` 和 `onnx` 为可选实验格式，ONNX 需自行
安装 onnx/onnxscript，不能据此承诺所有运行时兼容。导出元数据包含类别、输入
尺寸、归一化和 letterbox。加载纯 state_dict 时须配套使用相同配置。

## 结果限制与旧权重

仓库不分发数据或预训练权重。已有本地 7 类权重仍可用于推理，但其历史验证
集按样本划分，与训练集共享原图来源；历史 96.16% 不能作为独立测试成绩。
已有 `test/dataset` 也不是独立测试集。

评估默认拒绝缺少分组记录的旧权重。`--allow-legacy-evaluation` 只用于诊断，
不能恢复独立性。应从头按新分组训练，并另留不参与选模型/调参的测试集。
留出集使用与训练数据相同的 `normal/` + `exception/data.csv` 布局，用
`python -m src.evaluate --checkpoint <ckpt> --data-root <留出集目录>` 评估；
来源组与训练/验证重叠时会直接报错（按来源名比对，改名的重复文件无法识别）。
请阅读 [模型说明](docs/MODEL_CARD.md)。
