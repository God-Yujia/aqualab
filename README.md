**AquaLab · 水下图像增强课程项目**

程序读取照片，经 WWE-UIE 预训练模型增强，可在保存前对增强幅度做局部控制。桌面版提供原图与增强结果预览、强度调节、系数图和 PNG 导出。实验结果通过命令行独立生成。

**启动桌面版**

本机已用 RTX 4060 Laptop GPU 验证。在项目目录启动窗口：

```powershell
.\.venv\Scripts\python.exe app.py
```

点“打开照片”，用滑杆调节后点“导出 PNG”。每张照片只做一次模型预测，拖动滑杆时复用预测结果。预览和导出保持宽高比，最长边为 768 像素。0% 显示处理尺寸下的原图，100% 对应内部实验的 `local@1.0`。使用 `--device cpu` 可在 CPU 上运行。

默认强度为 **100%**。在 40 对 UIEB 配对图的 60%、80%、100% 三档比较中，这一档的平均 SSIM 和 PSNR 均最高。默认值和选参依据记录在 `data/test_manifest.json`。 [三档实测](docs/day5.md)

勾选“显示局部控制”可并排查看实际系数，越亮表示该像素保留的模型改变量越多。系数的黑白端点分别为 0% 和 100%，滑杆设为 60% 时，未额外限幅的像素也只到 60% 灰度。使用“导出系数图”另存一张灰度 PNG。

本机可直接加载另行抽取的 UIEB C60 验收图 `605.png`：

```powershell
.\.venv\Scripts\python.exe app.py --input data\smoke\uieb_c60\605.png
```

已完成的 10 张开发样图结果位于 `outputs/first_run/`，打开 `contact_sheet.jpg` 查看总览。每个画面左侧为输入、右侧为原始预训练模型的结果。验证记录见 `docs/step1.md`。

**安装（Windows PowerShell，Python 3.10）**

在本目录打开终端。安装到独立 `.venv`，无需修改全局 Python，也无需激活环境。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cuda.txt
.\.venv\Scripts\python.exe tools\bootstrap_assets.py
```

GPU 环境使用 PyTorch 2.4.1 + CUDA 11.8，接近上游推荐的 PyTorch 2.4.0 环境。传统基线在 CPU 上使用 OpenCV。 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v241)

其他电脑使用 CPU 时，将 GPU 版 PyTorch 的安装命令改为：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

**运行**

处理下载的 10 张开发图片：

```powershell
.\.venv\Scripts\python.exe infer.py --output outputs\first_run
```

处理自己的图片或文件夹（支持中文路径）：

```powershell
.\.venv\Scripts\python.exe infer.py --input "D:\照片\海底.jpg" --output outputs\my_photo
.\.venv\Scripts\python.exe infer.py --input "D:\照片\潜水" --output outputs\my_batch
```

默认自动选择 GPU，最长边限制为 768 像素，保持宽高比，输出尺寸与实际处理尺寸一致。使用 `--max-side 0` 保持原始分辨率；显存需求随尺寸增加。CPU 验证可使用 `--device cpu --max-side 256`。

每次运行产生：

```text
outputs/first_run/
  enhanced/           增强后的 PNG
  comparisons/        输入与输出并排图，供开发观察
  contact_sheet.jpg   样图总览
  run.json            参数、环境、输入哈希、输出路径
  timings.csv         内部耗时记录
```

`inference_ms` 记录预热之后的模型推理耗时，传统基线的这一列为空。`processing_and_save_ms` 记录从读图到保存结果的时间，模型组的第一张还包含预热。

**传统对照基线**

灰度世界白平衡将三个 RGB 通道的均值调向同一目标，随后对 Lab 的 L 通道做 CLAHE。初始参数为 8×8 分块和 2.0 的裁剪限制。基线和模型组共用读图、尺寸和保存规则。 [OpenCV 的 CLAHE 示例](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe infer.py --method traditional --output outputs\day2\traditional
```

开发结果见 `outputs/day2/traditional/contact_sheet.jpg`。例如，`432` 经处理后水体偏红，背景条纹更明显，可用于分析全局白平衡和局部对比度增强的局限。

**局部控制原型**

`--method local` 在量化前沿着原图到模型输出的 RGB 方向融合。每个像素的三个通道共用一个系数。当请求的改变量接近该方向可用余量的 90% 时，开始逐渐减小系数。使用 3×3 灰度闭运算和以原图 HSV 的 V 为引导的滤波平滑系数图，再应用每个像素自身的边界约束。

`--strength` 在 0 到 1 之间，0 输出处理尺寸下的原图，1 请求模型的全量增强，由局部规则自动限幅。`strength_maps/` 中的灰度图记录实际系数，越亮表示保留的模型改变量越多。

内部对照包括默认的 `--method clip`（直接裁剪）、`--method fixed --strength 0.8`（全图同一系数）以及 `--method normalize`（按 Guo 等的 URanker 式（5）只拉伸越界通道）。四组开发结果和逐图统计位于 `outputs/local_control_v1/`。例如，`12709` 的零通道端点比例由 1.22% 增至 1.39%，新增的触底像素在输入中已有通道为零。

局部融合、亮度引导和色域边界的渐进压缩分别参考了 Tao 等、Ning 等和 Xu 等的方法。 [Tao 等](https://doi.org/10.1109/ACCESS.2020.3040505) · [Ning 等](https://doi.org/10.3389/fmars.2023.1249351) · [Xu 等](https://doi.org/10.1364/BOE.439324)

**内部平滑消融**

两条命令分别运行完整模块和仅逐像素限幅。后者跳过灰度闭运算与引导滤波。

```powershell
.\.venv\Scripts\python.exe infer.py --method local --output outputs\day3\smoothed
.\.venv\Scripts\python.exe infer.py --method local --no-smoothing --output outputs\day3\no_smoothing
```

10 张开发图的逐图统计和局部对照见 `outputs/day3/ablation/`。平滑使 `12972` 的触底像素占比从 5.33% 降到 3.51%，也使暗场 `12709` 从 1.22% 升到 1.39%。这里的触底指导出 PNG 至少一个 RGB 通道为 0。单图流程验收结果见 `outputs/day3/smoke/`。

**代码中的一张图片如何流动**

`infer.py` 先读取图片并处理拍摄方向，转换成 RGB；将像素从 0–255 转为 0–1，排列成模型需要的 `[1, 3, 高, 宽]` 张量。必要时保持宽高比缩小并补齐边缘，然后送入冻结的 WWE-UIE。最后去掉补边，将输出限制到合法范围、转回 8 位 RGB，再保存成 PNG。

`UnderwaterEnhancer` 负责加载和调用模型。桌面版和命令行共用模型与局部控制模块，命令行另外负责批处理与实验记录。

**来源与复现范围**

- 主模型：[WWE-UIE 官方仓库](https://github.com/chingheng0808/WWE-UIE)，WACV 2026，Apache-2.0。源码固定在 `fd0697558203f9f849800cc13b804116bb3f921c`；原始 `model.py`、原始许可和来源记录放在 `third_party/wwe_uie/`。
- 权重：官方 README 的 UIEB 权重链接。使用完整严格匹配加载，未训练或微调。
- 样图：[UIEB 官方主页](https://li-chongyi.github.io/proj_benchmark.html)的 Challenging-60；固定种子 42 抽取 10 张，清单见 `data/dev_manifest.json`。它们作为开发样本，后续最终测试另留图像。
- C60 没有参考图，本阶段不计算 PSNR/SSIM。UIEB 官方限制为学术、非商业用途并禁止重新分发；图像、权重、缓存和输出均已排除出 Git。
- 这里采用保持宽高比的推理预处理，上游部分基准测试会统一缩放到 256×256。当前结果用于流程验证，不代表对论文分数的复现。

**复现正式评测**

准备模型权重和 Python 依赖后，在项目目录执行以下命令。第一条补齐正式样本并校验，第二条按冻结口径生成内部评测结果和三档强度曲线。

```powershell
.\.venv\Scripts\python.exe tools\fetch_test_assets.py
.\.venv\Scripts\python.exe tools\evaluate.py --device cuda
```

用 `--verify-only` 可以离线校验已有数据。评测固定使用 scikit-image 0.24.0，逐图 CSV 和汇总 JSON 保存在 `outputs/day5/` 下。 [实验冻结记录](docs/day4.md)

安装开发依赖后运行性质和回归测试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

区间、零强度和合成图回归在 CPU 上运行。历史开发图的逐像素回归使用原来的 CUDA 推理路径，并读取 `bootstrap_assets.py` 准备的模型和开发图。
