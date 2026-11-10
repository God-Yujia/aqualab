开发样图通过 `tools/bootstrap_assets.py` 从 [UIEB 官方主页](https://li-chongyi.github.io/proj_benchmark.html)提供的 Challenging-60 下载入口获取。

固定随机种子为 42，默认抽取 10 张，保存在 `data/dev/uieb_c60/`；文件清单、原始成员路径和 SHA-256 记录在 `dev_manifest.json`。

准备脚本按仓库内已有的冻结清单下载并核验，不会用当前文件的新哈希覆盖清单。已有 10 张图时可直接离线校验。不同的 `--seed` 或 `--count` 会被拒绝，避免在保留正式测试集时改变开发集划分。

官方 RAR 压缩包使用公开密码 `5678901`。准备脚本会将 RARLAB 官方便携 UnRAR 下载到本项目缓存目录，并仅解出选中的图片；不安装系统软件。

这 10 张图用于开发和观察，不是未参与开发的最终测试集。C60 没有配对参考图，不计算 PSNR/SSIM。有参考评测使用下述另外冻结的 40 对 UIEB 图片。

桌面流程另用 `605.png` 验收。以种子 `20261107` 从排除这 10 张后的 50 张中抽取，文件名和 SHA-256 均不同。原图保存在 `smoke/uieb_c60/`，抽取规则和来源见 `smoke/manifest.json`。

正式评测以 `test_manifest.json` 为准，其中有 40 对 UIEB 配对图和 49 张 C60 图。安装项目依赖后，在项目目录运行 `python tools/fetch_test_assets.py`。工具从发布的归档补齐缺失文件，并校验 129 个文件的 SHA-256、原始与处理尺寸，以及 89 张输入的场景标签。使用 `--verify-only` 可离线核查，已有的原始归档可通过 `--paired-zip` 和 `--c60-rar` 指定。 [WWE-UIE 数据划分入口](https://github.com/chingheng0808/WWE-UIE#dataset-preparation)

数据集仅限学术、非商业用途，官方禁止重新分发。图像和下载缓存已由 `.gitignore` 排除；请从官方来源自行获取。
