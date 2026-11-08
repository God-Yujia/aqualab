`wwe_uie_uieb.pth` 来自 [WWE-UIE 官方仓库](https://github.com/chingheng0808/WWE-UIE) README 的 UIEB 权重入口。

运行 `python tools/bootstrap_assets.py` 下载。`source.json` 记录下载地址、对应源码版本、架构参数和本地文件 SHA-256；实际加载使用 `weights_only=True` 和严格参数匹配。

权重文件已排除出 Git。这里使用作者提供的预训练模型，尚未进行本项目训练或微调。
