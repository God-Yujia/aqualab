`wwe_uie_uieb.pth` 来自 [WWE-UIE 官方仓库](https://github.com/chingheng0808/WWE-UIE) README 的 UIEB 权重入口。

运行 `python tools/bootstrap_assets.py` 下载并按仓库内已有 `source.json` 核验。该文件固定了下载地址、源码版本、架构参数和预期 SHA-256；重复运行准备脚本不会重写这些来源记录。已存在但哈希不符的文件会报错，应恢复对应版本或移走损坏文件后重新下载。实际加载使用 `weights_only=True` 和严格参数匹配。

权重文件已排除出 Git。这里使用作者提供的预训练模型，尚未进行本项目训练或微调。
