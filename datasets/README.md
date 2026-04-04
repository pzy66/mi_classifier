# datasets 目录说明

本目录只放数据相关内容，不放训练或实时推理代码。

## 1. 子目录含义

- `custom_mi/`
  本项目当前自采 MI 数据的 canonical 根目录。
- `reference_bci_iv_2a/`
  参考公开数据与说明。
- `cache/`
  MNE / MOABB 等缓存。
- `_tmp_*`
  临时检查目录，可按需清理。

## 2. 当前使用约定

1. 日常训练默认读取 `datasets/custom_mi`
2. 采集程序会自动在 `custom_mi` 下按 `sub-<id>/ses-<id>` 写入
3. `custom_mi` 中的 `collection_manifest.csv` 维护全局保存索引
4. 归档时优先整体复制 `sub-*` 或 `ses-*` 目录，不要拆散单文件

## 3. 关于旧数据兼容

当前仓库的数据主流程是 `schema_version=2` 单轨 schema。

这意味着：

- 训练默认读取任务分离 `npz`
- viewer 默认只看 `*_mi_epochs.npz`
- 当前不再为旧 `*_epochs.npz` 或旧 manifest 自动迁移兜底

## 4. 新仓库为什么可能是空目录

新仓库通常不包含你的自采数据，因此：

- `datasets/custom_mi` 初始可能只有 README
- 这是正常现象
- 完成一次采集后，该目录才会生成真实数据树

更详细的目录、字段和文件用途说明见：

- [datasets/custom_mi/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/datasets/custom_mi/README.md)
