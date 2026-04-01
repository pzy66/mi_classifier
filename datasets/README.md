# datasets 目录说明

本目录只放数据相关内容，不放训练/实时代码。

## 子目录含义

- `custom_mi/`：本项目自采 MI 数据（训练主输入）
- `reference_bci_iv_2a/`：参考公开数据及说明文件
- `cache/`：MNE/MOABB 等缓存
- `_tmp_*`：临时运行检查目录（可按需清理）

## 使用约定

1. 日常训练默认读取 `datasets/custom_mi`
2. 采集程序会自动在 `custom_mi` 下按 `sub/ses` 写入
3. 不建议手工改写 `custom_mi` 内文件名，避免破坏 run-stem 解析
4. 需要归档时，优先整体复制 `sub-xxx` 目录，而不是拆文件

补充：仓库通常不包含你的自采数据；`custom_mi` 可能初始只有 README。训练前请先完成采集。

更多结构细节见：`datasets/custom_mi/README.md`
