# 训练数据读取规则（多次采集）

训练脚本 `train_custom_dataset.py` 现在支持同时读取：

- 新命名：`*_epochs.npz`
- 旧命名：`epochs.npz`

## 多次采集（不同 run / 不同试次数）如何处理

- 可以混合放在同一被试目录下训练。
- 每个文件先按 `accepted` 过滤坏试次，再合并。
- 不同文件 trial 数量可以不同（例如 run-001 40 条，run-002 24 条）。

## 必须保持一致的字段

以下字段跨文件必须一致，否则训练会报错：

- `channel_names`
- `class_names`
- `sampling_rate`

## 训练报告新增来源明细

训练输出的 summary（json）新增 `source_records`，可追踪每个来源文件：

- 文件路径
- subject/session/run
- 每类试次数配置（若可解析）
- 原始 trial 数、有效 trial 数、无效 trial 数

这能直接检查“不同训练次数的数据是否被正确识别并纳入训练”。
