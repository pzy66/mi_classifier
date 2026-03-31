# custom_mi 数据目录规范

本目录存放自采 MI 会话数据，是训练主输入目录。

## 1. 目录结构

```text
datasets/custom_mi/
|-- collection_manifest.csv
`-- sub-<subject>/
    `-- ses-<session>/
        |-- <run_stem>_raw.fif
        |-- <run_stem>_events.csv
        |-- <run_stem>_trials.csv
        |-- <run_stem>_session_meta.json
        |-- <run_stem>_quality_report.json
        |-- <run_stem>_mi_epochs.npz
        |-- <run_stem>_gate_epochs.npz
        |-- <run_stem>_artifact_epochs.npz
        |-- <run_stem>_continuous.npz
        `-- <run_stem>_epochs.npz   # legacy，可选
```

`run_stem` 形如：

```text
sub-001_ses-20260331_203000_run-001_tpc-10_n-160_ok-154
```

## 2. 文件用途

- `*_raw.fif`：原始连续 EEG（最核心原始真源）
- `*_events.csv`：事件日志（marker/sample/block/prompt 等）
- `*_trials.csv`：trial 级记录（含 accepted/rejected）
- `*_session_meta.json`：会话参数与统计汇总
- `*_quality_report.json`：按通道质量统计
- `*_mi_epochs.npz`：主分类训练样本
- `*_gate_epochs.npz`：gate 训练样本
- `*_artifact_epochs.npz`：坏窗训练样本
- `*_continuous.npz`：continuous 仿实时评估数据
- `*_epochs.npz`：旧版兼容缓存（可选）

## 3. manifest 作用

`collection_manifest.csv` 每行对应一次保存 run，可快速筛选：

- 哪些 run 完整
- 哪些 run 可用于训练
- 各 run 的试次数、有效数、路径

## 4. 训练前最小检查

1. 至少有 `*_mi_epochs.npz`
2. `channel_names/class_names/sampling_rate` 跨 run 一致
3. 如果希望启用 gate/artifact，需存在对应 task 文件
4. `*_session_meta.json` 中 `accepted_trial_count` 不应异常偏低

## 5. 注意事项

- 不要手动改 `run_stem` 中字段（训练会解析该命名）
- 不要只保留 `*_epochs.npz` 丢弃 `*_raw.fif`（会损失可追溯性）
- 会话内多次保存时，`run-001/run-002/...` 会自动递增
