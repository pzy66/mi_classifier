# custom_mi 数据目录规范

本目录存放自采 MI 数据，是当前训练主输入目录，也是当前 `schema_version=2` 的 canonical 数据集目录。

新仓库里该目录可能为空，只有 README。这是正常情况，训练前需要先完成采集。

## 1. 当前目录结构

```text
datasets/custom_mi/
|-- collection_manifest.csv
`-- sub-<subject>/
    `-- ses-<session>/
        |-- <run_stem>_board_data.npy
        |-- <run_stem>_board_map.json
        |-- <run_stem>_raw.fif
        |-- <run_stem>_events.csv
        |-- <run_stem>_trials.csv
        |-- <run_stem>_segments.csv
        |-- <run_stem>_session_meta.json
        |-- session_meta_latest.json
        |-- <run_stem>_quality_report.json
        |-- <run_stem>_mi_epochs.npz
        |-- <run_stem>_mi_epochs.meta.json
        |-- <run_stem>_gate_epochs.npz
        |-- <run_stem>_gate_epochs.meta.json
        |-- <run_stem>_artifact_epochs.npz
        |-- <run_stem>_artifact_epochs.meta.json
        |-- <run_stem>_continuous.npz
        `-- <run_stem>_continuous.meta.json
```

`run_stem` 示例：

```text
sub-001_ses-20260401_203000_run-001_tpc-10_n-160_ok-154
```

## 2. 目录里四层数据分别是什么

### 2.1 真源层

- `*_board_data.npy`
- `*_board_map.json`
- `*_raw.fif`
- `*_events.csv`

这一层回答“采集时到底发生了什么”。

### 2.2 语义层

- `*_trials.csv`
- `*_segments.csv`

这一层回答“这些原子事件在协议语义上意味着什么”。

### 2.3 会话描述层

- `*_session_meta.json`
- `session_meta_latest.json`
- `*_quality_report.json`
- `collection_manifest.csv`

这一层回答“这次保存的背景、统计和路径索引是什么”。

### 2.4 派生层

- `*_mi_epochs.npz` + sidecar
- `*_gate_epochs.npz` + sidecar
- `*_artifact_epochs.npz` + sidecar
- `*_continuous.npz` + sidecar

这一层回答“训练和评估要用哪些窗口化数据”。

## 3. 这套目录现在到底保存了哪些数据

### 3.1 `board_data.npy`

特点：

- 裁剪范围是 `session_start` 到 `session_end`
- 保存的是完整 BrainFlow 矩阵，而不只是 EEG
- `dtype=float32`

使用注意：

- EEG 行仍然是 BrainFlow 的原始单位口径，当前会话里为微伏
- marker/timestamp/package 行与 EEG 行语义不同，必须结合 `board_map.json` 解读

### 3.2 `board_map.json`

用途：

- 解释 `board_data.npy` 的每一行
- 记录 crop 边界
- 记录板卡描述和通道映射

关键字段：

- `schema_version`
- `save_index`
- `run_stem`
- `board_id`
- `board_name`
- `board_descr`
- `crop_start_sample`
- `crop_end_sample`
- `selected_eeg_rows`
- `marker_row`
- `timestamp_row`
- `package_num_row`
- `channel_rows`

### 3.3 `raw.fif`

用途：

- 让 MNE / 人工回看更方便

内容：

- 只包含选中的 EEG 通道
- 额外包含一个 `STI 014` stim 通道
- 附加注释，既有点事件，也有带 duration 的区间事件

注意：

- 它不是完整 board 真源
- timestamp/package 等行不会进入 `raw.fif`

### 3.4 `events.csv`

这是原子事件真源。

每一行对应一次 marker 记录，典型字段：

- `event_name`
- `marker_code`
- `trial_id`
- `mi_run_index`
- `run_trial_index`
- `block_index`
- `prompt_index`
- `class_name`
- `command_duration_sec`
- `execution_success`
- `sample_index`
- `absolute_sample_index`
- `elapsed_sec`
- `iso_time`

### 3.5 `trials.csv`

这是 MI trial 摘要表。

适合直接统计：

- 每类 trial 数
- 每类 accepted 数
- cue / imagery 起止 sample
- 被标记坏试次的情况

### 3.6 `segments.csv`

这是协议语义区间表。

当前可能出现的 `segment_type`：

- `trial`
- `baseline`
- `cue`
- `imagery`
- `iti`
- `quality_check`
- `calibration`
- `practice`
- `run_rest`
- `eyes_open_rest`
- `eyes_closed_rest`
- `idle_block`
- `idle_prepare`
- `continuous_block`
- `artifact_block`
- `continuous_prompt`

### 3.7 `session_meta.json`

这是一份总览文件，建议人工审查时优先打开。

它会告诉你：

- 这次保存是哪位被试、哪个 session、哪个 `save_index`
- 采样率和通道映射
- 真源文件和派生文件的相对路径
- trial / segment / gate / artifact / continuous 的数量
- board 数据的 SHA-256
- 本次保存用到的会话参数

### 3.8 `quality_report.json`

这是一份连续信号质量摘要，主要看：

- 整体标准差
- 峰峰值
- 每通道 RMS / peak-to-peak

### 3.9 `mi_epochs.npz`

主分类训练样本。

核心键：

- `X_mi`
- `y_mi`
- `mi_trial_ids`
- 公共元数据键

语义：

- 只包含 accepted 的 imagery 窗口
- 单位统一为 `volt`

### 3.10 `gate_epochs.npz`

gate 训练样本。

核心键：

- `X_gate_pos`
- `X_gate_neg`
- `X_gate_hard_neg`
- `gate_neg_sources`
- `gate_hard_neg_sources`
- 公共元数据键

### 3.11 `artifact_epochs.npz`

artifact rejector 训练样本。

核心键：

- `X_artifact`
- `artifact_labels`
- 公共元数据键

### 3.12 `continuous.npz`

continuous online-like 评估数据。

核心键：

- `X_continuous`
- `continuous_event_labels`
- `continuous_event_samples`
- `continuous_event_end_samples`
- `continuous_block_indices`
- `continuous_prompt_indices`
- `continuous_execution_success`
- `continuous_command_duration_sec`
- `continuous_block_start_samples`
- `continuous_block_end_samples`
- 公共元数据键

## 4. sidecar `*.meta.json` 有什么用

每个派生 `npz` 都配一个 sidecar meta。

它的用途不是重复数组，而是保存：

- 来自哪个 `run_stem`
- 来自哪个 `save_index`
- 来自哪份 `board_data.npy`
- 来自哪些真源文件
- 当前派生策略是什么

这让你可以在不打开大数组的情况下，先确认一个任务文件是不是你想要的那一批数据。

## 5. `collection_manifest.csv` 是什么

这是整个 `datasets/custom_mi` 的全局采集索引。

每一行对应一次保存，记录：

- 保存时间
- `schema_version`
- `subject_id`
- `session_id`
- `save_index`
- `run_stem`
- trial 数量统计
- 采样率
- 通道和类别名称
- 所有真源、语义、派生文件的相对路径

重要约束：

- manifest 中路径全部是相对路径
- 当前不做旧 schema 自动迁移
- 如果目录里还留着旧表头 manifest，继续保存会直接报错

## 6. 单位和口径

这是最容易混的部分。

- `board_data.npy`
  保存 board 原始矩阵口径，EEG 行是微伏，其他行按各自行语义解释
- `raw.fif`
  EEG 已换成 `volt`
- 所有 `npz`
  信号统一保存为 `volt`
- `quality_report.json`
  统计口径主要基于微伏连续信号

## 7. 训练和回看分别会用什么

训练：

- 主分类用 `mi_epochs.npz`
- gate 用 `gate_epochs.npz`
- artifact 用 `artifact_epochs.npz`
- continuous 只做评估

viewer：

- 只看 `mi_epochs.npz`

人工审计：

- 优先看 `session_meta.json`
- 再看 `trials.csv` / `segments.csv`
- 必要时追到 `events.csv` / `raw.fif` / `board_data.npy`

## 8. 归档和复制建议

建议：

- 归档时整体复制 `sub-xxx` 或 `ses-xxx` 目录
- 保留真源层、语义层、会话层和派生层完整数据包

不要这样做：

- 只保留 `mi_epochs.npz`
- 手工改 `run_stem`
- 把 manifest 里的路径改成绝对路径
- 只复制单个文件，不复制同一 `run_stem` 的全套文件

## 9. 最小可训练检查

训练前至少确认：

1. 存在 `*_mi_epochs.npz`
2. 如果要训练 gate / artifact，对应任务文件也存在
3. `session_meta.json` 里的 `accepted_trial_count` 不至于过低
4. 跨保存批次的 `channel_names/class_names/sampling_rate` 一致

## 10. 当前不再保留什么

本目录当前不再产出：

- `*_epochs.npz`

如果你在目录里看到这类文件，说明它们不是当前单轨 schema 产生的文件。
