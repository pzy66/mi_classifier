# 采集保存命名与字段规范

本文件描述当前 `schema_version=2` 的落盘命名、目录结构、manifest 以及各文件字段口径。对应实现位于 [mi_collection.py](C:/Users/P1233/Desktop/brain/mi_classifier/code/shared/src/mi_collection.py)。

## 1. 先统一几个术语

- `session`
  指目录层级里的 `ses-<session_id>`。
- `save_index`
  指同一个 `subject/session` 目录下第几次保存。它会体现在文件名里的 `run-<NNN>`。
- `mi_run_index`
  指一次协议内部第几个 MI run，例如 run 1、run 2、run 3。
- `run_trial_index`
  指某个 `mi_run_index` 内第几个 trial。
- `run_stem`
  指一次保存所有文件共用的统一前缀。

最容易混淆的一点：

- 文件名里的 `run-003` 是 `save_index=3`
- 它不是 `mi_run_index=3`

## 2. run_stem 命名格式

每次保存都会生成一个统一前缀：

```text
sub-<subject>_ses-<session>_run-<NNN>_tpc-<TT>_n-<NNN>_ok-<NNN>
```

字段含义：

- `sub-<subject>`
  安全化后的被试编号。
- `ses-<session>`
  安全化后的会话编号。
- `run-<NNN>`
  三位数 `save_index`。
- `tpc-<TT>`
  `trials_per_class`。
- `n-<NNN>`
  本次保存包含的 trial 总数。
- `ok-<NNN>`
  accepted trial 数。

示例：

```text
sub-001_ses-20260401_203000_run-002_tpc-10_n-160_ok-154
```

## 3. 保存序号如何递增

`save_index` 通过扫描当前 `ses-*` 目录下所有带 `run-XXX` 的文件得到最大值，再加 1。

这意味着：

- 即使上一次保存只写出了一部分文件，只要留下了带 `run-XXX` 的文件，编号也会被占用。
- 这样做是为了避免重试保存时覆盖旧的半成品。

## 4. 目录结构

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

当前不再生成：

- `*_epochs.npz`

## 5. 每个文件的角色

### 5.1 真源相关

- `<run_stem>_board_data.npy`
  裁剪后的完整 BrainFlow 二维矩阵，`dtype=float32`。
- `<run_stem>_board_map.json`
  说明 `board_data.npy` 每一行对应什么物理意义。
- `<run_stem>_raw.fif`
  MNE 可直接加载的工作文件，只含 EEG 和 stim，不含完整 board 行。
- `<run_stem>_events.csv`
  原子事件流。

### 5.2 语义相关

- `<run_stem>_trials.csv`
  trial 级摘要。
- `<run_stem>_segments.csv`
  区间级语义层。

### 5.3 会话描述相关

- `<run_stem>_session_meta.json`
  本次保存的总体描述和文件索引。
- `session_meta_latest.json`
  当前 `ses-*` 目录最近一次保存的指针。
- `<run_stem>_quality_report.json`
  连续信号质量统计。

### 5.4 派生任务相关

- `<run_stem>_mi_epochs.npz`
- `<run_stem>_mi_epochs.meta.json`
- `<run_stem>_gate_epochs.npz`
- `<run_stem>_gate_epochs.meta.json`
- `<run_stem>_artifact_epochs.npz`
- `<run_stem>_artifact_epochs.meta.json`
- `<run_stem>_continuous.npz`
- `<run_stem>_continuous.meta.json`

## 6. 关键字段口径

### 6.1 `events.csv`

字段：

- `event_index`
- `save_index`
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

说明：

- `sample_index` 是裁剪后 session 内的相对 sample。
- `absolute_sample_index` 是裁剪前原始 board buffer 上的 sample。
- `marker_code` 和 `event_name` 是 1:1 对齐保存的。

### 6.2 `trials.csv`

字段：

- `trial_id`
- `save_index`
- `mi_run_index`
- `run_trial_index`
- `class_name`
- `display_name`
- `accepted`
- `cue_onset_sample`
- `imagery_onset_sample`
- `imagery_offset_sample`
- `trial_end_sample`
- `note`

### 6.3 `segments.csv`

字段：

- `segment_id`
- `save_index`
- `segment_type`
- `label`
- `start_sample`
- `end_sample`
- `duration_sec`
- `trial_id`
- `mi_run_index`
- `run_trial_index`
- `block_index`
- `prompt_index`
- `accepted`
- `execution_success`
- `source_start_event`
- `source_end_event`

说明：

- `segments.csv` 是从 `events.csv` 配对得到的语义区间层。
- 它方便审查和下游处理，但 marker 真源仍然是 `events.csv`。

### 6.4 `board_map.json`

关键字段：

- `schema_version`
- `exporter_name`
- `saved_at`
- `subject_id`
- `session_id`
- `save_index`
- `run_stem`
- `board_id`
- `board_name`
- `board_descr`
- `crop_start_sample`
- `crop_end_sample`
- `cropped_sample_count`
- `row_count`
- `selected_eeg_rows`
- `marker_row`
- `timestamp_row`
- `package_num_row`
- `channel_rows`

### 6.5 `session_meta.json`

关键字段：

- `schema_version`
- `exporter_name`
- `saved_at`
- `task_name`
- `subject_id`
- `session_id`
- `save_index`
- `run_stem`
- `session`
- `sampling_rate_hz`
- `recording_type`
- `power_line_frequency_hz`
- `eeg_reference`
- `eeg_ground`
- `preview_filter_applied_to_saved_signal`
- `selected_eeg_rows`
- `marker_row`
- `timestamp_row`
- `package_num_row`
- `sample_count`
- `duration_sec`
- `brainflow_eeg_unit`
- `saved_fif_unit`
- `epochs_unit`
- `quality_report`
- `event_count`
- `segment_count`
- `trials_per_run`
- `mi_run_count`
- `trial_count`
- `accepted_trial_count`
- `rejected_trial_count`
- `mi_epochs_saved`
- `gate_pos_segments`
- `gate_neg_segments`
- `gate_hard_neg_segments`
- `artifact_segments`
- `continuous_blocks`
- `continuous_prompts`
- `raw_preservation_level`
- `board_data_sha256`
- `source_alignment_policy`
- `session_start_wall_time`
- `session_end_wall_time`
- `first_board_timestamp`
- `last_board_timestamp`
- `files`

说明：

- `files` 里的路径全部是相对 `datasets/custom_mi` 的相对路径。
- `session["output_root"]` 固定写 `"."`，避免把本机绝对路径写进元数据。

### 6.6 各任务 `npz`

所有任务 `npz` 都有这些公共键：

- `schema_version`
- `class_names`
- `channel_names`
- `sampling_rate`
- `signal_unit`
- `subject_id`
- `session_id`
- `save_index`
- `run_stem`
- `trials_per_class`
- `mi_run_count`
- `trials_per_run`
- `total_trials`
- `accepted_trials`
- `rejected_trials`
- `created_at`

任务特有键：

- `mi_epochs.npz`
  - `X_mi`
  - `y_mi`
  - `mi_trial_ids`
- `gate_epochs.npz`
  - `X_gate_pos`
  - `X_gate_neg`
  - `X_gate_hard_neg`
  - `gate_neg_sources`
  - `gate_hard_neg_sources`
- `artifact_epochs.npz`
  - `X_artifact`
  - `artifact_labels`
- `continuous.npz`
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

### 6.7 各任务 `*.meta.json`

所有 sidecar meta 都继承这组公共字段：

- `schema_version`
- `exporter_name`
- `created_at`
- `source_run_stem`
- `source_save_index`
- `source_sha256`
- `source_files`
- `subject_id`
- `session_id`

再加各自的：

- `mi_epochs.meta.json`
  - `derivation_name=mi_epochs`
  - `derivation_policy.window_sec`
  - `derivation_policy.source_segment=imagery`
  - `derivation_policy.accepted_trials_only=true`
- `gate_epochs.meta.json`
  - `derivation_name=gate_epochs`
  - `derivation_policy.positive_source`
  - `derivation_policy.negative_sources`
  - `derivation_policy.hard_negative_sources`
  - `derivation_policy.window_sec`
  - `derivation_policy.step_sec`
- `artifact_epochs.meta.json`
  - `derivation_name=artifact_epochs`
  - `derivation_policy.artifact_types`
  - `derivation_policy.window_sec`
  - `derivation_policy.step_sec`
- `continuous.meta.json`
  - `derivation_name=continuous`
  - `derivation_policy.contains_blocks`
  - `derivation_policy.contains_prompt_metadata`
  - `derivation_policy.stores_execution_success`

## 7. manifest 规则

全局索引文件：

- `datasets/custom_mi/collection_manifest.csv`

当前表头字段固定为：

- `saved_at`
- `schema_version`
- `subject_id`
- `session_id`
- `save_index`
- `run_stem`
- `trials_per_class`
- `mi_run_count`
- `trials_per_run`
- `trial_count`
- `accepted_trial_count`
- `rejected_trial_count`
- `sampling_rate_hz`
- `channel_names`
- `class_names`
- `board_data_npy`
- `board_map_json`
- `mi_epochs_npz`
- `mi_epochs_meta_json`
- `gate_epochs_npz`
- `gate_epochs_meta_json`
- `artifact_epochs_npz`
- `artifact_epochs_meta_json`
- `continuous_npz`
- `continuous_meta_json`
- `session_raw_fif`
- `events_csv`
- `trials_csv`
- `segments_csv`
- `session_meta_json`
- `quality_report_json`

重要约束：

- manifest 中所有路径都是相对路径。
- 当前不再做旧 schema manifest 自动迁移。
- 如果你留下了旧表头 manifest，新的保存会直接报 schema mismatch。

## 8. 训练和 viewer 对命名的依赖

训练端依赖：

- `run_stem`
- `save_index`
- 任务文件后缀

viewer 依赖：

- `*_mi_epochs.npz`
- 文件名中的 `run_stem` 结构

因此：

- 不要手工重命名 `run_stem`
- 不要把 `mi`、`gate`、`artifact`、`continuous` 任务文件拆散到别的目录再改名
- 归档时优先整体复制整个 `sub-*` 或 `ses-*` 目录
