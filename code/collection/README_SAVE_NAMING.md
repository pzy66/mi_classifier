# 采集保存命名规则

本文件说明 `code/shared/src/mi_collection.py` 的实际落盘命名与索引规则。

## 1. run-stem 命名格式

每次保存都会生成统一前缀（run stem）：

```text
sub-<subject>_ses-<session>_run-<NNN>_tpc-<TT>_n-<NNN>_ok-<NNN>
```

字段含义：

- `run-<NNN>`：同一 `subject/session` 下第几次保存（自动递增）
- `tpc-<TT>`：`trials_per_class`
- `n-<NNN>`：本次会话总 trial 数
- `ok-<NNN>`：本次 accepted trial 数

示例：

```text
sub-001_ses-20260331_203000_run-002_tpc-10_n-160_ok-154
```

## 2. 同一 run 的输出文件

以上面 run stem 为前缀，会生成：

- `*_raw.fif`
- `*_events.csv`
- `*_trials.csv`
- `*_session_meta.json`
- `session_meta_latest.json`（不带 run-stem，指向当前 `ses-*` 目录内最近一次保存）
- `*_quality_report.json`
- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`
- `*_epochs.npz`（可选 legacy）

## 3. 目录层级

```text
datasets/custom_mi/
`-- sub-<subject>/
    `-- ses-<session>/
        |-- <run_stem>_raw.fif
        |-- <run_stem>_mi_epochs.npz
        `-- ...
```

`subject` 和 `session` 会做文件名安全化处理（允许输入中文，但落盘会转成安全 token）。

## 4. 全局采集索引（manifest）

每次保存会追加到：

- `datasets/custom_mi/collection_manifest.csv`

当前 manifest 关键列包括：

- `saved_at`
- `subject_id / session_id / run_index / run_stem`
- `trials_per_class / trial_count / accepted_trial_count / rejected_trial_count`
- `sampling_rate_hz / channel_names / class_names`
- `mi_epochs_npz / gate_epochs_npz / artifact_epochs_npz / continuous_npz / epochs_npz`
- `session_raw_fif / events_csv / trials_csv / session_meta_json`

## 5. 旧版 manifest 兼容

若检测到旧 schema：

1. 旧文件会被自动备份为 `*_legacy_schema_<timestamp>.csv`
2. 新文件按当前 schema 重建后继续追加

因此可以直接升级，不需要手工改历史数据。

## 6. 训练兼容性

训练端兼容：

- 新格式：`*_mi_epochs.npz` 等 task-specific 文件
- 旧格式：`*_epochs.npz` / `epochs.npz`

建议优先使用新格式，因为 gate/artifact/continuous 信息更完整。
