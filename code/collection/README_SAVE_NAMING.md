# 采集保存命名规则（新增）

为了解决“多次采集文件难区分”的问题，采集模块现在会自动使用**带编号**的文件名。

## 新命名格式

每次停止并保存后，会在当前 `sub-xxx/ses-xxx/` 下生成一组同前缀文件：

`sub-<subject>_ses-<session>_run-<编号>_tpc-<每类试次数>_n-<本次总试次>_ok-<本次有效试次>_<类型后缀>`

示例：

- `sub-001_ses-20260329_210000_run-002_tpc-10_n-040_ok-038_epochs.npz`
- `sub-001_ses-20260329_210000_run-002_tpc-10_n-040_ok-038_raw.fif`
- `sub-001_ses-20260329_210000_run-002_tpc-10_n-040_ok-038_session_meta.json`

## 字段含义

- `run-002`：同一 `subject/session` 下第 2 次采集
- `tpc-10`：每类目标试次数（trials per class）
- `n-040`：本次采集总试次
- `ok-038`：本次有效试次

## 兼容性

- 训练程序已兼容读取：
  - 新格式：`*_epochs.npz`
  - 旧格式：`epochs.npz`
- 可视化程序也已兼容以上两种格式。

## 新增数据索引（推荐用于管理多次采集）

采集保存时会自动在 `output_root`（默认 `datasets/custom_mi`）下追加：

- `collection_manifest.csv`

每一行对应一次采集 run，包含：
- `subject_id / session_id / run_index / run_stem`
- `trials_per_class / trial_count / accepted_trial_count / rejected_trial_count`
- `epochs_npz / session_raw_fif / events_csv / trials_csv / session_meta_json`

这样可以快速筛选“哪次采集可用于训练”，不需要手工翻目录。
