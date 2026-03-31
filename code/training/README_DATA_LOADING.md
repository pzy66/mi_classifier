# 训练数据读取规则（多次采集）

本文件说明 `train_custom_dataset.py` 如何整合多被试/多会话/多 run 数据。

## 1. 支持的文件类型

新格式（推荐）：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

旧格式（兼容）：

- `*_epochs.npz`
- `epochs.npz`

## 2. 目录兼容

训练可识别两种层级：

- 新结构：`sub-<id>/ses-<id>/...`
- 旧结构：`ses-<id>/...`

建议的数据分层：

- 真源层：`*_raw.fif / *_events.csv / *_trials.csv / *_session_meta.json / *_quality_report.json`
- 训练层：`*_mi_epochs.npz / *_gate_epochs.npz / *_artifact_epochs.npz`
- 在线验证层：`*_continuous.npz`

## 3. 统一检查（不通过会报错）

跨文件必须一致：

- `channel_names`
- `class_names`
- `sampling_rate`

若样本长度有细微差异，会按逻辑裁剪到可对齐长度；若差异超容忍范围会拒绝合并。

## 4. accepted / 坏试次处理

- 主 MI 数据只使用有效 trial（accepted）
- legacy 文件中 `accepted=0` 会被过滤
- gate/artifact/continuous 按各自字段加载，不混淆到主分类标签

## 5. gate 负类与泄漏控制

- gate 负类来自 baseline/iti/idle/prepare 等片段
- continuous 来源的 gate 负类会在训练阶段做防泄漏处理（避免评估污染）
- `X_gate_hard_neg` 可为空，流程会自动兼容
- 每个 run 的剔除数量会记录在 `source_records[*].gate_neg_dropped_continuous`

## 6. continuous 数据用途

continuous 仅用于 online-like 评估，输出指标例如：

- `evaluated_prompt_count`
- `mi_prompt_accuracy`
- `no_control_false_activation_rate`

默认不会并入主分类训练样本。

## 7. source_records（训练摘要）

训练摘要中包含 `source_records`，用于追踪每个来源文件：

- 文件相对路径
- subject/session/run 信息（可解析时）
- 各任务片段数量（mi/gate/artifact/continuous）
- 每个 run 的 MI 类别计数（`mi_class_counts`）
- dropped 统计（如 gate continuous-source negatives）

出现“训练样本数量异常”时，先看这里。

## 8. 最小可训练条件

- 至少存在可用的 MI 样本
- 每类 MI 有效 trial 数满足 `--min-class-trials`（默认 5）
- 若要启用 gate/artifact，对应任务样本需满足基本二分类切分

不满足时会降级为仅主分类或直接报错。

## 9. 训练准入（建议阈值）

训练摘要会输出 `dataset_readiness`：

- `total_class_counts`：所有已纳入 run 的每类 accepted trial 总数
- `run_checks`：每个 run 的 `mi_class_counts` 与最小类计数
- `warnings`：未达到建议阈值时的告警

相关参数：

- `--recommended-total-class-trials`（默认 30）
- `--recommended-run-class-trials`（默认 8）
- `--enforce-readiness`（开启后，不达标直接报错）
