# 数据清洗与训练准入规范

本规范用于把采集数据按任务分流后，稳定地纳入训练，避免“离线分数高但在线不稳”。

## 1. 数据分层

- 真源层（必须长期保留）：
  - `*_raw.fif`
  - `*_events.csv`
  - `*_trials.csv`
  - `*_session_meta.json`
  - `*_quality_report.json`
- 训练层（任务专用）：
  - `*_mi_epochs.npz`
  - `*_gate_epochs.npz`
  - `*_artifact_epochs.npz`
- 在线验证层：
  - `*_continuous.npz`

原则：训练读取 `*_npz`，追溯和重切回到真源层。

## 2. 任务边界

- 主 MI 分类只用 `*_mi_epochs.npz`。
- `control_gate` 只用 `*_gate_epochs.npz`，且会自动剔除 `continuous_*` 来源的 gate 负类片段。
- `artifact_rejector` 只用 `*_artifact_epochs.npz`（clean negatives 由 gate 正负样本拼接得到）。
- `continuous` 只做 online-like 评估，不回流主分类训练。

## 3. 每次训练前的会话检查顺序

1. 先看 `*_trials.csv`：确认每类 accepted 是否均衡，坏试次是否集中在某一类。
2. 再看 `*_quality_report.json` 与 `*_raw.fif`：排查掉电极、长时间漂移、明显工频污染。
3. 最后看 `source_records`：
   - `mi_class_counts`
   - `gate_neg_dropped_continuous`
   - 各任务片段数量（mi/gate/artifact/continuous）

## 4. 推荐准入阈值

- 最低可训练阈值（硬条件）：`--min-class-trials`（默认 5）。
- 稳定比较阈值（建议条件）：
  - 每类累计 accepted trial `>=30`
  - 单个 run 每类 accepted trial `>=8`

相关参数：

- `--recommended-total-class-trials`（默认 30）
- `--recommended-run-class-trials`（默认 8）
- `--enforce-readiness`（开启后，建议阈值不达标会直接报错）

## 5. gate 与 artifact 的额外建议

- gate 负类优先覆盖：`baseline / iti / eyes_open_rest / idle_block / idle_prepare`。
- `include_eyes_closed_rest_in_gate_neg` 默认保持 `False`，只在有明确需求时再并入闭眼静息。
- artifact 训练关注类型均衡，避免单一类型（如眨眼）占比过高。
- 实验报告中建议单独监控 `tongue` 在启用 rejector 后的误拒情况。

## 6. continuous 的使用口径

- continuous 只用于评估三件事：
  - prompt 期间是否稳定输出
  - no-control 误触发率
  - execution_success=0 片段是模型问题还是交互问题
- 不用 continuous 去宣称主模型训练效果提升。
