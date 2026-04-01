# 数据清洗与训练准入规范

本规范用于保证采集数据在进入训练前可复核、可追溯，并尽量避免离线高分但在线不稳。

## 1. 数据分层（必须）

真源层（长期保留）：

- `*_raw.fif`
- `*_events.csv`
- `*_trials.csv`
- `*_session_meta.json`
- `*_quality_report.json`

训练层（任务专用缓存）：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`

在线验证层：

- `*_continuous.npz`

原则：训练读取 `*_npz`，追溯与重切片回到真源层。

## 2. 任务边界（防止数据串用）

- 主 MI 分类只使用 `*_mi_epochs.npz`
- `control_gate` 只使用 `*_gate_epochs.npz`
- `artifact_rejector` 只使用 `*_artifact_epochs.npz`（clean negatives 由流程拼接）
- `continuous` 仅用于 online-like 评估，不回流主分类训练

额外规则：训练会自动剔除 gate 负类里来源于 `continuous_*` 的片段，避免评估泄漏。

## 3. 训练前检查顺序（建议每次都做）

1. 看 `*_trials.csv`
   - 每类 accepted 是否明显失衡
   - 坏试次是否集中在某一类
2. 看 `*_quality_report.json` 与 `*_raw.fif`
   - 是否有掉电极、饱和、长时漂移、明显工频污染
3. 看训练摘要 `source_records`
   - `mi_class_counts`
   - `gate_neg_dropped_continuous`
   - 各任务片段数（mi/gate/artifact/continuous）

## 4. 准入阈值（与代码参数对齐）

硬阈值（最小可训练）：

- `--min-class-trials`（默认 5）

建议阈值（稳定比较建议）：

- 每类累计 accepted trial 建议 `>=30`
- 单 run 每类 accepted trial 建议 `>=8`

对应参数：

- `--recommended-total-class-trials`（默认 30）
- `--recommended-run-class-trials`（默认 8）
- `--enforce-readiness`（开启后，建议阈值不达标直接报错）

说明：`run_02_training.py` 和 `run_training_pycharm.py` 默认已带 `--enforce-readiness`。

## 5. 切分策略要求

默认要求使用泄漏安全切分：

- 主任务：`session_holdout` 或 `group_shuffle`
- 辅助任务（gate/artifact）：`session_holdout` / `group_shuffle` / `aligned_to_main_split`

若 run/session 无法切分，默认不允许 trial 级回退；如确有需要，显式加：

- `--allow-trial-level-fallback`

## 6. gate 与 artifact 的数据建议

- gate 负类尽量覆盖：`baseline / iti / eyes_open_rest / idle_block / idle_prepare`
- `include_eyes_closed_rest_in_gate_neg` 默认保持 `False`，仅在明确需求下再开启
- artifact 样本尽量多类型、相对均衡（blink/eye/swallow/jaw/head）

## 7. continuous 的使用口径

continuous 只用于评估以下问题：

- prompt 期间是否稳定输出
- no-control 误触发率
- `execution_success=0` 片段对应的是模型问题还是交互执行问题

不要用 continuous 指标替代主分类离线指标，也不要把 continuous 回流进主分类训练。

## 8. 推荐发布前检查清单

1. `dataset_readiness.ready_for_stable_comparison == true`
2. `control_gate.enabled == true`（如目标场景需要 no-control）
3. `artifact_rejector.enabled == true`（如目标场景存在明显伪迹）
4. `continuous_online_like_eval.available == true`
5. 已记录当前版本模型的 `recommended_thresholds` / `recommended_gate_thresholds` / `recommended_artifact_thresholds`
