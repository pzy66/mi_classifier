# 数据清洗与训练准入规范

本规范用于保证当前 `schema_version=2` 采集数据在进入训练前可复核、可追溯，并尽量避免“离线分数高但在线不稳”。

## 1. 数据分层

### 1.1 真源层

这层必须长期保留。

- `*_board_data.npy`
- `*_board_map.json`
- `*_raw.fif`
- `*_events.csv`

解释：

- `board_data.npy` 是当前最接近板卡输出的保存物。
- `board_map.json` 负责解释 `board_data.npy` 的行语义。
- `raw.fif` 是 MNE 友好的工作视图，不等于完整真源。
- `events.csv` 是 marker 对齐后的原子事件真源。

### 1.2 语义层

这层把真源转成更好检查的结构。

- `*_trials.csv`
- `*_segments.csv`

解释：

- `trials.csv` 关注 MI trial。
- `segments.csv` 关注所有协议区间。

### 1.3 会话描述层

- `*_session_meta.json`
- `session_meta_latest.json`
- `*_quality_report.json`
- `collection_manifest.csv`

### 1.4 派生层

这层给训练和评估使用。

- `*_mi_epochs.npz` + `*_mi_epochs.meta.json`
- `*_gate_epochs.npz` + `*_gate_epochs.meta.json`
- `*_artifact_epochs.npz` + `*_artifact_epochs.meta.json`
- `*_continuous.npz` + `*_continuous.meta.json`

原则：

- 训练读取派生层
- 追溯、重切片、审计回到真源层和语义层

## 2. 训练任务边界

当前边界必须严格遵守：

- 主 MI 分类只使用 `*_mi_epochs.npz`
- gate 只使用 `*_gate_epochs.npz`
- artifact rejector 只使用 `*_artifact_epochs.npz`
- continuous 仅用于 online-like 评估，不并入主分类训练

额外规则：

- 训练阶段会自动剔除 gate 负类里来源于 `continuous_*` 的片段，避免评估泄漏。

## 3. 每次训练前建议检查什么

建议顺序：

1. 看 `*_session_meta.json`
2. 看 `*_trials.csv`
3. 看 `*_segments.csv`
4. 看 `*_quality_report.json`
5. 必要时看 `*_raw.fif`
6. 必要时追到 `*_events.csv` 和 `*_board_data.npy`
7. 训练后再看摘要里的 `source_records`

重点问题：

- 各类 accepted trial 是否严重失衡
- rejected trial 是否集中在某一类或某个 `mi_run_index`
- 是否存在掉线、饱和、长时漂移、明显工频污染
- `gate_neg_dropped_continuous` 是否异常偏高
- continuous 中 `execution_success=0` 是否集中出现在某几类提示

## 4. 各层分别适合回答什么问题

### 4.1 `board_data.npy + board_map.json`

适合回答：

- 当前保存到底保留了哪些 board 行
- timestamp / marker / package 行是否存在
- 原始采样序列是否有异常

不适合直接回答：

- 哪个 MI trial 被拒绝
- 哪个 block 对应什么协议语义

### 4.2 `events.csv`

适合回答：

- marker 是否 1:1 对齐
- 某个事件是否真的被写入
- 某个 prompt / block / trial 的起止 sample 是多少

### 4.3 `trials.csv`

适合回答：

- 哪些 trial 被 accepted / rejected
- 每类 accepted trial 数量
- cue / imagery / trial_end 的 sample 是否合理

### 4.4 `segments.csv`

适合回答：

- 哪些 interval 被导出成语义区间
- baseline / cue / imagery / iti 是否拼接正确
- calibration / idle / continuous / artifact block 的范围是否合理

### 4.5 `quality_report.json`

适合回答：

- 连续信号总体振幅是否异常
- 某些通道是否明显偏离

### 4.6 派生 `npz`

适合回答：

- 当前训练到底用了多少窗口
- 不同任务的数据量是否足够
- 窗口来源是否符合预期

## 5. 准入阈值

硬阈值：

- `--min-class-trials`，默认 `5`

建议阈值：

- 每类累计 accepted trial 建议 `>=30`
- 单次保存或单次稳定批次内每类 accepted trial 建议 `>=8`

对应参数：

- `--recommended-total-class-trials`
- `--recommended-run-class-trials`
- `--enforce-readiness`

说明：

- `run_02_training.py`
- `run_training_pycharm.py`

这两个入口默认都带 `--enforce-readiness`。

## 6. gate 数据的准入口径

当前保存端会把以下来源切成 gate 负类候选：

- `baseline`
- `iti`
- `eyes_open_rest`
- `eyes_closed_rest`，仅当 `include_eyes_closed_rest_in_gate_neg=True`
- `idle_block`
- `idle_prepare`
- `continuous_no_control`

当前保存端会把 accepted imagery 作为 gate 正类。

当前保存端会把以下来源切成 hard negative：

- `artifact_block` 各类伪迹区间
- `rejected_trial`

准入建议：

- gate 负类不要只有单一来源
- 如果 `idle_block` 和 `idle_prepare` 全部缺失，需确认是否协议本身被关闭
- 若 `continuous_no_control` 比例过高，训练时虽然会剔除，但说明采集协议口径已经偏向 continuous

## 7. artifact 数据的准入口径

当前 `artifact_labels` 可能包括：

- `eye_movement`
- `blink`
- `swallow`
- `jaw`
- `head_motion`
- `rejected_trial`

准入建议：

- 各类伪迹尽量都覆盖
- 不要让 `rejected_trial` 占绝对多数，否则 rejector 会更像“拒绝任意异常 trial”而不是识别具体伪迹窗口

## 8. continuous 数据的准入口径

`continuous.npz` 用于回答：

- prompt 期间输出是否稳定
- `no_control` 误触发率是否可接受
- 交互命令执行失败到底是模型问题还是执行问题

关键字段：

- `continuous_event_labels`
- `continuous_block_indices`
- `continuous_prompt_indices`
- `continuous_execution_success`
- `continuous_command_duration_sec`

解释：

- `continuous_execution_success`
  - `1` 表示操作成功
  - `0` 表示命令失败
  - `-1` 表示未知或未记录

不要做的事：

- 不要把 continuous 指标直接替代主分类离线指标
- 不要把 continuous 直接并入主分类训练

## 9. 发布前检查清单

1. `dataset_readiness.ready_for_stable_comparison == true`
2. `control_gate_enabled == true`，如果目标场景需要 no-control
3. `artifact_rejector_enabled == true`，如果目标场景伪迹显著
4. `continuous_online_like_eval.available == true`
5. 已记录推荐阈值：
   - 主分类
   - gate
   - artifact
6. 抽查过至少一批真源层文件，确认派生层不是建立在错位 marker 上
