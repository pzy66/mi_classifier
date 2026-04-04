# 训练数据读取规则

本文件说明 [train_custom_dataset.py](C:/Users/P1233/Desktop/brain/mi_classifier/code/training/train_custom_dataset.py) 当前如何发现、分组、加载多被试、多会话、多次保存数据。

## 1. 文件发现规则

训练会递归扫描 `dataset_root` 下所有 `*.npz`，只识别下面四种后缀：

- `*_mi_epochs.npz` -> `mi`
- `*_gate_epochs.npz` -> `gate`
- `*_artifact_epochs.npz` -> `artifact`
- `*_continuous.npz` -> `continuous`

不再识别：

- `*_epochs.npz`
- `epochs.npz`

扫描结果会按 `run_stem` 分组，也就是把同一次保存的 `mi/gate/artifact/continuous` 重新并成一个来源 run。

## 2. `run_stem` 和 `save_index`

训练从文件名里解析：

- `subject_id`
- `session_id`
- `save_index`
- `trials_per_class`
- `total_trials_tag`
- `accepted_trials_tag`

说明：

- `save_index` 来自文件名中的 `run-<NNN>`
- 它表示一次保存序号，不是协议内部第几个 MI run

## 3. `--subject` 过滤规则

`--subject` 同时支持：

- `001`
- `sub-001`

过滤时既会检查路径层级，也会检查完整路径字符串，因此只要命名遵守当前目录结构，过滤结果会稳定。

## 4. 读取有效性检查

每个任务文件在真正进入加载前，都会先做可读性检查：

- 文件存在
- 是普通文件
- 大小大于 0
- `np.load(..., allow_pickle=False)` 可正常打开

如果失败：

- 当前文件会被跳过
- 原因会写进 `source_records[*].dropped_reason`
- 不会因为单个坏文件直接中断全部训练

## 5. 元数据一致性要求

跨 run / 跨文件必须一致：

- `channel_names`
- `class_names`
- `sampling_rate`

不一致时会直接报错，拒绝合并。

原因很简单：

- 通道顺序一变，空间特征就失真
- 类别定义一变，标签语义就不再可比
- 采样率一变，窗口长度和滤波口径就不再一致

## 6. MI 数据怎么读取

`mi_epochs.npz` 当前读取这些核心键：

- `X_mi`
- `y_mi`
- `mi_trial_ids`
- 公共元数据键

信号口径：

- 当前保存端写入 `signal_unit=volt`
- 加载器仍会按 `signal_unit` 做保险转换，最终统一转成 `volt`

加载后：

- 记录 `mi_trials`
- 统计 `mi_class_counts`
- 为每个 trial 生成稳定的 `trial_keys`

## 7. gate 数据怎么读取

`gate_epochs.npz` 读取：

- `X_gate_pos`
- `X_gate_neg`
- `X_gate_hard_neg`
- `gate_neg_sources`
- `gate_hard_neg_sources`

然后执行一条固定防泄漏规则：

- 所有来源名以 `continuous` 开头的 gate 负类窗口，会被自动剔除

剔除数量会记录到：

- `source_records[*].gate_neg_dropped_continuous`
- 汇总后的 readiness 统计

## 8. artifact 数据怎么读取

`artifact_epochs.npz` 读取：

- `X_artifact`
- `artifact_labels`

它和主分类数据完全分开，不会混入主分类标签。

## 9. continuous 数据怎么读取

`continuous.npz` 读取：

- `X_continuous`
- `continuous_event_labels`
- `continuous_event_samples`
- `continuous_block_start_samples`
- `continuous_block_end_samples`
- `continuous_block_indices`

内部处理时：

- 文件里的 `continuous_block_indices` 是 1-based 编号
- 加载后会转成内部使用的 0-based block index
- 如果某个 prompt 没有显式 block index，但 block 起止 sample 存在，加载器会按 sample 落点自动回填 block 归属

continuous 最终只用于 online-like 评估，不参与主分类训练。

## 10. 统一样本长度的规则

不同次保存出来的窗口长度理论上应该一致，但训练端仍会做保护：

1. 先统计各来源 `sample_count`
2. 找出现次数最多的参考长度 `reference_samples`
3. 过短的 run 直接丢弃，并写入：
   - `dropped_reason=too_short_for_common_epoch_length:<sample_count><<reference_samples>`
4. 保留下来的 run 再统一裁到共同长度

主分类、gate、artifact 各自都会在合并前对齐长度。

## 11. `source_records` 会记录什么

每个来源 run 都会留下一个 `source_record`，关键字段包括：

- `run_stem`
- `subject_id`
- `session_id`
- `save_index`
- `schema_version`
- `mi_file`
- `gate_file`
- `artifact_file`
- `continuous_file`
- `mi_trials`
- `mi_class_counts`
- `gate_pos_segments`
- `gate_neg_segments`
- `gate_hard_neg_segments`
- `gate_neg_dropped_continuous`
- `artifact_segments`
- `continuous_blocks`
- `continuous_prompts`
- `dropped_reason`

排查训练样本异常时，优先看这里。

## 12. `load_custom_epochs` 和 `load_custom_task_datasets` 的区别

- `load_custom_epochs`
  只扫描并合并 `*_mi_epochs.npz`
- `load_custom_task_datasets`
  会同时发现并加载 `mi/gate/artifact/continuous` 四类任务文件

主训练流程使用的是：

- `load_custom_task_datasets`

## 13. 最小可训练条件

至少满足：

- 存在可用 `*_mi_epochs.npz`
- 每类 MI trial 数达到 `--min-class-trials`

如果 gate 或 artifact 数据不足：

- 可能降级为仅主分类
- 也可能在某些严格路径下直接报错，具体取决于当前训练配置

## 14. readiness 输出怎么看

训练摘要会输出 `dataset_readiness`，重点字段包括：

- `total_class_counts`
- `run_checks`
- `warnings`
- `ready_for_stable_comparison`

常用相关参数：

- `--recommended-total-class-trials`
- `--recommended-run-class-trials`
- `--enforce-readiness`

## 15. 当前不再兼容什么

下面这些口径已经退出当前主流程：

- legacy `*_epochs.npz`
- legacy `epochs.npz`
- 依赖 `accepted` 字段去二次过滤旧主文件
- 依赖 `continuous_events` JSON payload 解析 continuous block
- `run_index` 作为保存编号的旧叫法

现在统一用：

- `save_index`
- `mi_run_index`
- 任务分离 `npz`
