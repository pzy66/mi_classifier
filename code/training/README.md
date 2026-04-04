# 训练模块说明

训练模块从 `datasets/custom_mi` 读取当前 `schema_version=2` 数据，产出实时可加载的模型工件。

配套文档：

- [README_DATA_LOADING.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/training/README_DATA_LOADING.md)
- [README_DATA_ADMISSION.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/training/README_DATA_ADMISSION.md)

## 1. 入口

推荐从项目根目录启动：

```powershell
python run_02_training.py
```

等价入口：

```powershell
python code/training/run_training_pycharm.py
```

直接 CLI：

```powershell
python code/training/train_custom_dataset.py
```

说明：

- `run_02_training.py` 和 `run_training_pycharm.py` 默认附带 `--enforce-readiness`。
- 排障时建议直接运行 `train_custom_dataset.py`，能看到完整 traceback。

## 2. 默认输入和输出

默认输入目录：

- `datasets/custom_mi`

默认输出文件：

- 模型：`code/realtime/models/custom_mi_realtime.joblib`
- 训练摘要：`code/training/reports/custom_mi_training_summary.json`
- 旁路 JSON：`code/realtime/models/custom_mi_realtime.json`

如果输入目录没有可训练数据，会报：

- `No trainable data files were found`
- 或 `No usable MI training samples were found`

## 3. 训练只认哪些文件

当前训练只认任务分离文件：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

不再支持：

- `*_epochs.npz`
- `epochs.npz`

训练时会按 `run_stem` 把同一次保存的多任务文件重新组装起来。

## 4. 每类文件分别怎么用

- `*_mi_epochs.npz`
  用于四分类主模型，类别是：
  - `left_hand`
  - `right_hand`
  - `feet`
  - `tongue`
- `*_gate_epochs.npz`
  用于 `control_gate`，也就是 control vs no-control。
- `*_artifact_epochs.npz`
  用于 `artifact_rejector`，也就是 clean vs artifact。
- `*_continuous.npz`
  不参与主分类训练，只用于 online-like 评估。

防泄漏规则：

- gate 负类里来源于 `continuous_*` 的片段，会在训练时自动剔除。

## 5. 训练前会检查什么

跨 run / 跨文件必须一致：

- `channel_names`
- `class_names`
- `sampling_rate`

每个文件还会做可读性检查：

- 文件存在
- 文件大小大于 0
- `np.load(..., allow_pickle=False)` 可正常打开

异常文件不会立刻中断整次训练，而是记录到训练摘要里的 `source_records[*].dropped_reason`。

## 6. 关键默认参数

预处理：

- `4-40Hz` 带通
- `50Hz` 陷波
- `CAR`
- `standardize=False`

主模型窗口：

- `window_secs=2.0,2.5,3.0`
- `window_offset_secs=0.5,0.75`

FBCSP 频带：

- 主 bank：`4-8,8-12,...,36-40`
- `central_fbcsp_lda` 锚点 bank：`8-12,...,28-32`

最小可训练阈值：

- `min_class_trials=5`

稳定比较建议阈值：

- 每类累计 accepted trial `>=30`
- 单个保存批次内每类 accepted trial 建议 `>=8`

对应参数：

- `--recommended-total-class-trials`
- `--recommended-run-class-trials`
- `--enforce-readiness`

## 7. split 策略

默认要求泄漏安全切分。

主任务：

- `session_holdout`
- `group_shuffle`

辅助任务：

- `session_holdout`
- `group_shuffle`
- `aligned_to_main_split`

默认不允许 trial 级回退。只有显式传入下面参数时才允许：

- `--allow-trial-level-fallback`

## 8. 采集参数和训练参数的硬关系

当前默认采集 `imagery_sec=4.0s`，而训练默认最大窗口和 offset 组合要求：

```text
imagery_sec >= max(window_secs) + max(window_offset_secs)
```

默认窗口配置下，最低要求是：

- `3.75s`

因此当前默认采集长度 `4.0s` 是满足训练要求的。

## 9. 默认候选模型

主分类默认候选：

- `central_fbcsp_lda`
- `central_prior_dual_branch_fblight_tcn`
- `riemann+lda`

gate 默认候选：

- `central_gate_fblight`
- `central_prior_gate_fblight`

artifact 默认候选：

- `full8_fblight`

如果环境里没有 torch，深度候选会自动退回 classical 可用候选。

## 10. 常用命令

训练全部数据：

```powershell
python code/training/train_custom_dataset.py
```

只训练某个被试：

```powershell
python code/training/train_custom_dataset.py --subject 001
```

指定窗口和 offset：

```powershell
python code/training/train_custom_dataset.py --window-secs 2.0,2.5 --window-offset-secs 0.5,0.75
```

启用严格准入：

```powershell
python code/training/train_custom_dataset.py --enforce-readiness
```

## 11. 训练结果里优先看什么

训练摘要和控制台输出里优先看：

- `selected_variant`
- `selection_objective`
- `control_gate_enabled`
- `artifact_rejector_enabled`
- `recommended_thresholds`
- `recommended_gate_thresholds`
- `recommended_artifact_thresholds`
- `continuous_online_like`
- `dataset_readiness.ready_for_stable_comparison`
- `dataset_readiness.warnings`
- `source_records`

## 12. 常见失败

### 12.1 `No trainable data files were found`

原因通常是：

- `datasets/custom_mi` 目录下没有任务文件
- `--dataset-root` 指错

快速检查：

```powershell
Get-ChildItem -LiteralPath .\datasets\custom_mi -Recurse -Filter *_mi_epochs.npz
```

### 12.2 `No usable MI training samples were found`

原因通常是：

- 没有任何可用 `*_mi_epochs.npz`
- 发现了文件，但全部被 `dropped_reason` 丢弃

优先看训练摘要中的：

- `source_records[*].dropped_reason`

### 12.3 `channel_names/class_names/sampling_rate` 不一致

说明你在不同次采集中改了通道顺序、类别定义或采样率。当前训练不会自动对齐这些变化，必须先统一采集协议。
