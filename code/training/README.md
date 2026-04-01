# 训练模块说明

训练入口会从 `datasets/custom_mi` 读取采集结果，产出实时可加载工件。

配套规范文档：`README_DATA_LOADING.md`、`README_DATA_ADMISSION.md`。

## 1. 入口

推荐（项目根目录）：

```powershell
python run_02_training.py
```

等价入口：

```powershell
python code/training/run_training_pycharm.py
```

直接使用 CLI：

```powershell
python code/training/train_custom_dataset.py
```

如果终端里没有 `python` 命令，请改用：

```powershell
& 'C:\Users\P1233\miniconda3\envs\MI\python.exe' code/training/train_custom_dataset.py --help
```

说明：`run_02_training.py` / `run_training_pycharm.py` 默认会附带 `--enforce-readiness`。
如果你在纯终端里排障，优先直接跑 `train_custom_dataset.py`，避免弹窗式错误提示阻塞脚本退出。

## 2. 默认输入与输出

默认输入：

- `datasets/custom_mi`

重要：若该目录为空（只有 README，没有 `*_mi_epochs.npz` 等数据文件），训练会直接失败并提示：

- `No trainable data files were found`

默认输出：

- 模型：`code/realtime/models/custom_mi_realtime.joblib`
- 训练摘要：`code/training/reports/custom_mi_training_summary.json`
- 旁路 JSON：`code/realtime/models/custom_mi_realtime.json`

## 3. 训练读取规则

优先读取 task-specific 文件：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

兼容 legacy：

- `*_epochs.npz` / `epochs.npz`

数据要求：

- `channel_names`、`class_names`、`sampling_rate` 跨文件必须一致
- `accepted=0` 的 trial 会自动过滤
- 推荐把 `*_raw.fif / *_events.csv / *_trials.csv` 当作真源保留；`*_npz` 是训练派生缓存

## 4. 训练任务拆分

- 主模型：四类 MI 分类（left/right/feet/tongue）
- `control_gate`：control vs no-control
- `artifact_rejector`：clean vs artifact
- continuous：online-like 离线评估
- gate 负类中来自 `continuous_*` 的片段会在训练前自动剔除，避免评估泄漏

## 5. 默认关键参数

- 预处理：`4-40Hz` 带通 + `50Hz` 陷波 + `CAR` + `standardize=False`
- `window_secs=2.0,2.5,3.0`
- `window_offset_secs=0.5,0.75`
- FBCSP 两层：默认 filter bank `4-8,8-12,...,36-40`；`central_fbcsp_lda` 保守锚点 `8-12,...,28-32`
- `min_class_trials=5`

重要约束：

- 采集 `imagery_sec` 需要满足：`>= max(window_secs)+max(window_offset_secs)`
- 默认至少 `3.75s`，建议采集时用 `4.0s` 以上

训练准入建议（默认只告警，不强制）：

- 每类累计 accepted trial 建议 `>=30`
- 单个 run 内每类 accepted trial 建议 `>=8`
- `run_02_training.py` / `run_training_pycharm.py` 默认会带 `--enforce-readiness`
- 直接手工跑 `train_custom_dataset.py` 时，可用 `--enforce-readiness` 启用不达标即报错
- 可通过 `--recommended-total-class-trials` / `--recommended-run-class-trials` 调整阈值
- 默认强制使用 `session_holdout/group_shuffle/aligned_to_main_split`；若确实需要旧的 trial 级回退，可显式加 `--allow-trial-level-fallback`

## 6. 候选模型

默认主分类候选：

- `central_fbcsp_lda`
- `central_prior_dual_branch_fblight_tcn`
- `riemann+lda`

默认 gate 候选：

- `central_gate_fblight`
- `central_prior_gate_fblight`

默认 artifact 候选：

- `full8_fblight`

若环境无 torch，深度候选会自动回退为 classical 可用组合。

## 7. 常用命令

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

覆盖候选模型：

```powershell
python code/training/train_custom_dataset.py `
  --candidate-names central_fbcsp_lda,riemann+lda `
  --gate-candidate-names central_gate_fblight `
  --artifact-candidate-names full8_fblight
```

启用严格准入：

```powershell
python code/training/train_custom_dataset.py `
  --enforce-readiness `
  --recommended-total-class-trials 30 `
  --recommended-run-class-trials 8
```

## 8. 训练结果中值得重点看

命令行会打印：

- `bank_test_acc / bank_macro_f1 / bank_kappa`
- `control_gate_enabled`
- `artifact_rejector_enabled`
- `recommended_thresholds`
- `recommended_gate_thresholds`
- `recommended_artifact_thresholds`
- `continuous_online_like`（evaluated、mi_acc、no_control_fa）
- `dataset_readiness_ready / dataset_readiness_warnings`
- `selection_objective`（offline+continuous 加权分数与 selected_variant）

建议先确认 gate/rejector 是否可用，再做实时测试。

## 9. 常见失败与处理

### 9.1 `No trainable data files were found`

原因：默认数据目录 `datasets/custom_mi` 没有可训练文件，或 `--dataset-root` 指向错误。

处理：

1. 先采集并确认落盘 `*_mi_epochs.npz`
2. 如数据不在默认目录，显式传 `--dataset-root`

可先用命令快速检查：

```powershell
Get-ChildItem -LiteralPath .\datasets\custom_mi -Recurse -Filter *_mi_epochs.npz
```
