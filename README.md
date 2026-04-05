# MI Classifier

本项目聚焦运动想象 EEG（MI）端到端流程，不包含 SSVEP。当前仓库的主链路是：

```text
采集 -> 保存真源/语义/派生数据 -> 训练 -> 实时推理 -> 回看
```

仓库当前以 `schema_version=2` 的单轨数据格式为主，训练和回看都围绕这一套目录结构与任务分离文件工作。

## 项目范围

- 采集 MI 会话并落盘为可追溯数据包
- 从 `datasets/custom_mi` 训练实时可加载工件
- 加载训练产物做实时推理
- 对单次 run bundle 做保存后回看与审计

当前不做：

- SSVEP 流程
- 旧 `*_epochs.npz` 数据格式兼容
- 以 README 替代模块级说明文档

## 快速开始

推荐在 `MI` 环境中运行：

```powershell
conda activate MI
pip install -r requirements.txt
pip install -r requirements-realtime.txt
```

如果需要深度候选模型，再额外安装：

```powershell
pip install -r requirements-deep.txt
```

如果终端里没有 `python` 命令，可以直接使用解释器绝对路径：

```powershell
& 'C:\Users\P1233\miniconda3\envs\MI\python.exe' run_02_training.py
```

## 常用入口

主流程：

```powershell
python run_01_collection_only.py
python run_02_training.py
python run_03_realtime_infer.py
```

辅助工具：

```powershell
python run_04_view_collected_npz.py
python run_05_channel_monitor.py
```

## 仓库结构

- `code/collection`
  MI 采集 UI、会话流程、marker 记录、落盘导出
- `code/training`
  读取 `schema_version=2` 数据并训练实时模型
- `code/realtime`
  加载训练工件并执行在线推理
- `code/viewer`
  对单次保存结果做 run-bundle 回看
- `datasets/custom_mi`
  自采 MI 数据目录，也是训练默认输入目录

## 端到端流程

### 1. 采集

采集模块负责：

- 驱动会话 UI 和阶段流转
- 写入原子事件与语义区间
- 保存连续 board 数据、工作视图和训练派生文件

默认会话顺序：

1. `quality_check`
2. `calibration`
3. `MI run 1`
4. `MI run 2`
5. `continuous block 1`
6. `MI run 3`
7. `continuous block 2`
8. `idle_block`
9. `idle_prepare`
10. 自动保存

固定通道约束：

- `channel_names = C3,Cz,C4,PO3,PO4,O1,Oz,O2`
- `channel_positions = 0,1,2,3,4,5,6,7`

关键默认参数：

- `trials_per_class=10`
- `run_count=3`
- `baseline/cue/imagery/iti = 2.0/2.0/4.0/2.0s`
- `run_rest_sec=60`
- `long_run_rest_every=2`
- `long_run_rest_sec=120`
- `practice_sec=0`
- `idle_block = 2 x 60s`
- `idle_prepare = 2 x 60s`
- `continuous = 2 x 240s`
- `continuous_command = 4-5s`
- `continuous_gap = 2-3s`
- `include_eyes_closed_rest_in_gate_neg=False`

### 2. 训练

训练模块默认读取：

- `datasets/custom_mi`

默认输出：

- `code/realtime/models/custom_mi_realtime.joblib`
- `code/realtime/models/custom_mi_realtime.json`
- `code/training/reports/custom_mi_training_summary.json`

训练当前只认任务分离文件：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

训练侧的基本约束：

- 主分类只使用 `*_mi_epochs.npz`
- gate 只使用 `*_gate_epochs.npz`
- artifact rejector 只使用 `*_artifact_epochs.npz`
- `*_continuous.npz` 只做 online-like 评估，不并入主分类训练
- gate 中来源于 `continuous_*` 的负样本会自动剔除
- 跨 run 必须保证 `channel_names`、`class_names`、`sampling_rate` 一致

默认采集与训练窗口是匹配的：

```text
imagery_sec >= max(window_secs) + max(window_offset_secs)
```

当前默认 `imagery_sec=4.0s`，满足训练默认窗口配置。

### 3. 实时推理

实时模块只负责加载训练工件做推理，不负责采集训练数据。

默认模型加载顺序：

1. `code/realtime/models/custom_mi_realtime.joblib`
2. 若不存在，则回退到 `code/realtime/models/subject_1_mi.joblib`

当前支持两种模式：

- `continuous`
- `guided`

实时决策顺序：

1. 信号质量规则
2. artifact rejector
3. control gate
4. 主 MI 分类
5. 平滑、迟滞和释放逻辑

### 4. 回看

viewer 已经是 run-bundle viewer，不再只看 `*_mi_epochs.npz`。

它可以直接接收：

- session 目录
- `*_session_meta.json`
- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`
- `*_events.csv`
- `*_trials.csv`
- `*_segments.csv`
- `*_board_data.npy`

viewer 会解析同一 `run_stem` 的整套文件，并联合展示：

- trial / event / segment 摘要
- continuous prompt 信息
- 保存后连续信号
- MI/gate/artifact/continuous 派生数据概览

## 数据存储口径

一次完整保存会同时生成四层数据。

### 1. 真源层

这层回答“采集时真实发生了什么”：

- `*_board_data.npy`
- `*_board_map.json`
- `*_raw.fif`
- `*_events.csv`

### 2. 语义层

这层把原子事件整理成协议语义：

- `*_trials.csv`
- `*_segments.csv`

### 3. 会话描述层

这层描述一次保存的统计、参数和文件索引：

- `*_session_meta.json`
- `session_meta_latest.json`
- `*_quality_report.json`
- `collection_manifest.csv`

### 4. 派生层

这层服务训练和评估，不替代真源层：

- `*_mi_epochs.npz` + `*.meta.json`
- `*_gate_epochs.npz` + `*.meta.json`
- `*_artifact_epochs.npz` + `*.meta.json`
- `*_continuous.npz` + `*.meta.json`

默认保存目录：

```text
datasets/custom_mi/sub-<subject>/ses-<session>/
```

同一次落盘共享一个 `run_stem`：

```text
sub-<subject>_ses-<session>_run-<NNN>_tpc-<TT>_n-<NNN>_ok-<NNN>
```

需要特别记住的边界：

- 预览滤波只用于显示，不写入落盘文件
- `*_board_data.npy` 是最接近板卡输出的保存物
- `*_raw.fif` 是 MNE 友好的工作视图，不是完整 board 真源
- `*_events.csv` 是原子事件真源
- `*_segments.csv` 是语义区间，不是 marker 真源
- 所有派生 `npz` 信号单位统一写成 `volt`
- `collection_manifest.csv` 与 `session_meta.json["files"]` 中的路径使用相对路径

## 文档索引

- [code/README.md](code/README.md)
- [code/collection/README.md](code/collection/README.md)
- [code/collection/README_SAVE_NAMING.md](code/collection/README_SAVE_NAMING.md)
- [code/training/README.md](code/training/README.md)
- [code/training/README_DATA_LOADING.md](code/training/README_DATA_LOADING.md)
- [code/training/README_DATA_ADMISSION.md](code/training/README_DATA_ADMISSION.md)
- [code/realtime/README.md](code/realtime/README.md)
- [code/viewer/README.md](code/viewer/README.md)
- [datasets/README.md](datasets/README.md)
- [datasets/custom_mi/README.md](datasets/custom_mi/README.md)

## 维护规则

如果后续继续修改以下任何一类内容，需要同步检查对应 README：

- 采集流程
- 保存字段
- 训练读取规则
- 实时运行约束
- viewer 支持范围

根 README 负责项目总览；模块级 README 负责细节。两者如果偏离，优先修正文档，而不是让使用者猜当前口径。
