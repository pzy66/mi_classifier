# MI Classifier（四类 MI + Gate + Bad-Window + Continuous）

本项目只做运动想象（MI），不包含 SSVEP。

目标不是只做离线分类分数，而是完成可部署链路：

1. 采集四类 MI 主任务数据（left/right/feet/tongue）
2. 采集 no-control（idle/prepare 等）数据用于 gate
3. 采集伪迹/坏窗数据用于 rejector
4. 采集 continuous 连续数据用于仿实时评估
5. 训练主分类 + gate + artifact rejector
6. 实时按顺序执行：`bad-window -> gate -> main classifier`

## 1. 快速开始（推荐）

```powershell
conda activate MI
pip install -r requirements.txt
pip install -r requirements-realtime.txt
```

可选（训练深度候选）：

```powershell
pip install -r requirements-deep.txt
```

三步跑通：

```powershell
python run_01_collection_only.py
python run_02_training.py
python run_03_realtime_infer.py
```

查看采集 npz：

```powershell
python run_04_view_collected_npz.py
```

8 通道在线波形监视（可选）：

```powershell
python run_05_channel_monitor.py
```

## 2. 目录与入口

```text
mi_classifier/
|-- code/
|   |-- collection/    # 采集 UI
|   |-- training/      # 训练脚本
|   |-- realtime/      # 实时推理 UI
|   |-- viewer/        # npz 可视化
|   |-- shared/        # 公共逻辑
|   `-- legacy/        # 历史代码（不参与当前主流程）
|-- datasets/
|   `-- custom_mi/     # 自采数据根目录
|-- run_01_collection_only.py
|-- run_02_training.py
|-- run_03_realtime_infer.py
|-- run_04_view_collected_npz.py
|-- run_05_channel_monitor.py
`-- README.md
```

## 3. 采集规范（必须对齐训练）

### 3.1 固定通道约束

采集端强校验固定 8 通道、固定顺序：

- `C3, Cz, C4, PO3, PO4, O1, Oz, O2`
- `channel_positions = 0,1,2,3,4,5,6,7`

不满足会阻止开始采集。

### 3.2 默认流程顺序

每次会话按以下阶段推进：

1. `quality_check`
2. `calibration`（open/closed/eye/blink/swallow/jaw/head）
3. `practice`
4. `MI runs`
5. `idle_block`
6. `idle_prepare`
7. `continuous`
8. 保存导出

### 3.3 默认关键参数

- 主 trial：`baseline=2.0s, cue=1.0s, imagery=4.0s, iti=2.5s`
- run：`trials_per_class=10, run_count=4`
- run 休息：`90s`，每 2 个 run 长休 `180s`
- quality_check：`45s`
- calibration：`120/60/60/30/30/30/30s`
- practice：`180s`
- idle：`2 x 90s`
- prepare：`1 x 90s`
- continuous：`2 x 150s`，命令 `3-6s`，间隔 `1-3s`

### 3.4 单/双窗口说明

当前默认 `use_separate_participant_screen = True`（双窗口，受试者全屏提示）。

如果你想单窗口运行，可在采集 UI 中关闭该选项。

### 3.5 快捷键

- `Space`：暂停/继续
- `B`：
  - trial 阶段：标记坏试次
  - continuous 阶段：标记当前命令执行失败（`execution_success=0`）
- `Esc`：停止并保存

### 3.6 保存结构与文件

每次保存到：

```text
datasets/custom_mi/sub-<subject>/ses-<session>/
```

每次 run 自动生成带编号文件名前缀（run stem），例如：

```text
sub-001_ses-20260331_203000_run-001_tpc-10_n-160_ok-154
```

核心输出文件：

- `*_raw.fif`
- `*_events.csv`
- `*_trials.csv`
- `*_session_meta.json`
- `*_quality_report.json`
- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`
- `*_epochs.npz`（legacy，可选）

数据根目录会维护：

- `collection_manifest.csv`

旧表头会自动迁移并备份为 `*_legacy_schema_*.csv`。

## 4. 训练说明

训练入口：`python run_02_training.py`（实际调用 `code/training/train_custom_dataset.py`）。

### 4.1 训练读取数据

默认扫描：`datasets/custom_mi`

优先读取新格式：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

兼容 legacy：

- `*_epochs.npz` / `epochs.npz`

### 4.2 三条模型线

- 主分类：4 类 MI
- gate：`control` vs `no-control`
- artifact rejector：`clean` vs `artifact`

continuous 数据只用于 online-like 评估，不直接并入主分类训练。

### 4.3 关键默认训练参数

- 预处理：`4-40Hz` 带通 + `50Hz` 陷波 + `CAR`
- 窗口：`window_secs=2.0,2.5,3.0`
- offset：`window_offset_secs=0.25,0.5,0.75`

重要约束：

- 需满足 `imagery_sec >= max(window_secs) + max(window_offset_secs)`
- 默认即至少 `3.75s`，建议采集用 `4.0s` 或以上

### 4.4 默认输出

- 模型：`code/realtime/models/custom_mi_realtime.joblib`
- 训练摘要：`code/training/reports/custom_mi_training_summary.json`
- 旁路摘要：`code/realtime/models/custom_mi_realtime.json`

## 5. 实时说明

实时入口：`python run_03_realtime_infer.py`。

每个滑窗决策顺序：

1. 质量规则（flatline/异常通道）
2. artifact rejector（如可用）
3. control gate（如可用）
4. 主分类
5. 平滑与迟滞

常见状态：

- `WARMING UP`
- `BAD WINDOW/ARTIFACT`
- `NO CONTROL`
- `LEFT HAND / RIGHT HAND / FEET / TONGUE`
- `UNCERTAIN`

非 synthetic 板卡必须设置 `board_channel_positions`，且顺序必须与训练通道一致。

## 6. 推荐 SOP

1. 采集：每位被试至少 3 个不同日期 session，完整跑完所有阶段。
2. 训练：先用默认参数训练并检查 `control_gate` / `artifact_rejector` 是否启用。
3. 实时：优先使用工件推荐阈值（默认开启）。
4. 回看：用 viewer 检查类别平衡、坏窗比例、continuous 标注质量。

## 7. 常见问题

### 7.1 训练报窗口超界（imagery 长度不足）

含义：训练窗口和 offset 超过采集到的 imagery 有效长度。

处理：

1. 重新采集并增大 `imagery_sec`（建议 `>=4.0s`）
2. 或训练时下调 `--window-secs` / `--window-offset-secs`

### 7.2 实时报 `board_channel_positions` 错误

在 `code/realtime/mi_realtime_infer_only.py` 的 `USER_CONFIG` 中明确设置，长度和顺序需与模型通道一致。

### 7.3 gate/artifact 未启用

通常是对应数据不足或不平衡：

- 检查是否有 `*_gate_epochs.npz` / `*_artifact_epochs.npz`
- 检查负类样本是否足够
- 检查切分后是否出现单类集合

## 8. 子文档索引

- `code/README.md`：代码目录总览
- `code/collection/README.md`：采集模块
- `code/collection/README_SAVE_NAMING.md`：命名与 manifest
- `code/training/README.md`：训练模块
- `code/training/README_DATA_LOADING.md`：训练装载规则
- `code/realtime/README.md`：实时模块
- `code/viewer/README.md`：可视化模块
- `datasets/README.md`：数据目录说明
- `datasets/custom_mi/README.md`：自采数据目录规范

如果你修改了采集或训练参数，请同步更新对应 README，避免“代码与文档不一致”。
