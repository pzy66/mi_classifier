# MI Classifier（四类 MI + Gate + Bad-Window + Continuous）

本项目只做 `MI`（不含 SSVEP）。

目标不是只在离线 trial 上拿高分，而是把完整链路做对：

1. 采集规范的四类 MI 主数据
2. 单独采集 no-control/idle 数据
3. 单独采集伪迹/坏窗数据
4. 采集 continuous 数据做仿实时验证
5. 训练三类模型（主分类、gate、bad-window rejector）
6. 实时按顺序执行：`bad-window -> gate -> 主分类`

---

## 1. 全链路总览

```text
采集（单 session）
  -> 保存（按被试/会话/运行编号）
  -> 数据分流（MI / gate / artifact / continuous）
  -> 训练（主模型 + gate + rejector + 阈值）
  -> continuous 仿实时评估
  -> 实时部署（拒判优先，分类在后）
```

四条数据线的用途：

- 主 4 类 MI：训练 `left_hand/right_hand/feet/tongue` 主分类器
- gate：训练 `control vs no-control`
- bad-window：训练 `clean vs artifact`
- continuous：评估误触发、延迟、稳定性（默认不参与主分类训练）

---

## 2. 入口脚本

根目录推荐入口：

- `run_01_collection_only.py`：采集 UI
- `run_02_training.py`：训练与导出工件
- `run_03_realtime_infer.py`：实时推理 UI
- `run_04_view_collected_npz.py`：查看采集缓存
- `run_05_channel_monitor.py`：8 通道波形监视（可选）

也可以直接调用对应子脚本：

- `code/collection/mi_data_collector.py`
- `code/training/train_custom_dataset.py`
- `code/realtime/mi_realtime_infer_only.py`

---

## 3. 环境准备

建议使用 `MI` conda 环境。

```powershell
conda activate MI
pip install -r requirements.txt
pip install -r requirements-realtime.txt
```

如果要训练深度候选（FBLight/dual-branch 等），额外安装：

```powershell
pip install -r requirements-deep.txt
```

如果你的终端没有 `python` 命令，直接使用解释器绝对路径，例如：

```powershell
& 'C:\Users\P1233\miniconda3\envs\MI\python.exe' run_01_collection_only.py
```

---

## 4. 采集流程（当前实现）

### 4.1 固定通道约束（强校验）

采集端固定 8 通道，且顺序必须一致：

- `C3, Cz, C4, PO3, PO4, O1, Oz, O2`

在 UI 里如果不是这个名称顺序，或位置不是 `0,1,2,3,4,5,6,7`，程序会直接报错阻止开始采集。

### 4.2 单窗口 UI（默认）

当前默认是**单窗口运行**，不自动弹出受试者全屏窗口。

- `use_separate_participant_screen = False`
- 所有关键提示都在主界面展示
- 支持窗口缩放与自适应重排（2/3/4 列）

### 4.3 采集阶段顺序

每次会话按这个顺序推进：

1. `quality_check`（质量检查）
2. `calibration`（静息/伪迹校准）
3. `practice`（想象方式练习）
4. `MI runs`（主任务，多 run）
5. `idle_block`（无控制）
6. `idle_prepare`（准备但不执行）
7. `continuous`（仿实时连续块）
8. 保存并导出多份文件

### 4.4 关键默认参数（采集 UI 默认值）

#### 主 trial

- `baseline_sec = 2.0`
- `cue_sec = 1.0`
- `imagery_sec = 4.0`
- `iti_sec = 2.5`

即：`2.0 + 1.0 + 4.0 + 2.5 = 9.5s / trial`

#### run 结构

- `trials_per_class = 10`
- `run_count = 4`
- 每 run 共 `40 trial`，4 类均衡
- `max_consecutive_same_class = 2`
- run 间休息 `90s`
- 每 `2` 个 run 后长休息 `180s`

#### 校准与附加块

- 质量检查：`45s`
- 睁眼静息：`120s`
- 闭眼静息：`60s`
- 眼动：`60s`
- 眨眼：`30s`
- 吞咽：`30s`
- 咬牙：`30s`
- 头动：`30s`
- MI 练习：`180s`
- idle 段：`2 x 90s`
- prepare-no-exec：`1 x 90s`（可改段数）
- continuous：`2 x 150s`
  - 命令时长：`3-6s`
  - 间隔：`1-3s`

### 4.5 会话元数据（已更新）

会话保存时记录：

- 被试编号/名称、会话编号、板卡、串口、采样率、参考设置
- 被试状态、咖啡/茶、近期运动、睡眠备注、备注
- 自动随机种子（每次开始采集自动生成并保存）

不再保存 `operator` 字段。

### 4.6 快捷键与标记

采集中快捷键：

- `Space`：暂停/继续
- `B`：
  - 在主 trial 阶段：标记坏试次
  - 在 continuous 阶段：标记“当前命令执行失败”
- `Esc`：停止并保存

continuous 命令失败会写入 `execution_success=0`（用于后续分析）。

---

## 5. 保存结构与文件含义

### 5.1 保存路径（按被试分目录）

每次保存路径：

```text
datasets/custom_mi/
  sub-<被试编号或姓名>/
    ses-<会话编号>/
      sub-..._ses-..._run-001_tpc-10_n-160_ok-154_*.xxx
```

说明：

- `sub-...` 支持中文名（会自动做文件名安全处理）
- 同一会话下多次保存会自动递增 `run-001/run-002/...`

### 5.2 每次会话保存导出文件

会导出以下核心文件：

- `*_raw.fif`：连续原始 EEG（含注释）
- `*_events.csv`：事件日志（含 marker、sample、block/prompt 等）
- `*_trials.csv`：trial 级记录（accepted/rejected）
- `*_session_meta.json`：会话元数据与统计
- `*_quality_report.json`：信号质量摘要
- `*_mi_epochs.npz`：主 4 类训练数据
- `*_gate_epochs.npz`：gate 训练数据
- `*_artifact_epochs.npz`：bad-window 训练数据
- `*_continuous.npz`：continuous 仿实时数据
- `*_epochs.npz`：legacy 兼容包（可在 UI 勾选）

此外在数据根目录追加：

- `collection_manifest.csv`：全局采集清单

### 5.3 四类数据如何落盘

#### 1) 主分类数据（`*_mi_epochs.npz`）

- `X_mi, y_mi, mi_trial_ids`
- 只包含 accepted imagery 窗口

#### 2) gate 数据（`*_gate_epochs.npz`）

- 正类：`X_gate_pos`（MI 窗口）
- 负类：`X_gate_neg`（baseline/iti/idle/prepare/continuous no_control 等）
- 硬负类：`X_gate_hard_neg`（伪迹块、rejected trial 等）
- 来源标签：`gate_neg_sources, gate_hard_neg_sources`

闭眼静息默认**不加入** gate 负类，勾选“闭眼静息加入门控负类”后才会加入。

#### 3) bad-window 数据（`*_artifact_epochs.npz`）

- `X_artifact`：伪迹/坏窗样本
- `artifact_labels`：伪迹标签来源

训练时会和 clean 负类组合成二分类（good vs artifact）。

#### 4) continuous 数据（`*_continuous.npz`）

- `X_continuous`：连续块信号
- `continuous_event_labels/start/end_samples`
- `continuous_events`（JSON 字符串数组，含 prompt、class、execution_success 等）

---

## 6. 训练流程（`run_02_training.py`）

训练入口：`code/training/train_custom_dataset.py`

### 6.1 输入文件扫描

训练会递归扫描 `--dataset-root`（默认 `datasets/custom_mi`），优先使用新格式：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

兼容 legacy：

- `*_epochs.npz` / `epochs.npz`

### 6.2 数据分流与任务

- 主模型（4 类）：用 `mi` 数据
- gate（二分类）：`X_gate_pos` vs `X_gate_neg + X_gate_hard_neg`
- bad-window（二分类）：`X_artifact` vs clean negatives
- continuous：仅做仿实时评估

### 6.3 切分策略（防泄漏）

优先使用分组切分（按 run/session），策略顺序：

1. `session_holdout`
2. `group_shuffle`
3. `trial_stratified_fallback`（最后兜底）

当 run 太少回退到 trial 级切分时，CLI 会打印 warning，需谨慎解读离线分数。

### 6.4 预处理（训练/实时一致）

- 带通：`4-40 Hz`
- 陷波：`50 Hz`
- `CAR`
- 不做窗级标准化（`standardize_data=False`）

### 6.5 默认窗口与 offset

- `window_secs = 2.0,2.5,3.0`
- `window_offset_secs = 0.25,0.5,0.75`

用于多窗口融合与 offset 搜索。

### 6.6 默认候选模型

#### 主分类默认（保守）

- `central_fbcsp_lda`

#### gate 默认

- 有 torch：`central_gate_fblight`, `central_prior_gate_fblight`
- 无 torch：回退 classical 候选

#### bad-window 默认

- 有 torch：`full8_fblight`
- 无 torch：回退 classical 候选

可通过 CLI 覆盖，例如：

```powershell
python code/training/train_custom_dataset.py `
  --candidate-names central_fbcsp_lda,central_prior_dual_branch_fblight_tcn `
  --gate-candidate-names central_gate_fblight,central_prior_gate_fblight `
  --artifact-candidate-names full8_fblight
```

### 6.7 阈值与输出字段

训练会为主模型、gate、artifact rejector分别给出推荐阈值：

- `confidence_threshold`
- `margin_threshold`
- `recommended_threshold`（同值结构化复制）

并写明输出字段语义，例如：

- gate：`probability/margin/confidence`
- artifact：`probability/margin/confidence`

### 6.8 continuous 仿实时评估

训练阶段会对 continuous 数据执行 online-like 评估，输出：

- `evaluated_prompt_count`
- `mi_prompt_accuracy`
- `no_control_false_activation_rate`
- `decision_order = [bad_window_rejector, control_gate, main_mi_classifier]`

### 6.9 导出工件

默认输出：

- 模型工件：`code/realtime/models/custom_mi_realtime.joblib`
- 报告：`code/training/reports/custom_mi_training_summary.json`
- 旁路 JSON：`custom_mi_realtime.json`

CLI 会打印：

- 主模型指标（acc/macro_f1/kappa）
- gate/artifact 是否启用
- 推荐阈值
- continuous 指标摘要

---

## 7. 实时推理流程（`run_03_realtime_infer.py`）

实时入口：`code/realtime/mi_realtime_infer_only.py`

### 7.1 实时决策顺序

每个滑窗按顺序执行：

1. 坏窗与质量判断（flatline/异常通道等）
2. `artifact rejector`（若训练可用）
3. `control gate`（若训练可用）
4. 主 4 类分类
5. 平滑 + 迟滞 + hold/release

即：先拒判，再判 control，再做四分类。

### 7.2 输出状态（常见）

- `WARMING UP`：窗口长度尚不足
- `NO CONTROL`：gate 拒绝
- `BAD WINDOW/ARTIFACT`：质量或 rejector 拒绝
- 稳定类别：`LEFT HAND/RIGHT HAND/FEET/TONGUE`
- `UNCERTAIN`：证据不足（尤其在无 gate 工件时）

### 7.3 关键配置（`USER_CONFIG`）

重点项：

- `realtime_mode`: `continuous` / `guided`
- `model_path`
- `use_artifact_recommended_thresholds`
- `use_artifact_recommended_gate_thresholds`
- `step_sec`, `history_len`
- `confidence_threshold`, `margin_threshold`
- `gate_confidence_threshold`, `gate_margin_threshold`
- `release_windows`, `gate_release_windows`
- `artifact_freeze_windows`
- `board_channel_positions`

重要：真实板卡模式下必须正确设置 `board_channel_positions`，并与训练通道顺序一致。

---

## 8. 推荐执行 SOP（建议）

### 第一步：采集

```powershell
python run_01_collection_only.py
```

建议：

- 至少 3 个不同日期 session
- 每次完整执行全部阶段（含 idle/artifact/continuous）
- 采完立刻检查 `*_mi/_gate/_artifact/_continuous` 是否齐全

### 第二步：训练

```powershell
python run_02_training.py
```

或只训练指定被试：

```powershell
python code/training/train_custom_dataset.py --subject 001
```

### 第三步：实时

```powershell
python run_03_realtime_infer.py
```

优先使用工件内推荐阈值（默认已开启）。

---

## 9. 采集后质控清单（每个 session）

至少检查以下项目：

1. 文件完整性
   - `*_mi_epochs.npz`
   - `*_gate_epochs.npz`
   - `*_artifact_epochs.npz`
   - `*_continuous.npz`
   - `*_session_meta.json`
2. 类别平衡
   - 主任务 4 类是否均衡
   - accepted/rejected 比例是否异常
3. gate 负类来源
   - `gate_neg_sources` 是否覆盖 baseline/iti/idle/prepare/continuous
4. bad-window 样本
   - `artifact_labels` 是否包含 blink/eye/swallow/jaw/head 等
5. continuous 标注
   - prompt 起止时间、类别、execution_success 是否合理

---

## 10. 常见问题排查

### 10.1 UI 字显示不全

- 已实现自适应 2/3/4 列布局
- 先尝试放大窗口（建议 >= 1500px 宽）
- 系统缩放过高时可适当降低显示缩放

### 10.2 实时报 `board_channel_positions` 错误

- 在 `code/realtime/mi_realtime_infer_only.py` 的 `USER_CONFIG` 中设置正确索引
- 顺序必须和训练时一致（默认 8 通道顺序）

### 10.3 训练显示 gate/artifact 不可用

通常是数据不足或文件缺失：

- 检查是否存在 `*_gate_epochs.npz` / `*_artifact_epochs.npz`
- 检查负类是否太少
- 检查 split 后某个集合是否缺类

### 10.4 continuous 指标不可用

- 检查 `*_continuous.npz` 是否存在
- 检查 prompt 标注数量是否为 0

---

## 11. 目录说明

```text
mi_classifier/
|-- code/
|   |-- collection/      # 采集 UI
|   |-- realtime/        # 实时推理 UI
|   |-- shared/          # 采集/训练/实时共用逻辑
|   |-- training/        # 训练脚本、报告
|   `-- viewer/          # npz 查看工具
|-- datasets/
|   `-- custom_mi/       # 自采数据根目录
|-- runtime/             # 运行期缓存（如 mne home）
|-- run_01_collection_only.py
|-- run_02_training.py
|-- run_03_realtime_infer.py
|-- run_04_view_collected_npz.py
|-- run_05_channel_monitor.py
`-- README.md
```

---

## 12. 关键代码索引

采集：

- `code/collection/mi_data_collector.py`
- `code/shared/src/mi_collection.py`

训练：

- `code/training/train_custom_dataset.py`
- `code/shared/src/models.py`

实时：

- `code/realtime/mi_realtime_infer_only.py`
- `code/shared/src/realtime_mi.py`

查看：

- `code/viewer/mi_npz_viewer.py`

---

如果你后续改了采集参数（例如 trial 时长、run 数、continuous 时长），请同步修改本 README 对应章节，避免“代码与文档不一致”。
