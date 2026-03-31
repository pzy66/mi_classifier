# MI 采集模块说明

本模块只负责采集、标记、保存，不负责实时分类。

## 1. 入口

推荐（项目根目录）：

```powershell
python run_01_collection_only.py
```

等价入口（目录内）：

```powershell
python code/collection/run_collection_pycharm.py
```

直接运行采集主程序（可带预填参数）：

```powershell
python code/collection/mi_data_collector.py --subject-id 001 --session-id 20260331_203000
```

支持参数：

- `--serial-port COM3`
- `--output-root <path>`
- `--subject-id 001`
- `--session-id 20260331_203000`
- `--synthetic`

## 2. 核心文件

- `code/collection/mi_data_collector.py`：采集 UI 主程序
- `code/shared/src/mi_collection.py`：会话保存、事件映射、npz/fif 导出

## 3. 采集前检查

1. 使用 `MI` 环境启动（已安装 requirements）。
2. 板卡与串口正确（非 synthetic 必须填写串口）。
3. 通道配置固定为：
   - 名称：`C3,Cz,C4,PO3,PO4,O1,Oz,O2`
   - 位置：`0,1,2,3,4,5,6,7`
4. 输出根目录可写（默认 `datasets/custom_mi`）。

## 4. 会话流程

每次会话阶段顺序：

1. 连接设备后，操作员先在当前界面观察原始波形（raw EEG、无滤波、无预处理）
2. `calibration`
3. `practice`
4. 前半段 `MI runs`
5. 中途插入短 `continuous`
6. 后半段 `MI runs`
7. 末尾再插入短 `continuous`
8. `idle_block`
9. `idle_prepare`
10. 自动保存

默认关键参数：

- `trials_per_class=10`
- `run_count=4`
- `baseline/cue/imagery/iti = 2.0/1.0/4.0/2.5s`
- `max_consecutive_same_class=2`
- run 休息 `90s`，每 2 个 run 长休 `180s`
- `quality_check_sec=45s` 仅作开始前人工观察参考，不写入正式事件流
- `continuous = 2 x 150s`，默认插在完成 MI run 2 和 run 4 后

continuous 参数约束：

- `continuous_block_count <= run_count`
- `continuous_command_max_sec >= continuous_command_min_sec`
- `continuous_gap_max_sec >= continuous_gap_min_sec`
- `continuous_block_sec >= continuous_command_min_sec`

## 5. 单窗口 / 双窗口

默认 `use_separate_participant_screen=True`（双窗口）。

- 操作员窗口：参数配置、设备状态、控制按钮
- 受试者窗口：全屏提示词与倒计时

实际行为：

- 连接成功后，先留在操作员窗口观察 raw 波形和调参数
- 点击 `开始采集` 后，如启用受试者全屏，当前屏幕切到受试者提示界面
- 如果关闭该选项，则整场采集都在操作员窗口显示

## 6. 采集中操作

操作员按钮：

- `连接设备`
- `开始采集`
- `暂停/继续`
- `标记坏试次`
- `停止并保存`
- `断开设备`

快捷键：

- `Space`：暂停/继续
- `B`：
  - trial 阶段标记坏试次
  - continuous 阶段标记当前命令失败（`execution_success=0`）
- `Esc`：停止并保存

提示音与暂停行为：

- 每个 `imagery` 开始时播放开始提示音
- 每个 `imagery` 结束时播放结束提示音
- 在 `continuous` 中暂停后恢复，命令时间轴会继续沿暂停前的有效时间推进，不把暂停空档算进 prompt 调度

## 7. 保存内容

每次会话会写入：

- `*_raw.fif`
- `*_events.csv`
- `*_trials.csv`
- `*_session_meta.json`
- `*_quality_report.json`
- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`
- `*_epochs.npz`（勾选“同时导出”时）

并更新：

- `datasets/custom_mi/collection_manifest.csv`

文件命名规则见：`README_SAVE_NAMING.md`。

## 8. 常见问题

### 8.1 无法开始采集

检查：串口是否为空、设备是否连接、通道名/通道位置是否符合固定约束。

### 8.2 采集后训练找不到数据

确认当前会话目录里存在 `*_mi_epochs.npz`（以及可选 gate/artifact/continuous 文件）。

### 8.3 标记写入失败导致会话中止

程序会主动中止并不保存本次会话，需先排查设备连接与 marker 写入能力。
