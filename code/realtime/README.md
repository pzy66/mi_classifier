# 实时分类程序说明

主入口：
- `run_realtime_pycharm.py`

核心实现：
- `mi_realtime_infer_only.py`

## 运行

```bash
conda activate MI
python code/realtime/run_realtime_pycharm.py
```

## 实时程序如何调用训练好的模型

程序启动时按顺序找模型：
1. `code/realtime/models/custom_mi_realtime.joblib`
2. 若不存在，回退到 `code/realtime/models/subject_1_mi.joblib`

加载后会读取模型内的：
- `pipeline`（用于推理）
- `class_names/display_class_names`（用于显示类别）
- `channel_names`（期望输入通道数）
- `sampling_rate` 与 `window_sec`（用于窗口长度与重采样）
- `preprocessing`（实时预处理参数）

## 连接设备步骤（已集成到界面）

1. 选择 `Board`
2. 在 `Serial Port` 下拉框选串口（可点 `Refresh Ports`）
3. 点击 `Connect Device`
4. 连接成功后点击 `Start Realtime`
5. 结束时先点 `Stop`，再点 `Disconnect`

## 实时推理流程

1. 从 BrainFlow 拉取当前滑动窗口数据
2. 根据模型通道数提取通道
3. 若实时采样率与模型采样率不同，自动重采样
4. 进行与训练对齐的预处理
5. 调用训练好的 `pipeline` 输出概率与类别
6. 做置信度阈值与历史平滑，实时更新界面

## 运行前校验

连接与启动前会检查：
- 非 Synthetic 板卡时串口不能为空
- `board_channel_positions`（如果配置）长度需和模型通道数一致
- `board_channel_positions` 不能有负数和重复索引
- 板卡可用 EEG 通道数必须满足模型所需通道数

## 可选配置

可在 `USER_CONFIG` 修改默认值：
- `serial_port`
- `board_id`
- `step_sec`
- `history_len`
- `confidence_threshold`
- `board_channel_positions`（可选）

