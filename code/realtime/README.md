# 实时推理模块说明

本模块只负责加载已训练工件做实时推理，不负责采集训练数据。

## 1. 入口

推荐（项目根目录）：

```powershell
python run_03_realtime_infer.py
```

等价入口：

```powershell
python code/realtime/run_realtime_pycharm.py
```

可选：8 通道实时波形监视（不做分类）：

```powershell
python run_05_channel_monitor.py
```

## 2. 模型加载顺序

默认优先加载：

1. `code/realtime/models/custom_mi_realtime.joblib`
2. 若不存在，回退到 `code/realtime/models/subject_1_mi.joblib`

如果发生回退，界面会给出 warning，建议先完成自定义训练再进行实时评估。

## 3. 两种实时模式

在 `USER_CONFIG['realtime_mode']` 选择：

- `continuous`：每次滑窗都输出，低证据时显示 `UNCERTAIN`
- `guided`：按 `baseline -> cue -> imagery -> iti` 协议运行

## 4. 决策顺序

每个滑窗按顺序执行：

1. 信号质量规则（flatline/异常通道）
2. artifact rejector（若模型中可用）
3. control gate（若模型中可用）
4. 主 MI 分类
5. 平滑、迟滞、释放逻辑

## 5. 关键配置（`mi_realtime_infer_only.py`）

常用项：

- `model_path`
- `realtime_mode`
- `step_sec`, `history_len`
- `confidence_threshold`, `margin_threshold`
- `gate_confidence_threshold`, `gate_margin_threshold`
- `use_artifact_recommended_thresholds`
- `use_artifact_recommended_gate_thresholds`
- `board_id`, `serial_port`
- `board_channel_positions`

### 重要约束

非 `SYNTHETIC_BOARD` 时：

1. `serial_port` 不能为空
2. 必须显式设置 `board_channel_positions`
3. `board_channel_positions` 长度必须等于模型通道数
4. 位置索引不能重复、不能为负

## 6. 常见状态显示

- `WARMING UP`
- `BAD WINDOW/ARTIFACT`
- `NO CONTROL`
- `LEFT HAND / RIGHT HAND / FEET / TONGUE`
- `UNCERTAIN`

## 7. 常见问题

### 7.1 启动时报 board_channel_positions 错误

按模型通道顺序设置显式索引，例如 8 通道可设为：

```python
"board_channel_positions": [0,1,2,3,4,5,6,7]
```

### 7.2 输出总是 UNCERTAIN

先检查：

1. 是否启用了推荐阈值
2. 训练数据和实时通道顺序是否一致
3. 实时信号质量是否频繁触发坏窗冻结

### 7.3 一直显示 NO CONTROL

通常是 gate 判定未通过。先看训练摘要里 gate 是否启用，以及推荐 gate 阈值是否过严。
