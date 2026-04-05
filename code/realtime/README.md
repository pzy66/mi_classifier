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

可选（仅看波形，不做分类）：

```powershell
python run_05_channel_monitor.py
```

运行配置主要来自 `code/realtime/mi_realtime_infer_only.py` 的 `USER_CONFIG`。

## 2. 模型加载顺序

默认优先：

1. `code/realtime/models/custom_mi_realtime.joblib`
2. 若不存在，回退 `code/realtime/models/subject_1_mi.joblib`

发生回退时，界面会弹 warning。

兼容性说明：

- 运行时会对旧版 realtime artifact 做内存内兼容补齐，自动补回缺失的 `epoch_window` / `window_offset_sec` / `window_offset_secs_used`
- 如果工件缺少更关键的字段，仍会拒绝启动并提示重新导出模型
- Windows 下串口下拉会额外读取系统 Ports 设备清单，尽量避开 `Problem/Error` 的坏端口条目

## 3. 实时模式

`USER_CONFIG['realtime_mode']` 支持：

- `continuous`（默认）：每个滑窗都输出，低证据显示 `UNCERTAIN`
- `guided`：按协议 `baseline -> cue -> imagery -> iti`

guided 默认协议参数：

- `protocol_baseline_sec=2.0`
- `protocol_cue_sec=2.0`
- `protocol_imagery_sec=4.0`
- `protocol_iti_sec=2.0`
- `protocol_trials_per_class=4`
- `protocol_random_seed=42`

## 4. 决策顺序

每个滑窗按顺序执行：

1. 信号质量规则（flatline/异常通道）
2. artifact rejector（若模型中可用）
3. control gate（若模型中可用）
4. 主 MI 分类
5. 平滑、迟滞、释放逻辑

常见显示状态：

- `WARMING UP`
- `BAD WINDOW/ARTIFACT`
- `NO CONTROL`
- `LEFT HAND / RIGHT HAND / FEET / TONGUE`
- `UNCERTAIN`

## 5. 关键配置

常用项：

- `model_path`
- `window_model_paths`
- `realtime_mode`
- `step_sec`, `history_len`
- `confidence_threshold`, `margin_threshold`
- `gate_confidence_threshold`, `gate_margin_threshold`
- `use_artifact_recommended_thresholds`
- `use_artifact_recommended_gate_thresholds`
- `board_id`, `serial_port`
- `board_channel_positions`

## 6. 板卡约束

非 `SYNTHETIC_BOARD` 时：

1. `serial_port` 不能为空
2. 如果板卡 EEG 行数和模型通道数完全一致，程序会自动把 `board_channel_positions` 解析成顺序映射
3. 如果两者不一致，必须显式设置 `board_channel_positions`
4. `board_channel_positions` 长度必须等于模型通道数
5. 位置索引不能重复，不能为负，不能超过板卡 EEG 行数

## 7. 常见问题

### 7.1 启动时报 `board_channel_positions` 错误

如果是标准 Cyton 8 通道且模型也是 8 通道，现在可以留空让程序自动映射。

如果板卡通道数和模型通道数不一致，再按训练通道顺序配置显式索引，例如：

```python
"board_channel_positions": [0, 1, 2, 3, 4, 5, 6, 7]
```

### 7.2 输出长期 `UNCERTAIN`

优先检查：

1. 训练模型是否启用了推荐阈值
2. 实时通道顺序是否与训练一致
3. 是否频繁触发坏窗冻结

### 7.3 总是 `NO CONTROL`

通常是 gate 未通过，先看训练摘要中 gate 是否启用、阈值是否过严。
