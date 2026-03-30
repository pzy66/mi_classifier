# 训练程序说明

主入口：
- `run_training_pycharm.py`

核心实现：
- `train_custom_dataset.py`

## 运行

```bash
conda activate MI
python code/training/run_training_pycharm.py
```

## 训练程序如何读取采集数据

训练数据根目录默认是：
- `datasets/custom_mi/`

读取逻辑：
1. 递归搜索所有 `epochs.npz`
2. 读取每个文件的 `X/y/accepted/class_names/channel_names/sampling_rate`
3. 只保留 `accepted==1` 的试次
4. 校验所有 session 的 `channel_names/class_names/sampling_rate` 一致
5. 若各 session 样本长度不同，统一裁剪到最短长度后合并
6. 最终合并为一个训练集

## 训练前预处理

训练前会执行（优化版经典流程）：
- 带通：`4-40 Hz`
- 陷波：`50 Hz`
- CAR（common average reference）
- 标准化（按 trial、按通道）

## 模型选择与评估

候选管线：
- `hybrid+lda`
- `hybrid+svm`

流程：
1. 按类别分层切分 `train/val/test`（约 60%/20%/20%）
2. 在验证集上选择最佳候选
3. 用 `train+val` 重训最佳模型
4. 在测试集输出 `test_acc` 与 `kappa`

附加鲁棒性：
- `--min-class-trials` 控制每类最少有效 trial（默认 5）
- 候选模型失败会记录错误并继续尝试其他候选
- 输出每个类别的测试准确率 `per_class_test_acc`

## 输出文件

默认输出：
- 模型：`code/realtime/models/custom_mi_realtime.joblib`
- 模型摘要：`code/realtime/models/custom_mi_realtime.json`
- 训练摘要：`code/training/reports/custom_mi_training_summary.json`

其中 `joblib` 里包含实时程序需要的关键字段：
- `pipeline`（训练好的 sklearn 管线）
- `class_names` / `display_class_names`
- `channel_names`
- `sampling_rate`
- `window_sec`
- `preprocessing`
- `metrics`

## 常用命令

训练全部被试（默认）：
```bash
python code/training/train_custom_dataset.py
```

只训练一个被试：
```bash
python code/training/train_custom_dataset.py --subject 001
```

自定义最小 trial 数：
```bash
python code/training/train_custom_dataset.py --min-class-trials 8
```

