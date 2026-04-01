# NPZ Viewer 说明

`viewer/` 只用于可视化采集后的 `npz`，不参与训练或实时推理。

## 1. 入口

推荐（项目根目录）：

```powershell
python run_04_view_collected_npz.py
```

等价入口：

```powershell
python code/viewer/run_npz_viewer_pycharm.py
```

指定文件启动：

```powershell
python code/viewer/mi_npz_viewer.py --npz <path-to-npz>
```

如果终端里没有 `python` 命令，请改用：

```powershell
& 'C:\Users\P1233\miniconda3\envs\MI\python.exe' code/viewer/mi_npz_viewer.py --help
```

## 2. 支持读取的文件

会扫描：

- `*_epochs.npz`
- `epochs.npz`

并排除：

- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`

说明：viewer 面向主 MI epoch 结构（`X/y/accepted/trial_ids`），不是 gate/artifact 专用查看器。

## 3. 主要功能

- 最近文件自动扫描与选择
- Trial / 类别 / 通道统计
- 单 trial 波形查看
- 类别平均波形
- 类别平均频谱（PSD）
- UI 中英文友好字体回退

## 4. 导出

支持导出：

- JSON：`<stem>_viewer_stats.json`
- CSV（类别统计）：`<stem>_class_stats.csv`
- CSV（通道统计）：`<stem>_channel_stats.csv`

## 5. 常见问题

### 5.1 选择文件后提示缺少 X/y

该文件不是主 epoch 结构（可能是 gate/artifact/continuous npz），请换 `*_mi_epochs.npz` 或 `*_epochs.npz`。

### 5.2 图像单位看起来不对

viewer 会按 `signal_unit` 自动换算到 `uV` 显示；如果历史文件单位字段异常，统计值可能偏差，请以新版采集导出文件为准。
