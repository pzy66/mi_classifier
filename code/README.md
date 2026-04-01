# code 目录说明

`code/` 按职责拆分，避免采集、训练、实时逻辑互相耦合。

## 目录职责

- `collection/`：采集 UI 与会话保存（只采集，不做在线分类）
- `training/`：读取 `datasets/custom_mi` 并训练模型工件
- `realtime/`：加载已训练工件做实时推理
- `viewer/`：查看 `*_mi_epochs.npz` / `*_epochs.npz`
- `shared/`：采集、训练、实时共用底层模块
- `legacy/`：历史代码归档，不参与当前主流程

## 推荐入口

优先使用项目根目录启动脚本：

- `run_01_collection_only.py`
- `run_02_training.py`
- `run_03_realtime_infer.py`
- `run_04_view_collected_npz.py`
- `run_05_channel_monitor.py`

这些脚本会自动处理路径，适合直接运行。

如果终端没有 `python` 命令，统一改用：

```powershell
& 'C:\Users\P1233\miniconda3\envs\MI\python.exe' <script.py>
```

说明：`run_*_pycharm.py` 启动器偏 GUI 友好，异常时可能弹窗；在终端排障时优先直接运行对应主脚本（如 `train_custom_dataset.py`）。

## 开发约束

- 新增采集相关逻辑：放在 `collection/` + `shared/`
- 新增训练相关逻辑：放在 `training/` + `shared/`
- 新增实时相关逻辑：放在 `realtime/` + `shared/`
- 不要把 UI 代码、模型训练代码和在线推理代码写在同一文件
