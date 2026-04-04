# MI Classifier

本项目只做运动想象 EEG（MI），不包含 SSVEP。当前主链路是：

```text
采集 -> 保存真源/语义/派生数据 -> 训练 -> 实时推理 -> 回看
```

当前数据存储口径是单轨 `schema_version=2`。仓库现在不再为旧 `*_epochs.npz` 或旧 manifest 自动兼容；新的采集数据统一落到任务分离文件和相对路径 manifest 上。

## 1. 快速开始

```powershell
conda activate MI
pip install -r requirements.txt
pip install -r requirements-realtime.txt
```

如果需要深度候选模型：

```powershell
pip install -r requirements-deep.txt
```

如果终端没有 `python` 命令，可以直接用解释器绝对路径：

```powershell
& 'C:\Users\P1233\miniconda3\envs\MI\python.exe' run_02_training.py
```

## 2. 主流程入口

```powershell
python run_01_collection_only.py
python run_02_training.py
python run_03_realtime_infer.py
```

辅助入口：

```powershell
python run_04_view_collected_npz.py
python run_05_channel_monitor.py
```

## 3. 当前到底采什么数据

一次完整采集会同时得到四层数据：

1. 真源层  
   这层保留设备输出和原子事件，供以后审计、复切片、定位采样问题。
   - `*_board_data.npy`
   - `*_board_map.json`
   - `*_raw.fif`
   - `*_events.csv`

2. 语义层  
   这层把原子事件整理成可直接理解的 trial 和 interval。
   - `*_trials.csv`
   - `*_segments.csv`

3. 会话描述层  
   这层保存会话参数、统计、路径索引和质量信息。
   - `*_session_meta.json`
   - `session_meta_latest.json`
   - `*_quality_report.json`
   - `collection_manifest.csv`

4. 训练/评估派生层  
   这层是任务化缓存，只服务训练和回看，不替代真源层。
   - `*_mi_epochs.npz` + `*_mi_epochs.meta.json`
   - `*_gate_epochs.npz` + `*_gate_epochs.meta.json`
   - `*_artifact_epochs.npz` + `*_artifact_epochs.meta.json`
   - `*_continuous.npz` + `*_continuous.meta.json`

## 4. 默认采集流程

固定 8 通道约束：

- `channel_names = C3,Cz,C4,PO3,PO4,O1,Oz,O2`
- `channel_positions = 0,1,2,3,4,5,6,7`

默认会话顺序：

1. 连接设备后，操作员在预览面板观察信号。
2. `quality_check` 仅作为人工检查建议阶段。
3. `calibration`。
4. `MI run 1`。
5. `MI run 2`。
6. `continuous block 1`。
7. `MI run 3`。
8. `continuous block 2`。
9. `idle_block`。
10. `idle_prepare`。
11. 自动保存。

可选阶段：

- `practice` 仅在 `practice_sec > 0` 时插入到 `calibration` 和 `MI run 1` 之间；默认关闭。

默认关键参数：

- `trials_per_class=10`
- `run_count=3`
- `baseline/cue/imagery/iti = 2.0/2.0/4.0/2.0s`
- `run_rest_sec=60`
- `long_run_rest_every=2`
- `long_run_rest_sec=120`
- `practice_sec=0`（保留功能，默认关闭）
- `idle_block = 2 x 60s`
- `idle_prepare = 2 x 60s`
- `continuous = 2 x 240s`
- `continuous_command = 4-5s`
- `continuous_gap = 2-3s`
- `include_eyes_closed_rest_in_gate_neg=False`

## 5. 保存目录与命名

每次保存到：

```text
datasets/custom_mi/sub-<subject>/ses-<session>/
```

同一次保存共享一个 `run_stem`：

```text
sub-<subject>_ses-<session>_run-<NNN>_tpc-<TT>_n-<NNN>_ok-<NNN>
```

注意：

- 文件名里的 `run-<NNN>` 表示 `save_index`，即同一 `subject/session` 目录下第几次落盘。
- 它不等于会话内部第几个 MI run。
- 会话内部 MI 轮次统一写在 `mi_run_index` 字段里。

## 6. 训练现在读取什么

训练只认这四类任务文件：

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

不再使用旧 `*_epochs.npz`。

训练侧的基本规则：

- 主分类只读 `*_mi_epochs.npz`
- gate 只读 `*_gate_epochs.npz`
- artifact rejector 只读 `*_artifact_epochs.npz`
- continuous 只做 online-like 评估，不并入主分类训练
- gate 中来源于 `continuous_*` 的负样本会在训练时自动剔除

## 7. Viewer 现在看什么

`viewer/` 现在默认扫描 `*_session_meta.json`，并按一个完整 run bundle 观察数据。

04 现在会联合读取：

- `session_meta.json`
- `quality_report.json`
- `board_map.json`
- `board_data.npy`
- `events.csv`
- `trials.csv`
- `segments.csv`
- `mi_epochs.npz`
- `gate_epochs.npz`
- `artifact_epochs.npz`
- `continuous.npz`

也就是说：

- 它现在既能看 accepted MI epoch
- 也能看 trial / segment / continuous prompt / artifact / gate 摘要
- 还能直接从 `board_data.npy` 回看连续保存信号，并叠加 event / segment

## 8. 当前数据存储的重要边界

- 预览滤波只用于显示，不写入保存文件。
- `*_board_data.npy` 是当前最接近板卡输出的保存物。
- `*_raw.fif` 只是 MNE 友好的工作视图，只包含 EEG 和 `STI 014`，不包含完整 board 行。
- `*_events.csv` 是原子事件真源。
- `*_segments.csv` 是从事件配对得到的语义 interval，不是 marker 真源。
- 所有派生 `npz` 信号单位都写成 `volt`。
- `collection_manifest.csv` 和 `session_meta.json["files"]` 里的路径都是相对路径。

## 9. 文档索引

- [code/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/README.md)
- [code/collection/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/collection/README.md)
- [code/collection/README_SAVE_NAMING.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/collection/README_SAVE_NAMING.md)
- [code/training/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/training/README.md)
- [code/training/README_DATA_LOADING.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/training/README_DATA_LOADING.md)
- [code/training/README_DATA_ADMISSION.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/training/README_DATA_ADMISSION.md)
- [code/realtime/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/realtime/README.md)
- [code/viewer/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/code/viewer/README.md)
- [datasets/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/datasets/README.md)
- [datasets/custom_mi/README.md](C:/Users/P1233/Desktop/brain/mi_classifier/datasets/custom_mi/README.md)

如果继续改采集流程、保存字段、训练读取规则或 viewer 支持范围，必须同步修改这些 README，避免代码和文档再次偏离。
