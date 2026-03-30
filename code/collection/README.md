# MI 采集模块说明（仅采集，不做判别）

主入口：
- `code/collection/run_collection_pycharm.py`

核心文件：
- `code/collection/mi_data_collector.py`
- `code/shared/src/mi_collection.py`

## 模块定位

本模块只负责：
1. 给受试者显示 MI 提示（全屏提示）
2. 从 BrainFlow 设备采集脑电并打事件标记
3. 保存为后续训练可直接读取的数据

本模块不负责：
- 不做实时分类
- 不加载任何训练模型

## 运行方式

```bash
conda activate MI
python code/collection/run_collection_pycharm.py
```

可选命令行参数（用于预填 UI）：
- `--serial-port COM3`
- `--output-root <路径>`
- `--subject-id 001`
- `--session-id 20260329_210000`
- `--synthetic`（演示模式）

## 采集流程（程序内部逻辑）

每个 trial 固定 4 个阶段：
1. `baseline`：准备静息
2. `cue`：提示即将进行哪类想象
3. `imagery`：正式想象阶段
4. `iti`：休息恢复

4 个类别：
- `left_hand`（左手想象）
- `right_hand`（右手想象）
- `feet`（双脚想象）
- `tongue`（舌头想象）

序列生成规则：
- `每类试次数 = trials_per_class`
- `总试次数 = 4 * trials_per_class`
- 每一轮 4 类各出现 1 次，轮内顺序由 `random_seed` 决定（可复现）

## UI 每个设置项具体作用

### 会话信息

- `被试编号`：写入输出目录名与元数据，生成 `sub-<subject_id>`。
- `Session 编号`：写入输出目录名与元数据，生成 `ses-<session_id>`。
- `输出目录`：本次 session 根目录，默认 `datasets/custom_mi`。
- `操作员`：仅写入 `session_meta.json`，用于记录采集者。
- `备注`：仅写入 `session_meta.json`，用于记录实验条件或异常说明。

### 设备参数

- `板卡类型`：BrainFlow `board_id`，决定采样率、可用 EEG 行、marker 行。
- `串口`：设备串口。非 `Synthetic` 必填，程序会在开始前校验。
- `通道名称`：逗号分隔字符串，决定保存数据的 EEG 通道标签与顺序。
- `通道位置`：逗号分隔整数，表示“从板卡 EEG 列表中取第几个通道”，与通道名称一一对应。

示例（Cyton 8 通道）：
- `通道名称`: `C3,Cz,C4,PO3,PO4,O1,Oz,O2`
- `通道位置`: `0,1,2,3,4,5,6,7`

约束：
- `通道名称` 个数必须等于 `通道位置` 个数
- `通道位置` 不能重复、不能为负
- `通道位置` 不能超过板卡实际 EEG 通道范围

### 试次流程

- `每类试次数`：每个 MI 类别采集多少个 trial（总 trial 为其 4 倍）。
- `准备阶段 (s)`：baseline 时长。
- `提示阶段 (s)`：cue 时长。
- `想象阶段 (s)`：imagery 时长。
- `休息阶段 (s)`：iti 时长。
- `随机种子`：控制 trial 顺序随机化结果。
- `同时导出 epochs.npz`：勾选后额外保存训练缓存文件。

## 建议设置（8 通道 Cyton）

- 板卡：`Cyton (0)`
- 串口：对应 `COMx`
- 通道名称/位置：使用默认 8 通道配置
- 每类试次数：`10`（总 40 trial，适合先做初版训练）
- 时长建议：`baseline=4s`、`cue=1.5s`、`imagery=4s`、`iti=2s`

## 运行时操作

操作员窗口按钮：
- `连接设备`：启动 BrainFlow 采集线程
- `开始采集`：开始 session 并切换受试者全屏提示
- `暂停/继续`：暂停当前阶段计时
- `标记坏试次`：将当前 trial 标记为坏段（训练时会过滤）
- `停止并保存`：停止采集并落盘
- `断开设备`：断开采集线程

受试者全屏快捷键：
- `Space`：暂停/继续
- `B`：标记坏试次
- `Esc`：停止并保存

## 事件标记（Marker）说明

基础事件：
- `100`：`session_start`
- `101`：`session_end`
- `110/111`：`baseline_start` / `baseline_end`
- `200/210`：`trial_start` / `trial_end`
- `320`：`imagery_end`
- `330`：`iti_start`
- `901`：`bad_trial_marked`
- `950/951`：`pause` / `resume`

类别事件（以左手为例）：
- `301`：`cue_left_hand`
- `311`：`imagery_left_hand`

其他类别：
- 右手 `302/312`
- 双脚 `303/313`
- 舌头 `304/314`

## 保存目录与文件格式

输出目录结构：

```text
datasets/custom_mi/
└─ sub-<subject_id>/
   └─ ses-<session_id>/
      ├─ session_raw.fif
      ├─ events.csv
      ├─ trials.csv
      ├─ session_meta.json
      ├─ quality_report.json
      └─ epochs.npz   # 仅在勾选“同时导出 epochs.npz”时生成
```

### 1) `session_raw.fif`

- MNE Raw 格式
- 通道数 = `N + 1`
- `N` 是你设置的 EEG 通道数（即 `len(channel_names)`）
- 额外 1 路是事件刺激通道 `STI 014`
- EEG 数据单位保存为 `Volt`

### 2) `events.csv`

逐事件记录：
- `event_name`
- `marker_code`
- `trial_id`
- `class_name`
- `sample_index`
- `elapsed_sec`
- `iso_time`

### 3) `trials.csv`

逐 trial 记录：
- trial 类别
- 是否有效（`accepted`）
- cue/imagery 起止样本点
- 备注

### 4) `session_meta.json`

包含：
- 本次所有设置（`session`）
- 采样率、样本点数、时长
- 通道映射（`selected_eeg_rows`）
- 有效/坏试次数
- 输出文件路径索引

### 5) `quality_report.json`

每通道质量统计（uV）：
- `std_uV`
- `peak_to_peak_uV`
- `rms_uV`

### 6) `epochs.npz`（训练缓存）

关键字段：
- `X`：`(n_trials, n_channels, n_samples)`，`float32`，单位 `volt`
- `y`：类别索引，`int64`
- `accepted`：trial 是否有效，`int8`
- `trial_ids`：trial 编号，`int64`
- `class_names`、`channel_names`
- `sampling_rate`
- `signal_unit`（当前为 `volt`）

## 通道数到底是多少？

由你在 UI 配置决定：
- `n_channels = len(通道名称)`
- 并且必须与 `通道位置` 数量一致

默认配置是 8 通道，所以默认保存为：
- `session_raw.fif`: 8 路 EEG + 1 路 `STI 014`
- `epochs.npz`: `X` 的第二维是 8

## 与训练模块的对齐要求

训练程序递归读取：
- `datasets/custom_mi/**/epochs.npz`

要求所有 session 保持一致：
- `channel_names` 一致
- `class_names` 一致
- `sampling_rate` 一致

训练时会自动过滤：
- `accepted = 0` 的坏试次

## 程序检查与已修复项

已确认逻辑：
- 采集模块与实时判别模块完全分离
- trial 事件链完整（baseline/cue/imagery/iti + trial/session）
- 保存单位一致（BrainFlow 输入 uV，落盘转 volt）

本次已修复：
- 非 `Synthetic` 板卡时，强制校验串口不能为空
- 开始采集前，提前校验输出目录可创建可写

## 常见问题

- 报错“请先连接设备”
  - 先点 `连接设备`，确认状态栏显示“设备已准备好”

- 报错“通道名称数量与连接时的通道数量不一致”
  - 断开设备后统一修改 `通道名称/通道位置`，再重连

- 训练时报没有 `epochs.npz`
  - 采集时勾选“同时导出 epochs.npz”，并至少完成一个有效 trial
