# custom_mi

这里用于存放你自己采集的运动想象脑电数据。

目录结构约定：

```text
datasets/custom_mi/
`-- sub-001/
    `-- ses-20260327_204500/
        |-- session_raw.fif
        |-- session_meta.json
        |-- events.csv
        |-- trials.csv
        `-- epochs.npz
```

说明：
- `session_raw.fif` 是主原始文件，最重要
- `events.csv` 记录所有事件标记
- `trials.csv` 记录每个 trial 的关键时间点
- `epochs.npz` 是训练缓存，不是唯一真源
