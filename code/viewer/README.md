# viewer 模块说明

这个目录只做一件事：**查看采集后的 npz 数据**。  
与采集、训练、实时判别完全分开。

## 文件

- `mi_npz_viewer.py`：主程序（数字统计 + 波形 + 频谱 + 导出）
- `run_npz_viewer_pycharm.py`：PyCharm 启动入口

## 启动方式

1. 项目根目录启动（推荐）  
`python run_04_view_collected_npz.py`

2. viewer 目录启动  
`python code/viewer/run_npz_viewer_pycharm.py`

3. 指定文件启动  
`python code/viewer/mi_npz_viewer.py --npz <你的npz路径>`

## 支持读取的文件命名

- 新命名：`*_epochs.npz`（采集程序自动编号后的文件）
- 旧命名：`epochs.npz`（历史兼容）

## 功能点

- 中文界面（含中文字体回退设置）
- 自动扫描数据集目录并列出可视化文件
- 总览统计：试次数、有效率、采样率、时长、RMS
- 类别统计：每类总数/有效/无效
- 通道统计：Mean/Std/RMS/PtP/AbsMean（uV）
- 图像模式：
  - 单试次波形
  - 类别平均波形
  - 类别平均频谱（PSD）
- 统计导出：
  - JSON
  - CSV（类别统计 + 通道统计）
