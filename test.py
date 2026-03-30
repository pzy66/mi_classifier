from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import (
    DataFilter,
    FilterTypes,
    DetrendOperations,
    NoiseTypes
)


class CytonRealtimePlotter:
    def __init__(self, board_id=0, com_port="COM4", window_seconds=5):
        self.board_id = board_id
        self.com_port = com_port
        self.window_seconds = window_seconds

        self.board = None
        self.sampling_rate = None
        self.eeg_channels = None

        self.max_points = None
        self.buffers = []
        self.lines = []
        self.axes = None
        self.fig = None
        self.ani = None

        self.is_prepared = False
        self.is_streaming = False
        self.is_closed = False

    def setup_board(self):
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = self.com_port

        self.board = BoardShim(self.board_id, params)

        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

        self.max_points = int(self.window_seconds * self.sampling_rate)
        self.buffers = [deque(maxlen=self.max_points) for _ in range(len(self.eeg_channels))]

        print("=" * 60)
        print(f"Board ID: {self.board_id}")
        print(f"串口: {self.com_port}")
        print(f"采样率: {self.sampling_rate} Hz")
        print(f"EEG 通道索引: {self.eeg_channels}")
        print(f"显示窗口: 最近 {self.window_seconds} 秒")
        print("滤波链: 去直流 -> 50 Hz 工频滤波 -> 1–40 Hz 带通")
        print("=" * 60)

        print("正在连接 Cyton ...")
        self.board.prepare_session()
        self.is_prepared = True

        self.board.start_stream(45000, "")
        self.is_streaming = True
        print("连接成功，开始实时绘图。")

    def setup_plot(self):
        ch_num = len(self.eeg_channels)

        self.fig, self.axes = plt.subplots(ch_num, 1, figsize=(14, 10), sharex=True)

        if ch_num == 1:
            self.axes = [self.axes]

        self.fig.suptitle("OpenBCI Cyton 8通道实时EEG波形（1–40 Hz + 50 Hz工频滤波）", fontsize=14)

        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], linewidth=1.0)
            self.lines.append(line)

            ax.set_ylabel(f"CH{i + 1}")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xlim(-self.window_seconds, 0)
            ax.set_ylim(-100, 100)

        self.axes[-1].set_xlabel("时间 / 秒")
        self.fig.canvas.mpl_connect("close_event", self.on_close)

    def fetch_new_data(self):
        """
        从 BrainFlow 取出当前全部新数据，并清空内部缓冲区。
        """
        if self.board is None or self.is_closed:
            return

        data = self.board.get_board_data()
        if data is None or data.shape[1] == 0:
            return

        eeg_data = data[self.eeg_channels, :]  # shape: [8, N]

        n_samples = eeg_data.shape[1]
        n_channels = eeg_data.shape[0]

        for j in range(n_samples):
            for ch in range(n_channels):
                self.buffers[ch].append(float(eeg_data[ch, j]))

    def compute_x_axis(self, n_points):
        if n_points <= 0:
            return np.array([])
        return np.arange(-n_points, 0, dtype=float) / self.sampling_rate

    def get_filtered_window_data(self):
        """
        对当前窗口副本做:
        1) 去直流
        2) 50 Hz 工频滤波
        3) 1–40 Hz 带通
        """
        filtered_list = []

        for ch in range(len(self.buffers)):
            y = np.array(self.buffers[ch], dtype=np.float64)

            # 点数太少时不滤波
            if y.size > 10:
                y_filtered = y.copy()

                # 1. 去直流/基线偏移
                DataFilter.detrend(
                    y_filtered,
                    DetrendOperations.CONSTANT.value
                )

                # 2. 50 Hz 工频滤波
                DataFilter.remove_environmental_noise(
                    y_filtered,
                    self.sampling_rate,
                    NoiseTypes.FIFTY.value
                )

                # 3. 1–40 Hz 带通
                DataFilter.perform_bandpass(
                    y_filtered,
                    self.sampling_rate,
                    1.0,
                    40.0,
                    4,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                    0
                )

                filtered_list.append(y_filtered)
            else:
                filtered_list.append(y)

        return filtered_list

    def auto_adjust_ylim(self, filtered_list):
        for ch, ax in enumerate(self.axes):
            y = filtered_list[ch]
            if y.size < 20:
                continue

            low = np.percentile(y, 5)
            high = np.percentile(y, 95)

            if high - low < 20:
                center = (high + low) / 2.0
                low = center - 10
                high = center + 10

            pad = 0.2 * (high - low)
            ax.set_ylim(low - pad, high + pad)

    def animate(self, frame):
        if self.is_closed:
            return self.lines

        self.fetch_new_data()

        n_points = len(self.buffers[0]) if len(self.buffers) > 0 else 0
        if n_points < 2:
            return self.lines

        x = self.compute_x_axis(n_points)
        filtered_list = self.get_filtered_window_data()

        for ch, line in enumerate(self.lines):
            y = filtered_list[ch]

            if len(y) != len(x):
                min_len = min(len(y), len(x))
                line.set_data(x[-min_len:], y[-min_len:])
            else:
                line.set_data(x, y)

        self.auto_adjust_ylim(filtered_list)
        return self.lines

    def on_close(self, event=None):
        if self.is_closed:
            return

        self.is_closed = True
        print("正在停止采集并释放资源...")

        try:
            if self.board is not None and self.is_streaming:
                self.board.stop_stream()
                self.is_streaming = False
        except Exception as e:
            print(f"stop_stream 出错: {e}")

        try:
            if self.board is not None and self.is_prepared:
                self.board.release_session()
                self.is_prepared = False
        except Exception as e:
            print(f"release_session 出错: {e}")

        print("资源已释放。")

    def run(self):
        try:
            self.setup_board()
            self.setup_plot()

            self.ani = FuncAnimation(
                self.fig,
                self.animate,
                interval=50,
                blit=False,
                cache_frame_data=False
            )

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()

        except KeyboardInterrupt:
            print("用户中断。")
            self.on_close()
        except Exception as e:
            print(f"程序出错: {e}")
            self.on_close()


if __name__ == "__main__":
    plotter = CytonRealtimePlotter(
        board_id=0,
        com_port="COM4",
        window_seconds=5
    )
    plotter.run()