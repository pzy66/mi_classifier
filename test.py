import matplotlib
matplotlib.use("TkAgg")   # 必须在 pyplot 之前

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import (
    DataFilter,
    FilterTypes,
    DetrendOperations,
    NoiseTypes
)

# 中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class CytonRealtimeDualMode:
    MODE_EEG = "EEG"
    MODE_IMP = "IMP"
    DISPLAY_HIGHPASS_HZ = 1.0
    DISPLAY_LOWPASS_HZ = 40.0
    DISPLAY_FILTER_ORDER = 4

    def __init__(
        self,
        board_id=0,
        com_port="COM3",
        window_seconds=5,
        impedance_channel=1,
        lead_off_current_amps=6e-9,     # 常见 Cyton 阻抗换算假设
        series_resistor_ohms=2200.0     # OpenBCI 论坛提到减去 2.2k 串联电阻
    ):
        self.board_id = board_id
        self.com_port = com_port
        self.window_seconds = window_seconds
        self.impedance_channel = int(np.clip(impedance_channel, 1, 8))

        self.lead_off_current_amps = float(lead_off_current_amps)
        self.series_resistor_ohms = float(series_resistor_ohms)

        self.board = None
        self.sampling_rate = None
        self.eeg_channels = None

        self.max_points = None
        self.buffers = []
        self.lines = []
        self.axes = None
        self.fig = None
        self.ani = None
        self.status_text = None
        self.impedance_texts = []

        self.mode = self.MODE_EEG
        self.pending_action = None

        self.is_prepared = False
        self.is_streaming = False
        self.is_closed = False
        self.frame_count = 0

        self.last_impedance_ohms = [None] * 8

        # 按钮对象
        self.btn_eeg = None
        self.btn_imp = None
        self.btn_prev = None
        self.btn_next = None
        self.btn_reset = None
        self.btn_quit = None

    # =========================
    # 板卡初始化与资源管理
    # =========================
    def setup_board(self):
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = self.com_port

        self.board = BoardShim(self.board_id, params)

        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

        self.max_points = int(self.window_seconds * self.sampling_rate)
        self.buffers = [deque(maxlen=self.max_points) for _ in range(len(self.eeg_channels))]
        self.last_impedance_ohms = [None] * len(self.eeg_channels)

        print("=" * 70)
        print("OpenBCI Cyton 双模式程序")
        print(f"Board ID: {self.board_id}")
        print(f"串口: {self.com_port}")
        print(f"采样率: {self.sampling_rate} Hz")
        print(f"EEG 通道索引: {self.eeg_channels}")
        print(f"显示窗口: 最近 {self.window_seconds} 秒")
        print("快捷键：")
        print("  e -> EEG模式")
        print("  i -> 阻抗模式")
        print("  ← / → -> 切换阻抗检测通道")
        print("  r -> 恢复默认设置")
        print("  q -> 退出")
        print("=" * 70)

        print("正在连接 Cyton ...")
        self.board.prepare_session()
        self.is_prepared = True

        self.start_stream()
        print("连接成功，开始实时绘图。")

    def start_stream(self):
        if self.board is not None and not self.is_streaming:
            self.board.start_stream(45000, "")
            self.is_streaming = True

    def stop_stream(self):
        if self.board is not None and self.is_streaming:
            self.board.stop_stream()
            self.is_streaming = False

    def clear_buffers(self):
        for buf in self.buffers:
            buf.clear()

    def cleanup(self):
        if self.is_closed:
            return

        self.is_closed = True
        print("正在停止采集并释放资源...")

        board = self.board
        self.board = None

        try:
            if board is not None and self.is_streaming:
                board.stop_stream()
                self.is_streaming = False
        except Exception as e:
            print(f"stop_stream 出错: {e}")

        try:
            if board is not None and self.is_prepared:
                board.release_session()
                self.is_prepared = False
        except Exception as e:
            print(f"release_session 出错: {e}")

        print("资源已释放。")

    def on_close(self, event=None):
        self.cleanup()

    # =========================
    # Cyton 阻抗命令
    # z(CHANNEL, PCHAN, NCHAN)Z
    # =========================
    def build_impedance_command(self, channel, test_p=True, test_n=False):
        p = 1 if test_p else 0
        n = 1 if test_n else 0
        return f"z{channel}{p}{n}Z"

    def send_board_command(self, cmd):
        if self.board is None:
            return False
        try:
            resp = self.board.config_board(cmd)
            print(f"发送命令 {cmd!r} -> {resp}")
            return True
        except Exception as e:
            print(f"发送命令 {cmd!r} 失败: {e}")
            return False

    def apply_mode_change_safely(self, target_mode=None, target_channel=None, reset_default=False):
        try:
            self.stop_stream()

            # 先关闭所有通道阻抗测试
            for ch in range(1, len(self.eeg_channels) + 1):
                self.send_board_command(self.build_impedance_command(ch, False, False))

            if reset_default:
                self.send_board_command("d")

            if target_mode == self.MODE_IMP:
                self.send_board_command(self.build_impedance_command(target_channel, True, False))
                self.mode = self.MODE_IMP
                self.impedance_channel = target_channel

            elif target_mode == self.MODE_EEG:
                self.send_board_command("d")
                self.mode = self.MODE_EEG

            self.clear_buffers()
            self.last_impedance_ohms = [None] * len(self.eeg_channels)

        finally:
            self.start_stream()
            self.update_title_and_status()
            self.update_impedance_texts()

    # =========================
    # 图形界面
    # =========================
    def setup_plot(self):
        ch_num = len(self.eeg_channels)

        self.fig, self.axes = plt.subplots(ch_num, 1, figsize=(15.5, 10), sharex=True)

        if ch_num == 1:
            self.axes = [self.axes]

        # 给右侧阻抗值、底部按钮留空间
        plt.subplots_adjust(left=0.08, right=0.84, top=0.92, bottom=0.14, hspace=0.20)

        self.lines = []
        self.impedance_texts = []

        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], linewidth=1.0)
            self.lines.append(line)

            ax.set_ylabel(f"CH{i + 1}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlim(-self.window_seconds, 0)
            ax.set_ylim(-150, 150)

            txt = ax.text(
                1.02, 0.50, "",
                transform=ax.transAxes,
                va="center", ha="left",
                fontsize=10,
                clip_on=False
            )
            self.impedance_texts.append(txt)

        self.axes[-1].set_xlabel("时间 / 秒")
        self.status_text = self.fig.text(0.08, 0.965, "", fontsize=11, va="top")

        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_mouse_click)

        # 按钮
        ax_btn_eeg = self.fig.add_axes([0.08, 0.03, 0.10, 0.05])
        ax_btn_imp = self.fig.add_axes([0.20, 0.03, 0.10, 0.05])
        ax_btn_prev = self.fig.add_axes([0.34, 0.03, 0.10, 0.05])
        ax_btn_next = self.fig.add_axes([0.46, 0.03, 0.10, 0.05])
        ax_btn_reset = self.fig.add_axes([0.60, 0.03, 0.12, 0.05])
        ax_btn_quit = self.fig.add_axes([0.76, 0.03, 0.10, 0.05])

        self.btn_eeg = Button(ax_btn_eeg, "EEG模式")
        self.btn_imp = Button(ax_btn_imp, "阻抗模式")
        self.btn_prev = Button(ax_btn_prev, "上一通道")
        self.btn_next = Button(ax_btn_next, "下一通道")
        self.btn_reset = Button(ax_btn_reset, "恢复默认")
        self.btn_quit = Button(ax_btn_quit, "退出")

        self.btn_eeg.on_clicked(lambda event: self.set_pending_action("to_eeg"))
        self.btn_imp.on_clicked(lambda event: self.set_pending_action("to_imp"))
        self.btn_prev.on_clicked(lambda event: self.set_pending_action("prev_ch"))
        self.btn_next.on_clicked(lambda event: self.set_pending_action("next_ch"))
        self.btn_reset.on_clicked(lambda event: self.set_pending_action("reset"))
        self.btn_quit.on_clicked(lambda event: plt.close(self.fig))

        self.update_title_and_status()
        self.update_impedance_texts()

        try:
            self.fig.canvas.manager.window.focus_force()
        except Exception:
            pass

    def update_title_and_status(self):
        if self.fig is None:
            return

        if self.mode == self.MODE_EEG:
            self.fig.suptitle("OpenBCI Cyton 8通道实时EEG波形（EEG模式）", fontsize=14)
            text = (
                f"当前模式: EEG | 串口: {self.com_port} | "
                "滤波: 去直流 + 50 Hz工频滤波 + 1–40 Hz带通 | "
                "按 i 或点按钮进入阻抗模式"
            )
            for i, line in enumerate(self.lines, start=1):
                line.set_alpha(1.0)
                line.set_linewidth(1.0)
                self.axes[i - 1].set_facecolor("white")
        else:
            self.fig.suptitle(
                f"OpenBCI Cyton 阻抗检测模式（当前 CH{self.impedance_channel}）",
                fontsize=14
            )
            text = (
                f"当前模式: IMPEDANCE | 检测通道: CH{self.impedance_channel} | "
                "阻抗波形不做任何滤波 | 按 ←/→ 或点按钮切换通道"
            )
            for i, line in enumerate(self.lines, start=1):
                if i == self.impedance_channel:
                    line.set_alpha(1.0)
                    line.set_linewidth(1.6)
                    self.axes[i - 1].set_facecolor("#fff8e8")
                else:
                    line.set_alpha(0.30)
                    line.set_linewidth(0.8)
                    self.axes[i - 1].set_facecolor("white")

        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    def update_impedance_texts(self):
        if not self.impedance_texts:
            return

        for i, txt in enumerate(self.impedance_texts, start=1):
            if self.mode == self.MODE_EEG:
                txt.set_text("")
                continue

            if i == self.impedance_channel:
                z = self.last_impedance_ohms[i - 1]
                if z is None:
                    txt.set_text("Imp: --")
                else:
                    txt.set_text(f"Imp ≈ {z / 1000.0:.1f} kΩ")
            else:
                txt.set_text("Imp: --")

        if self.fig is not None:
            self.fig.canvas.draw_idle()

    # =========================
    # 数据获取与处理
    # =========================
    def fetch_new_data(self):
        if self.board is None or self.is_closed or not self.is_streaming:
            return

        data = self.board.get_board_data()
        if data is None or data.shape[1] == 0:
            return

        eeg_data = data[self.eeg_channels, :]

        for ch in range(eeg_data.shape[0]):
            self.buffers[ch].extend(eeg_data[ch, :].astype(float).tolist())

    def compute_x_axis(self, n_points):
        if n_points <= 0:
            return np.array([])
        return np.arange(-n_points, 0, dtype=float) / self.sampling_rate

    def estimate_impedance_ohms_from_raw_window(self, y_uV):
        """
        按你的要求：
        阻抗值直接基于当前窗口原始波形计算，不做任何滤波。
        """
        if y_uV is None or y_uV.size < 20:
            return None

        std_uV = float(np.std(y_uV, ddof=0))
        z_ohm = (np.sqrt(2.0) * std_uV * 1e-6) / self.lead_off_current_amps
        z_ohm -= self.series_resistor_ohms

        if not np.isfinite(z_ohm):
            return None
        return max(z_ohm, 0.0)

    def get_plot_data(self):
        """
        EEG模式: 去直流 -> 50 Hz陷波 -> 1–40 Hz带通
        阻抗模式: 完全不做滤波，直接显示原始窗口
        """
        out = []

        for ch in range(len(self.buffers)):
            y = np.array(self.buffers[ch], dtype=np.float64)

            if y.size <= 10:
                out.append(y)
                continue

            if self.mode == self.MODE_EEG:
                y_plot = y.copy()

                DataFilter.detrend(y_plot, DetrendOperations.CONSTANT.value)

                DataFilter.perform_highpass(
                    y_plot,
                    self.sampling_rate,
                    self.DISPLAY_HIGHPASS_HZ,
                    self.DISPLAY_FILTER_ORDER,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                    0
                )

                DataFilter.remove_environmental_noise(
                    y_plot,
                    self.sampling_rate,
                    NoiseTypes.FIFTY.value
                )

                DataFilter.perform_lowpass(
                    y_plot,
                    self.sampling_rate,
                    self.DISPLAY_LOWPASS_HZ,
                    self.DISPLAY_FILTER_ORDER,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
                    0
                )
                out.append(y_plot)
            else:
                # 阻抗模式：原始值，不滤波
                out.append(y)

        return out

    def update_impedance_values(self):
        if self.mode != self.MODE_IMP:
            return

        active_idx = self.impedance_channel - 1
        if active_idx < 0 or active_idx >= len(self.buffers):
            return

        y_raw = np.array(self.buffers[active_idx], dtype=np.float64)
        self.last_impedance_ohms[active_idx] = self.estimate_impedance_ohms_from_raw_window(y_raw)

    def auto_adjust_ylim(self, data_list):
        if self.frame_count % 10 != 0:
            return

        for ch, ax in enumerate(self.axes):
            y = data_list[ch]
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

    # =========================
    # 交互控制
    # =========================
    def set_pending_action(self, action):
        self.pending_action = action

    def on_mouse_click(self, event):
        try:
            self.fig.canvas.manager.window.focus_force()
        except Exception:
            pass

    def on_key_press(self, event):
        if event.key in ["e", "E"]:
            self.pending_action = "to_eeg"
        elif event.key in ["i", "I"]:
            self.pending_action = "to_imp"
        elif event.key == "left":
            self.pending_action = "prev_ch"
        elif event.key == "right":
            self.pending_action = "next_ch"
        elif event.key in ["r", "R"]:
            self.pending_action = "reset"
        elif event.key in ["q", "Q"]:
            plt.close(self.fig)

    def handle_pending_action(self):
        if self.pending_action is None:
            return

        action = self.pending_action
        self.pending_action = None

        try:
            if action == "to_eeg":
                print("切换到 EEG 模式 ...")
                self.apply_mode_change_safely(target_mode=self.MODE_EEG)

            elif action == "to_imp":
                print(f"切换到阻抗模式，当前通道 CH{self.impedance_channel} ...")
                self.apply_mode_change_safely(
                    target_mode=self.MODE_IMP,
                    target_channel=self.impedance_channel
                )

            elif action == "prev_ch":
                self.impedance_channel -= 1
                if self.impedance_channel < 1:
                    self.impedance_channel = len(self.eeg_channels)

                print(f"阻抗检测通道 -> CH{self.impedance_channel}")
                if self.mode == self.MODE_IMP:
                    self.apply_mode_change_safely(
                        target_mode=self.MODE_IMP,
                        target_channel=self.impedance_channel
                    )
                else:
                    self.update_title_and_status()
                    self.update_impedance_texts()

            elif action == "next_ch":
                self.impedance_channel += 1
                if self.impedance_channel > len(self.eeg_channels):
                    self.impedance_channel = 1

                print(f"阻抗检测通道 -> CH{self.impedance_channel}")
                if self.mode == self.MODE_IMP:
                    self.apply_mode_change_safely(
                        target_mode=self.MODE_IMP,
                        target_channel=self.impedance_channel
                    )
                else:
                    self.update_title_and_status()
                    self.update_impedance_texts()

            elif action == "reset":
                print("恢复板卡默认设置 ...")
                self.apply_mode_change_safely(target_mode=self.MODE_EEG, reset_default=True)

        except Exception as e:
            print(f"处理模式切换失败: {e}")
            try:
                self.apply_mode_change_safely(target_mode=self.MODE_EEG, reset_default=True)
            except Exception as e2:
                print(f"回退 EEG 模式也失败: {e2}")

    # =========================
    # 动画循环
    # =========================
    def animate(self, frame):
        if self.is_closed:
            return self.lines

        self.frame_count += 1

        self.handle_pending_action()
        self.fetch_new_data()
        self.update_impedance_values()

        if len(self.buffers) == 0:
            return self.lines

        n_points = len(self.buffers[0])
        if n_points < 2:
            self.update_impedance_texts()
            return self.lines

        x = self.compute_x_axis(n_points)
        data_list = self.get_plot_data()

        for ch, line in enumerate(self.lines):
            y = data_list[ch]
            min_len = min(len(x), len(y))
            if min_len > 0:
                line.set_data(x[-min_len:], y[-min_len:])

        self.auto_adjust_ylim(data_list)
        self.update_impedance_texts()
        return self.lines

    # =========================
    # 主程序
    # =========================
    def run(self):
        try:
            self.setup_board()
            self.setup_plot()

            self.ani = FuncAnimation(
                self.fig,
                self.animate,
                interval=80,
                blit=False,
                cache_frame_data=False
            )

            plt.show(block=True)

        except KeyboardInterrupt:
            print("用户中断。")
        except Exception as e:
            print(f"程序出错: {e}")
        finally:
            self.cleanup()


if __name__ == "__main__":
    plotter = CytonRealtimeDualMode(
        board_id=0,
        com_port="COM3",      # 改成你的实际串口
        window_seconds=5,
        impedance_channel=1,
        lead_off_current_amps=6e-9,
        series_resistor_ohms=2200.0
    )
    plotter.run()
