import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QApplication, QComboBox, QLabel, QPushButton


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))

from mi_data_collector import BoardCaptureWorker, BoardIds, RealtimeEEGPreviewWidget, build_fbcca_style_display_signal


class MIDataCollectorPreviewTests(unittest.TestCase):
    def test_fbcca_display_signal_is_display_only(self) -> None:
        sampling_rate = 250.0
        time_axis = np.arange(0.0, 5.0, 1.0 / sampling_rate, dtype=np.float64)
        source = (
            25.0
            + 120.0 * np.sin(2.0 * np.pi * 0.5 * time_axis)
            + 8.0 * np.sin(2.0 * np.pi * 12.0 * time_axis)
            + 3.0 * np.sin(2.0 * np.pi * 50.0 * time_axis)
        )
        source_before = source.copy()

        processed = build_fbcca_style_display_signal(source, sampling_rate)

        np.testing.assert_allclose(source, source_before)
        self.assertEqual(processed.shape, source.shape)
        self.assertFalse(np.allclose(processed, source_before))
        self.assertLess(abs(float(np.mean(processed))), 1e-8)

    def test_impedance_mode_switch_enables_all_channels(self) -> None:
        cyton_board = getattr(BoardIds, "CYTON_BOARD", None)
        self.assertIsNotNone(cyton_board)

        worker = BoardCaptureWorker(
            board_id=int(cyton_board.value),
            serial_port="COM3",
            channel_positions=list(range(8)),
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
        )
        worker.selected_rows = list(range(8))
        worker.current_quality_mode = worker.MODE_EEG
        worker._hw_impedance_channel = None

        class DummyBoard:
            def __init__(self) -> None:
                self.commands: list[str] = []

        dummy_board = DummyBoard()
        worker.board = dummy_board

        stream_calls = {"stop": 0, "start": 0}

        def fake_stop_stream(_board) -> None:
            stream_calls["stop"] += 1

        def fake_start_stream(_board, *, buffer_size=450000, retries=3, retry_delay_sec=0.08) -> None:
            del buffer_size, retries, retry_delay_sec
            stream_calls["start"] += 1

        def fake_config_board(_board, command: str, *, retries=5, retry_delay_sec=0.08) -> None:
            del retries, retry_delay_sec
            dummy_board.commands.append(command)

        worker._safe_stop_stream = fake_stop_stream
        worker._start_stream_with_retry = fake_start_stream
        worker._config_board_with_retry = fake_config_board

        ok, message = worker.switch_quality_mode_sync(
            target_mode=worker.MODE_IMPEDANCE,
            target_channel=2,
            reset_default=False,
        )

        self.assertTrue(ok)
        self.assertEqual(message, "")
        self.assertEqual(
            dummy_board.commands,
            [
                "z100Z",
                "z200Z",
                "z300Z",
                "z400Z",
                "z500Z",
                "z600Z",
                "z700Z",
                "z800Z",
                "z110Z",
                "z210Z",
                "z310Z",
                "z410Z",
                "z510Z",
                "z610Z",
                "z710Z",
                "z810Z",
            ],
        )
        self.assertEqual(stream_calls, {"stop": 1, "start": 1})
        self.assertIsNone(worker._hw_impedance_channel)
        self.assertEqual(worker.current_quality_mode, worker.MODE_IMPEDANCE)


class MIDataCollectorResponsiveUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _pump(self, seconds: float = 0.05) -> None:
        import time

        deadline = time.time() + seconds
        while time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    def _assert_wrapped_text_fits(self, label, *, message: str) -> None:
        available_width = max(20, label.contentsRect().width())
        text_rect = label.fontMetrics().boundingRect(
            QRect(0, 0, available_width, 4000),
            int(label.alignment()) | Qt.TextWordWrap,
            label.text(),
        )
        self.assertGreaterEqual(
            label.contentsRect().height() + 4,
            text_rect.height(),
            message,
        )

    def _collect_visible_text_clipping_issues(self, window) -> list[str]:
        issues: list[str] = []
        for label in window.findChildren(QLabel):
            if not label.isVisible() or not label.text().strip():
                continue
            if label.wordWrap():
                available_width = max(20, label.contentsRect().width())
                text_rect = label.fontMetrics().boundingRect(
                    QRect(0, 0, available_width, 4000),
                    int(label.alignment()) | Qt.TextWordWrap,
                    label.text(),
                )
                if label.contentsRect().height() + 2 < text_rect.height():
                    issues.append(f"wrapped label clipped: {label.text()[:24]}")
            else:
                required_width = label.fontMetrics().horizontalAdvance(label.text()) + 8
                if label.width() < required_width:
                    issues.append(f"label clipped: {label.text()[:24]}")

        for button in window.findChildren(QPushButton):
            if not button.isVisible() or not button.text().strip():
                continue
            required_width = button.fontMetrics().horizontalAdvance(button.text()) + 16
            if button.width() < required_width:
                issues.append(f"button clipped: {button.text()[:24]}")

        for combo in window.findChildren(QComboBox):
            if not combo.isVisible():
                continue
            required_width = combo.fontMetrics().horizontalAdvance(combo.currentText()) + 36
            if combo.width() < required_width:
                issues.append(f"combo clipped: {combo.currentText()[:24]}")

        return issues

    def test_protocol_and_preview_copy_fit_compact_windows(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            for width, height in ((1280, 760), (1100, 680), (980, 640)):
                window.resize(width, height)
                self._pump(0.1)

                self._assert_wrapped_text_fits(
                    window.preview_status_label,
                    message=f"preview status clipped at {width}x{height}",
                )

                window.session_tabs.setCurrentIndex(2)
                self._pump(0.05)
                self._assert_wrapped_text_fits(
                    window.protocol_text_label,
                    message=f"protocol copy clipped at {width}x{height}",
                )
                protocol_page = window.session_tabs.currentWidget()
                protocol_pos = window.protocol_text_label.mapTo(protocol_page, window.protocol_text_label.rect().topLeft())
                self.assertLessEqual(
                    protocol_pos.y() + window.protocol_text_label.height(),
                    protocol_page.height() + 2,
                    f"protocol copy overflowed its page at {width}x{height}",
                )
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.close()
            self._pump(0.05)

    def test_startup_layout_keeps_primary_content_visible_before_connection(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            for width, height in ((1280, 760), (980, 640)):
                window.resize(width, height)
                self._pump(0.12)

                self.assertFalse(window._preview_focus_active(), f"preview focus should stay off before connection at {width}x{height}")
                self.assertTrue(window.session_tabs.isVisible(), f"session tabs hidden before connection at {width}x{height}")
                self.assertFalse(window.preview_group.isVisible(), f"preview group should stay hidden before connection at {width}x{height}")
                self.assertFalse(window.preview_status_label.isVisible(), f"preview status should stay hidden before connection at {width}x{height}")
                self.assertTrue(window.config_stack.isVisible(), f"config stack hidden before connection at {width}x{height}")
                self.assertTrue(window.hero_title_label.isVisible(), f"hero title hidden before connection at {width}x{height}")
                self.assertIn("EEG / 阻抗", window.preview_group.title())
        finally:
            window.close()
            self._pump(0.05)

    def test_default_preview_layout_shows_all_eeg_channels(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            window.resize(1700, 960)
            self._pump(0.15)

            channel_names = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": channel_names,
                "selected_rows": list(range(8)),
            }
            window.preview_widget.configure_stream(sampling_rate=250.0, channel_names=channel_names)
            window.update_button_states()
            self._pump(0.05)

            metrics = window.preview_widget._layout_metrics(channel_count_hint=len(channel_names))
            content_rect = metrics["content_rect"]
            row_height = float(metrics["row_height"])
            row_gap = float(metrics["row_gap"])
            plot_width = float(content_rect.width()) - float(metrics["label_width"]) - float(metrics["right_info_width"])
            last_bottom = float(content_rect.top()) + len(channel_names) * row_height + (len(channel_names) - 1) * row_gap

            self.assertTrue(window.preview_group.isVisible())
            self.assertFalse(window.session_tabs.isVisible())
            self.assertFalse(window.config_stack.isVisible())
            self.assertGreaterEqual(window.preview_widget.width(), 1000)
            self.assertGreaterEqual(window.preview_widget.height(), 460)
            self.assertGreaterEqual(row_height, 40.0)
            self.assertGreaterEqual(plot_width, 900.0)
            self.assertLessEqual(last_bottom, float(content_rect.bottom()) + 1.0)
        finally:
            window.close()
            self._pump(0.05)

    def test_compact_impedance_preview_layout_shows_all_channels(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            window.resize(980, 640)
            self._pump(0.15)

            channel_names = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": channel_names,
                "selected_rows": list(range(8)),
            }
            window.preview_widget.configure_stream(sampling_rate=250.0, channel_names=channel_names)
            window._apply_preview_mode_locally(RealtimeEEGPreviewWidget.MODE_IMPEDANCE, channel=4)
            window.update_button_states()
            self._pump(0.08)

            metrics = window.preview_widget._layout_metrics(channel_count_hint=len(channel_names))
            content_rect = metrics["content_rect"]
            row_height = float(metrics["row_height"])
            row_gap = float(metrics["row_gap"])
            plot_width = float(content_rect.width()) - float(metrics["label_width"]) - float(metrics["right_info_width"])
            last_bottom = float(content_rect.top()) + len(channel_names) * row_height + (len(channel_names) - 1) * row_gap

            self.assertTrue(window._preview_focus_active())
            self.assertGreaterEqual(window.preview_widget.width(), 650)
            self.assertGreaterEqual(window.preview_widget.height(), 500)
            self.assertGreaterEqual(row_height, 40.0)
            self.assertGreaterEqual(plot_width, 440.0)
            self.assertLessEqual(last_bottom, float(content_rect.bottom()) + 1.0)
            self.assertIn("8通道阻抗", window.preview_mode_label.text())
            self.assertFalse(window.preview_prev_ch_button.isVisible())
            self.assertFalse(window.preview_next_ch_button.isVisible())
            self.assertFalse(window.preview_reset_button.isVisible())
        finally:
            window.close()
            self._pump(0.05)

    def test_preview_focus_layout_expands_preview_and_keeps_controls_readable(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            window.resize(980, 640)
            self._pump(0.15)

            channel_names = ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"]
            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": channel_names,
                "selected_rows": list(range(8)),
            }
            window.preview_widget.configure_stream(sampling_rate=250.0, channel_names=channel_names)
            window.update_button_states()
            self._pump(0.05)

            metrics = window.preview_widget._layout_metrics(channel_count_hint=len(channel_names))
            content_rect = metrics["content_rect"]
            plot_width = float(content_rect.width()) - float(metrics["label_width"]) - float(metrics["right_info_width"])

            self.assertGreaterEqual(window.preview_widget.width(), 650)
            self.assertGreaterEqual(plot_width, 450.0)
            self.assertFalse(window.session_tabs.isVisible())
            self.assertFalse(window.preview_status_label.isVisible())
            self.assertIsNotNone(window.control_group)
            self.assertGreaterEqual(window.control_group.height(), 240)

            for button in (
                window.connect_button,
                window.start_button,
                window.pause_button,
                window.bad_trial_button,
                window.stop_button,
                window.disconnect_button,
            ):
                self.assertIsNotNone(button)
                metrics = QFontMetrics(button.font())
                required_width = max(metrics.horizontalAdvance(line) for line in button.text().splitlines() or [""]) + 16
                self.assertGreaterEqual(
                    button.width(),
                    required_width,
                    f"{button.objectName()} text clipped at compact preview-focus width",
                )
                self.assertGreaterEqual(
                    button.height(),
                    56,
                    f"{button.objectName()} should expand vertically in preview-focus mode",
                )
                self.assertGreaterEqual(
                    button.font().pointSize(),
                    window.font().pointSize() + 2,
                    f"{button.objectName()} font should scale up in preview-focus mode",
                )
        finally:
            window.close()
            self._pump(0.05)

    def test_connected_state_enters_preview_focus_layout(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            window.resize(1700, 960)
            self._pump(0.15)

            window.device_info = {
                "sampling_rate": 250.0,
                "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
                "selected_rows": list(range(8)),
            }
            window.preview_widget.configure_stream(
                sampling_rate=250.0,
                channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            )
            window.update_button_states()
            self._pump(0.1)

            splitter_sizes = window.main_splitter.sizes()
            self.assertFalse(window.session_tabs.isVisible())
            self.assertFalse(window.log_group.isVisible())
            self.assertFalse(window.hero_subtitle_label.isVisible())
            self.assertFalse(window.preview_status_label.isVisible())
            self.assertGreater(splitter_sizes[1], splitter_sizes[0])
            self.assertGreaterEqual(window.preview_widget.minimumHeight(), 520)
            self.assertGreaterEqual(window.preview_widget.height(), 700)
            self.assertGreaterEqual(window.preview_widget.width(), 1000)
            self.assertIn("质量检查", window.preview_group.title())

        finally:
            window.close()
            self._pump(0.05)

    def test_running_layout_restores_session_tabs_and_text_fit(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            for width, height in ((1280, 760), (1100, 680), (980, 640)):
                window.resize(width, height)
                window.session_running = True
                window.waiting_for_save = False
                window.update_button_states()
                self._pump(0.12)

                self.assertTrue(window.session_tabs.isVisible(), f"session tabs hidden at {width}x{height}")
                self.assertTrue(window.preview_status_label.isVisible(), f"preview status hidden at {width}x{height}")
                self.assertEqual(window.log_group.isVisible(), height >= 720, f"log visibility mismatch at {width}x{height}")

                window.session_tabs.setCurrentIndex(0)
                self._pump(0.03)
                self._assert_wrapped_text_fits(
                    window.preview_status_label,
                    message=f"running preview status clipped at {width}x{height}",
                )

                window.session_tabs.setCurrentIndex(1)
                self._pump(0.03)
                self._assert_wrapped_text_fits(
                    window.sequence_hint_label,
                    message=f"sequence hint clipped at {width}x{height}",
                )
                self._assert_wrapped_text_fits(
                    window.sequence_label,
                    message=f"sequence label clipped at {width}x{height}",
                )

                window.session_tabs.setCurrentIndex(2)
                self._pump(0.03)
                self._assert_wrapped_text_fits(
                    window.protocol_text_label,
                    message=f"running protocol clipped at {width}x{height}",
                )
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.close()
            self._pump(0.05)

    def test_running_layout_keeps_config_controls_readable(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            for width, height in ((1280, 760), (1100, 680), (980, 640)):
                window.resize(width, height)
                window.session_running = True
                window.waiting_for_save = False
                window.update_button_states()
                self._pump(0.12)

                for button in (
                    window.connect_button,
                    window.start_button,
                    window.pause_button,
                    window.bad_trial_button,
                    window.stop_button,
                    window.disconnect_button,
                ):
                    self.assertIsNotNone(button)
                    metrics = QFontMetrics(button.font())
                    required_width = max(metrics.horizontalAdvance(line) for line in button.text().splitlines() or [""]) + 16
                    self.assertGreaterEqual(
                        button.width(),
                        required_width,
                        f"{button.objectName()} clipped in running layout at {width}x{height}",
                    )

                for index in (0, 3, 4, 8, 10):
                    window.config_section_combo.setCurrentIndex(index)
                    self._pump(0.03)
                    page = window.config_stack.currentWidget()
                    self.assertIsNotNone(page)
                    self.assertGreaterEqual(
                        page.height() + 2,
                        page.sizeHint().height(),
                        f"config section {index} clipped in running layout at {width}x{height}",
                    )
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.close()
            self._pump(0.05)

    def test_focus_panel_labels_remain_readable_in_compact_windows(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            window.phase_label.setText("质量检查")
            window.countdown_label.setText("剩余时间：12.5s")
            window.trial_banner_label.setText("当前试次：左手 12 / 40")
            window.instruction_label.setText("请先检查 8 个通道脑电波形和接触质量，确认稳定后再开始正式采集。")
            window.progress_text.setText("总进度：12 / 40")
            window.next_task_label.setText("下一任务：想象 4.0s")

            for width, height in ((1280, 760), (1100, 680), (980, 640)):
                window.resize(width, height)
                self._pump(0.12)

                self._assert_wrapped_text_fits(
                    window.instruction_label,
                    message=f"instruction label clipped at {width}x{height}",
                )

                for label, minimum_padding, message in (
                    (window.phase_label, 28, "phase label"),
                    (window.countdown_label, 28, "countdown label"),
                    (window.trial_banner_label, 20, "trial banner label"),
                    (window.progress_text, 10, "progress text"),
                    (window.next_task_label, 10, "next task label"),
                ):
                    metrics = QFontMetrics(label.font())
                    required_width = metrics.horizontalAdvance(label.text()) + minimum_padding
                    self.assertGreaterEqual(
                        label.width(),
                        required_width,
                        f"{message} clipped at {width}x{height}",
                    )
        finally:
            window.close()
            self._pump(0.05)

    def test_startup_layout_visible_controls_do_not_clip(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            for width, height in ((1280, 760), (1100, 680), (980, 640)):
                window.resize(width, height)
                self._pump(0.12)
                for index in range(window.config_stack.count()):
                    window.config_section_combo.setCurrentIndex(index)
                    self._pump(0.03)
                    issues = self._collect_visible_text_clipping_issues(window)
                    self.assertFalse(
                        issues,
                        f"startup layout clipped at {width}x{height} section {index}: {issues}",
                    )
        finally:
            window.close()
            self._pump(0.05)

    def test_running_layout_visible_controls_do_not_clip(self) -> None:
        from mi_data_collector import MIDataCollectorWindow

        window = MIDataCollectorWindow()
        window.show()
        try:
            for width, height in ((1280, 760), (1100, 680), (980, 640)):
                window.resize(width, height)
                window.session_running = True
                window.waiting_for_save = False
                window.update_button_states()
                self._pump(0.12)
                for index in range(window.config_stack.count()):
                    window.config_section_combo.setCurrentIndex(index)
                    self._pump(0.03)
                    issues = self._collect_visible_text_clipping_issues(window)
                    self.assertFalse(
                        issues,
                        f"running layout clipped at {width}x{height} section {index}: {issues}",
                    )
        finally:
            window.session_running = False
            window.waiting_for_save = False
            window.close()
            self._pump(0.05)


if __name__ == "__main__":
    unittest.main()
