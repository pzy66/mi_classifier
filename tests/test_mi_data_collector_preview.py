import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))

from mi_data_collector import BoardCaptureWorker, BoardIds, build_fbcca_style_display_signal


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

    def test_impedance_channel_fast_switch_skips_stream_restart(self) -> None:
        cyton_board = getattr(BoardIds, "CYTON_BOARD", None)
        self.assertIsNotNone(cyton_board)

        worker = BoardCaptureWorker(
            board_id=int(cyton_board.value),
            serial_port="COM3",
            channel_positions=list(range(8)),
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
        )
        worker.selected_rows = list(range(8))
        worker.current_quality_mode = worker.MODE_IMPEDANCE
        worker._hw_impedance_channel = 1

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
        self.assertEqual(dummy_board.commands, ["z100Z", "z210Z"])
        self.assertEqual(stream_calls, {"stop": 0, "start": 0})
        self.assertEqual(worker._hw_impedance_channel, 2)
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
            window.close()
            self._pump(0.05)


if __name__ == "__main__":
    unittest.main()
