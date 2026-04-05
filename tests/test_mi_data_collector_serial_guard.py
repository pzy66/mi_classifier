import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_ROOT = PROJECT_ROOT / "code" / "collection"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(COLLECTION_ROOT) not in sys.path:
    sys.path.insert(0, str(COLLECTION_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from mi_data_collector import MIDataCollectorWindow


class MIDataCollectorSerialGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @patch("mi_data_collector.detect_serial_ports", return_value=[])
    def test_refresh_serial_ports_does_not_fallback_to_com3_when_nothing_is_detected(self, _mock_detect) -> None:
        window = MIDataCollectorWindow()
        try:
            window.serial_combo.setCurrentText("")
            window.refresh_serial_ports()
            self.assertEqual(window.serial_combo.currentText().strip(), "")
        finally:
            window.close()

    @patch("mi_data_collector.detect_serial_ports", return_value=[])
    def test_collect_settings_explains_when_no_ports_are_available(self, _mock_detect) -> None:
        window = MIDataCollectorWindow()
        try:
            window.serial_combo.setCurrentText("COM4")
            with self.assertRaisesRegex(ValueError, "当前没有检测到可用串口"):
                window.collect_settings()
        finally:
            window.close()
