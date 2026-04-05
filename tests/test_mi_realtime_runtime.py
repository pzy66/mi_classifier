import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from sklearn.exceptions import InconsistentVersionWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REALTIME_ROOT = PROJECT_ROOT / "code" / "realtime"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(REALTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REALTIME_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from brainflow.board_shim import BoardIds

from mi_realtime_infer_only import MIRealtimeWindow, resolve_board_channel_positions
from src.realtime_mi import load_realtime_model


class _StubPredictor:
    def __init__(self) -> None:
        self.artifact = {
            "class_names": ["left_hand", "right_hand", "feet", "tongue"],
            "display_class_names": ["LEFT HAND", "RIGHT HAND", "FEET", "TONGUE"],
            "channel_names": ["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            "selected_pipeline": "stub",
        }
        self.expected_channel_count = len(self.artifact["channel_names"])
        self.minimum_window_sec = 2.0


class MIRealtimeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _load_model(path: Path) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            return load_realtime_model(path)

    def test_load_legacy_custom_bank_backfills_required_preprocessing_fields(self) -> None:
        artifact = self._load_model(PROJECT_ROOT / "code" / "realtime" / "models" / "custom_mi_realtime.joblib")

        self.assertEqual(str(artifact.get("artifact_type")), "multi_window_bank")
        self.assertIn("preprocessing", artifact)
        self.assertTrue(isinstance(artifact.get("members"), list) and artifact["members"])

        top_preproc = dict(artifact["preprocessing"])
        member_preproc = dict(artifact["members"][0]["preprocessing"])

        self.assertEqual(top_preproc["epoch_window"], [0.0, 2.0])
        self.assertAlmostEqual(float(top_preproc["window_offset_sec"]), 0.25, places=6)
        self.assertEqual([float(item) for item in top_preproc["window_offset_secs_used"]], [0.25])
        self.assertEqual(member_preproc["epoch_window"], [0.0, 2.0])
        self.assertAlmostEqual(float(member_preproc["window_offset_sec"]), 0.25, places=6)
        self.assertEqual([float(item) for item in member_preproc["window_offset_secs_used"]], [0.25])

    def test_load_legacy_single_window_model_backfills_required_preprocessing_fields(self) -> None:
        artifact = self._load_model(PROJECT_ROOT / "code" / "realtime" / "models" / "subject_1_mi.joblib")

        preprocessing = dict(artifact["preprocessing"])
        self.assertEqual(preprocessing["epoch_window"], [0.0, 2.0])
        self.assertAlmostEqual(float(preprocessing["window_offset_sec"]), 0.0, places=6)
        self.assertEqual([float(item) for item in preprocessing["window_offset_secs_used"]], [0.0])

    def test_resolve_board_channel_positions_auto_maps_exact_match_board(self) -> None:
        positions = resolve_board_channel_positions(
            board_id=int(BoardIds.CYTON_BOARD.value),
            expected_channels=8,
            positions=None,
            model_channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
        )

        self.assertEqual(positions, list(range(8)))

    def test_refresh_serial_ports_prefers_detected_port_when_config_is_blank(self) -> None:
        with mock.patch("mi_realtime_infer_only.detect_serial_ports", return_value=["COM4"]):
            window = MIRealtimeWindow(
                {
                    "realtime_mode": "continuous",
                    "serial_port": "",
                    "board_id": int(BoardIds.CYTON_BOARD.value),
                    "step_sec": 0.25,
                    "protocol_baseline_sec": 2.0,
                    "protocol_cue_sec": 2.0,
                    "protocol_imagery_sec": 4.0,
                    "protocol_iti_sec": 2.0,
                    "protocol_trials_per_class": 4,
                    "protocol_random_seed": 42,
                },
                _StubPredictor(),
            )
            try:
                self.assertEqual(window.serial_combo.currentText(), "COM4")
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
