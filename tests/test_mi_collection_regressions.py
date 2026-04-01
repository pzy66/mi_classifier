import csv
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from src.mi_collection import SessionSettings, TrialRecord, make_event, save_mi_session


class MICollectionRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = PROJECT_ROOT / "runtime" / f"mi_collection_test_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_single_trial_session(self) -> dict[str, str]:
        sampling_rate = 250.0
        brainflow_data = np.zeros((10, 2400), dtype=np.float32)
        marker_row = 8

        for channel_index in range(8):
            brainflow_data[channel_index] = 10.0 * np.sin(
                np.linspace(0.0, 10.0, brainflow_data.shape[1], dtype=np.float32) + float(channel_index)
            )

        markers = [
            (0, "session_start", {}),
            (10, "trial_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (11, "fixation_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (12, "baseline_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (511, "baseline_end", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (512, "cue_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (513, "cue_left_hand", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (762, "imagery_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (763, "imagery_left_hand", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (1762, "imagery_end", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (1763, "iti_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (2263, "trial_end", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (2399, "session_end", {}),
        ]

        event_log = []
        for sample_index, event_name, extra in markers:
            event = make_event(event_name, **extra)
            event_log.append(event)
            brainflow_data[marker_row, sample_index] = float(event["marker_code"])

        settings = SessionSettings(
            subject_id="regression",
            session_id="single_trial",
            output_root=str(self.temp_dir),
            board_id=0,
            serial_port="COM3",
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            channel_positions=list(range(8)),
            trials_per_class=1,
            baseline_sec=2.0,
            cue_sec=1.0,
            imagery_sec=4.0,
            iti_sec=2.0,
            random_seed=0,
            run_count=1,
        )
        trial_records = [
            TrialRecord(
                trial_id=1,
                class_name="left_hand",
                display_name="left_hand",
                run_index=1,
                run_trial_index=1,
            )
        ]

        return save_mi_session(
            brainflow_data=brainflow_data,
            sampling_rate=sampling_rate,
            eeg_rows=list(range(8)),
            marker_row=marker_row,
            timestamp_row=None,
            settings=settings,
            event_log=event_log,
            trial_records=trial_records,
        )

    def test_trial_onsets_keep_earliest_generic_marker(self) -> None:
        result = self._save_single_trial_session()

        with Path(result["trials_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            first_row = next(csv.DictReader(handle))

        self.assertEqual(int(first_row["cue_onset_sample"]), 512)
        self.assertEqual(int(first_row["imagery_onset_sample"]), 762)

        with np.load(result["mi_epochs_path"], allow_pickle=True) as data:
            self.assertEqual(tuple(data["X_mi"].shape), (1, 8, 1000))

    def test_gate_negatives_include_short_baseline_and_iti(self) -> None:
        result = self._save_single_trial_session()

        with np.load(result["gate_epochs_path"], allow_pickle=True) as data:
            gate_neg = np.asarray(data["X_gate_neg"], dtype=np.float32)
            gate_neg_sources = [str(item) for item in np.asarray(data["gate_neg_sources"], dtype=object).tolist()]

        self.assertEqual(gate_neg.shape[1:], (8, 500))
        self.assertEqual(gate_neg.shape[0], 2)
        self.assertEqual(gate_neg_sources, ["baseline", "iti"])


if __name__ == "__main__":
    unittest.main()
