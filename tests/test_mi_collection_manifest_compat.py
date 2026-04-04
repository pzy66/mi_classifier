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

from src.mi_collection import (
    COLLECTION_MANIFEST_FIELDS,
    SessionSettings,
    TrialRecord,
    create_session_folder,
    make_event,
    save_mi_session,
)


class MICollectionManifestCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = PROJECT_ROOT / "runtime" / f"mi_manifest_compat_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_single_trial_session(self, *, subject_id: str, session_id: str) -> dict[str, str]:
        sampling_rate = 250.0
        brainflow_data = np.zeros((10, 2700), dtype=np.float32)
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
            (1012, "imagery_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (1013, "imagery_left_hand", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (2012, "imagery_end", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (2013, "iti_start", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (2513, "trial_end", {"trial_id": 1, "class_name": "left_hand", "run_index": 1, "run_trial_index": 1}),
            (2699, "session_end", {}),
        ]

        event_log = []
        for sample_index, event_name, extra in markers:
            event = make_event(event_name, **extra)
            event_log.append(event)
            brainflow_data[marker_row, sample_index] = float(event["marker_code"])

        settings = SessionSettings(
            subject_id=subject_id,
            session_id=session_id,
            output_root=str(self.temp_dir),
            board_id=0,
            serial_port="COM3",
            channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
            channel_positions=list(range(8)),
            trials_per_class=1,
            baseline_sec=2.0,
            cue_sec=2.0,
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

    def test_save_upgrades_legacy_manifest_before_appending(self) -> None:
        manifest_path = self.temp_dir / "collection_manifest.csv"
        legacy_fields = [
            "saved_at",
            "subject_id",
            "session_id",
            "run_index",
            "run_stem",
            "trials_per_class",
            "mi_run_count",
            "trials_per_run",
            "trial_count",
            "accepted_trial_count",
            "rejected_trial_count",
            "sampling_rate_hz",
            "channel_names",
            "class_names",
            "mi_epochs_npz",
            "gate_epochs_npz",
            "artifact_npz",
            "continuous_npz",
            "epochs_npz",
            "session_raw_fif",
            "events_csv",
            "trials_csv",
            "session_meta_json",
        ]
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=legacy_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "saved_at": "2026-03-01T10:00:00",
                    "subject_id": "legacy_subject",
                    "session_id": "legacy_session",
                    "run_index": "7",
                    "run_stem": "sub-legacy_subject_ses-legacy_session_run-007_tpc-01_n-001_ok-001",
                    "trials_per_class": "1",
                    "mi_run_count": "1",
                    "trials_per_run": "4",
                    "trial_count": "1",
                    "accepted_trial_count": "1",
                    "rejected_trial_count": "0",
                    "sampling_rate_hz": "250.0",
                    "channel_names": "C3,Cz,C4,PO3,PO4,O1,Oz,O2",
                    "class_names": "left_hand,right_hand,feet,tongue",
                    "mi_epochs_npz": "",
                    "gate_epochs_npz": "legacy_gate.npz",
                    "artifact_npz": "legacy_artifact.npz",
                    "continuous_npz": "legacy_continuous.npz",
                    "epochs_npz": "legacy_epochs.npz",
                    "session_raw_fif": "legacy_raw.fif",
                    "events_csv": "legacy_events.csv",
                    "trials_csv": "legacy_trials.csv",
                    "session_meta_json": "legacy_session_meta.json",
                }
            )

        result = self._save_single_trial_session(subject_id="compat", session_id="manifest_fix")

        backup_paths = list(self.temp_dir.glob("collection_manifest_legacy_schema_*.csv"))
        self.assertEqual(len(backup_paths), 1)

        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            self.assertEqual(list(reader.fieldnames or []), COLLECTION_MANIFEST_FIELDS)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["schema_version"], "1")
        self.assertEqual(rows[0]["protocol_mode"], "full")
        self.assertEqual(rows[0]["save_index"], "7")
        self.assertEqual(rows[0]["mi_epochs_npz"], "legacy_epochs.npz")
        self.assertEqual(rows[0]["artifact_epochs_npz"], "legacy_artifact.npz")
        self.assertEqual(rows[1]["subject_id"], "compat")
        self.assertEqual(rows[1]["session_id"], "manifest_fix")
        self.assertEqual(rows[1]["protocol_mode"], "full")
        self.assertEqual(rows[1]["save_index"], "1")
        self.assertEqual(rows[1]["segments_csv"], Path(result["segments_csv_path"]).relative_to(self.temp_dir).as_posix())

    def test_legacy_session_meta_reserves_first_save_index(self) -> None:
        session_dir = create_session_folder(self.temp_dir, "legacy_subject", "legacy_session")
        (session_dir / "session_meta.json").write_text("{}", encoding="utf-8")

        result = self._save_single_trial_session(subject_id="legacy_subject", session_id="legacy_session")

        self.assertEqual(int(result["save_index"]), 2)
        self.assertIn("_run-002_", Path(result["meta_json_path"]).name)


if __name__ == "__main__":
    unittest.main()
