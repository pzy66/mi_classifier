import csv
import json
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
TRAINING_ROOT = PROJECT_ROOT / "code" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from src.mi_collection import CLASS_NAME_TO_LABEL, SessionSettings, TrialRecord, make_event, save_mi_session
from train_custom_dataset import load_custom_task_datasets


class MICollectionRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = PROJECT_ROOT / "runtime" / f"mi_collection_test_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_single_trial_session(self, *, protocol_mode: str = "full") -> dict[str, str]:
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
            subject_id="regression",
            session_id="single_trial",
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
            protocol_mode=protocol_mode,
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

    def _save_multiphase_session(self) -> dict[str, str]:
        sampling_rate = 250.0
        marker_row = 10
        timestamp_row = 11
        n_channels = 8
        baseline = 1000
        cue = 500
        imagery = 1000
        iti = 1000
        quality_block = 750
        artifact_block = 1250
        rest_block = 1250
        continuous_prompt = 1000
        continuous_gap = 250
        block_gap = 24

        markers: dict[int, int] = {}
        event_log: list[dict[str, object]] = []
        trial_records: list[TrialRecord] = []
        cursor = 0

        def add_event(event_name: str, sample_index: int, **extra: object) -> None:
            self.assertNotIn(sample_index, markers)
            event = make_event(event_name, **extra)
            event_log.append(event)
            markers[sample_index] = int(event["marker_code"])

        add_event("session_start", cursor)
        cursor += 8

        add_event("quality_check_start", cursor)
        cursor += quality_block
        add_event("quality_check_end", cursor)
        cursor += block_gap

        add_event("calibration_start", cursor)
        add_event("eyes_open_rest_start", cursor + 1)
        add_event("eyes_open_rest_end", cursor + 1 + rest_block)
        offset = cursor + 1 + rest_block + block_gap
        add_event("eyes_closed_rest_start", offset)
        add_event("eyes_closed_rest_end", offset + rest_block)
        offset += rest_block + block_gap
        for artifact_name in ["eye_movement", "blink", "swallow", "jaw", "head_motion"]:
            add_event(f"{artifact_name}_block_start", offset)
            add_event(f"{artifact_name}_block_end", offset + artifact_block)
            offset += artifact_block + block_gap
        add_event("calibration_end", offset)
        cursor = offset + block_gap

        add_event("practice_start", cursor)
        cursor += rest_block
        add_event("practice_end", cursor)
        cursor += block_gap

        run_index = 1
        add_event("mi_run_start", cursor, run_index=run_index)
        cursor += block_gap
        class_order = ["left_hand", "right_hand", "feet", "tongue"]
        for trial_id, class_name in enumerate(class_order, start=1):
            run_trial_index = trial_id
            trial_start = cursor
            fixation_start = trial_start + 1
            baseline_start = fixation_start + 1
            baseline_end = baseline_start + baseline
            cue_start = baseline_end + 1
            cue_specific = cue_start + 1
            imagery_start = cue_specific + cue
            imagery_specific = imagery_start + 1
            imagery_end = imagery_start + imagery
            iti_start = imagery_end + 1
            trial_end = iti_start + iti

            add_event("trial_start", trial_start, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("fixation_start", fixation_start, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("baseline_start", baseline_start, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("baseline_end", baseline_end, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("cue_start", cue_start, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event(f"cue_{class_name}", cue_specific, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("imagery_start", imagery_start, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event(f"imagery_{class_name}", imagery_specific, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("imagery_end", imagery_end, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("iti_start", iti_start, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            add_event("trial_end", trial_end, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            accepted = trial_id != 3
            if not accepted:
                add_event("bad_trial_marked", trial_end + 1, trial_id=trial_id, class_name=class_name, run_index=run_index, run_trial_index=run_trial_index)
            trial_records.append(
                TrialRecord(
                    trial_id=trial_id,
                    class_name=class_name,
                    display_name=class_name,
                    run_index=run_index,
                    run_trial_index=run_trial_index,
                    accepted=accepted,
                    note="" if accepted else "synthetic rejected trial",
                )
            )
            cursor = trial_end + block_gap
        add_event("mi_run_end", cursor, run_index=run_index)
        cursor += block_gap
        add_event("run_rest_start", cursor, run_index=run_index)
        cursor += rest_block
        add_event("run_rest_end", cursor, run_index=run_index)
        cursor += block_gap

        add_event("continuous_block_start", cursor, block_index=1)
        cursor += 1
        for label, success, prompt_index in [("left_hand", 1, 1), ("no_control", 0, 2)]:
            add_event(
                f"continuous_command_{label}",
                cursor,
                class_name=label,
                block_index=1,
                prompt_index=prompt_index,
                command_duration_sec=4.0,
                execution_success=success,
            )
            cursor += continuous_prompt
            add_event(
                "continuous_command_end",
                cursor,
                class_name=label,
                block_index=1,
                prompt_index=prompt_index,
                command_duration_sec=4.0,
                execution_success=success,
            )
            cursor += continuous_gap
        add_event("continuous_block_end", cursor, block_index=1)
        cursor += block_gap

        add_event("idle_block_start", cursor, block_index=1)
        cursor += rest_block
        add_event("idle_block_end", cursor, block_index=1)
        cursor += block_gap

        add_event("idle_prepare_start", cursor, block_index=1)
        cursor += rest_block
        add_event("idle_prepare_end", cursor, block_index=1)
        cursor += block_gap

        add_event("session_end", cursor)

        n_samples = cursor + 8
        rng = np.random.default_rng(20260403)
        brainflow_data = np.zeros((12, n_samples), dtype=np.float32)
        time_axis = np.linspace(0.0, n_samples / sampling_rate, n_samples, dtype=np.float32)
        for channel_index in range(n_channels):
            brainflow_data[channel_index] = (
                15.0 * np.sin(2.0 * np.pi * (0.8 + 0.2 * channel_index) * time_axis)
                + 3.0 * np.cos(2.0 * np.pi * (0.15 + 0.05 * channel_index) * time_axis)
                + rng.normal(0.0, 0.8, size=n_samples)
            ).astype(np.float32)
        for sample_index, marker_code in markers.items():
            brainflow_data[marker_row, sample_index] = float(marker_code)
        brainflow_data[timestamp_row] = np.linspace(1000.0, 1000.0 + n_samples / sampling_rate, n_samples, dtype=np.float32)

        settings = SessionSettings(
            subject_id="regression",
            session_id="multiphase",
            output_root=str(self.temp_dir),
            board_id=0,
            serial_port="COM3",
            channel_names=["C3", "Cz", "C4", "CP3", "CP4", "O1", "Oz", "O2"],
            channel_positions=list(range(8)),
            trials_per_class=1,
            baseline_sec=4.0,
            cue_sec=2.0,
            imagery_sec=4.0,
            iti_sec=4.0,
            random_seed=20260403,
            run_count=1,
            include_eyes_closed_rest_in_gate_neg=True,
            artifact_types=["eye_movement", "blink", "swallow", "jaw", "head_motion"],
        )
        return save_mi_session(
            brainflow_data=brainflow_data,
            sampling_rate=sampling_rate,
            eeg_rows=list(range(8)),
            marker_row=marker_row,
            timestamp_row=timestamp_row,
            settings=settings,
            event_log=event_log,
            trial_records=trial_records,
        )

    def test_trial_onsets_keep_earliest_generic_marker(self) -> None:
        result = self._save_single_trial_session()

        with Path(result["trials_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            first_row = next(csv.DictReader(handle))

        self.assertEqual(int(first_row["cue_onset_sample"]), 512)
        self.assertEqual(int(first_row["imagery_onset_sample"]), 1012)

        with np.load(result["mi_epochs_path"], allow_pickle=False) as data:
            self.assertEqual(tuple(data["X_mi"].shape), (1, 8, 1000))

    def test_gate_negatives_include_short_baseline_and_iti(self) -> None:
        result = self._save_single_trial_session()

        with np.load(result["gate_epochs_path"], allow_pickle=False) as data:
            gate_neg = np.asarray(data["X_gate_neg"], dtype=np.float32)
            gate_neg_sources = [str(item) for item in np.asarray(data["gate_neg_sources"], dtype=object).tolist()]

        self.assertEqual(gate_neg.shape[1:], (8, 500))
        self.assertEqual(gate_neg.shape[0], 2)
        self.assertEqual(gate_neg_sources, ["baseline", "iti"])

    def test_v2_schema_outputs_board_segments_and_relative_manifest(self) -> None:
        result = self._save_single_trial_session()

        board_data_path = Path(result["board_data_path"])
        board_map_path = Path(result["board_map_path"])
        segments_csv_path = Path(result["segments_csv_path"])
        meta_json_path = Path(result["meta_json_path"])
        manifest_path = Path(result["manifest_csv_path"])

        self.assertTrue(board_data_path.exists())
        self.assertTrue(board_map_path.exists())
        self.assertTrue(segments_csv_path.exists())

        board_data = np.load(board_data_path, allow_pickle=False)
        self.assertEqual(tuple(board_data.shape), (10, 2700))

        with board_map_path.open("r", encoding="utf-8") as handle:
            board_map = json.load(handle)
        self.assertEqual(int(board_map["schema_version"]), 2)
        self.assertEqual(int(board_map["save_index"]), 1)
        self.assertEqual(board_map["selected_eeg_rows"], list(range(8)))
        self.assertEqual(int(board_map["marker_row"]), 8)
        self.assertIsNone(board_map["timestamp_row"])

        with segments_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            segment_rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["segment_type"] for row in segment_rows],
            ["trial", "baseline", "cue", "imagery", "iti"],
        )

        with Path(result["events_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            event_fields = list(csv.DictReader(handle).fieldnames or [])
        self.assertIn("mi_run_index", event_fields)
        self.assertNotIn("run_index", event_fields)

        with Path(result["trials_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            trial_fields = list(csv.DictReader(handle).fieldnames or [])
        self.assertIn("mi_run_index", trial_fields)
        self.assertNotIn("run_index", trial_fields)

        with meta_json_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertEqual(int(meta["schema_version"]), 2)
        self.assertEqual(int(meta["save_index"]), 1)
        self.assertEqual(meta["protocol_mode"], "full")
        self.assertEqual(meta["session"]["protocol_mode"], "full")
        self.assertEqual(meta["raw_preservation_level"], "cropped_full_board_matrix")
        self.assertEqual(meta["session"]["output_root"], ".")
        self.assertIn("board_data_npy", meta["files"])
        self.assertIn("segments_csv", meta["files"])
        self.assertNotIn(":", str(meta["files"]["events_csv"]))
        self.assertTrue(str(meta["files"]["mi_epochs_meta_json"]).endswith("_mi_epochs.meta.json"))

        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0]["schema_version"], "2")
        self.assertEqual(manifest_rows[0]["protocol_mode"], "full")
        self.assertEqual(manifest_rows[0]["save_index"], "1")
        self.assertTrue(manifest_rows[0]["board_data_npy"].endswith("_board_data.npy"))
        self.assertTrue(manifest_rows[0]["segments_csv"].endswith("_segments.csv"))
        self.assertNotIn(":", manifest_rows[0]["session_meta_json"])
        self.assertNotIn("epochs_npz", manifest_rows[0])

        with np.load(result["mi_epochs_path"], allow_pickle=False) as data:
            self.assertEqual(int(np.asarray(data["schema_version"]).reshape(-1)[0]), 2)
            self.assertEqual(int(np.asarray(data["save_index"]).reshape(-1)[0]), 1)
            self.assertEqual(str(np.asarray(data["protocol_mode"]).reshape(-1)[0]), "full")
            self.assertNotIn("X", data.files)
            self.assertNotIn("y", data.files)
            self.assertNotIn("run_index", data.files)
        with np.load(result["gate_epochs_path"], allow_pickle=False) as data:
            self.assertEqual(str(np.asarray(data["protocol_mode"]).reshape(-1)[0]), "full")
            self.assertNotEqual(np.asarray(data["gate_neg_sources"]).dtype.kind, "O")
            self.assertNotIn("run_index", data.files)
        with np.load(result["continuous_path"], allow_pickle=False) as data:
            self.assertEqual(str(np.asarray(data["protocol_mode"]).reshape(-1)[0]), "full")
            self.assertIn("continuous_block_indices", data.files)
            self.assertIn("continuous_prompt_indices", data.files)
            self.assertIn("continuous_execution_success", data.files)
            self.assertNotIn("continuous_events", data.files)

    def test_protocol_mode_is_saved_for_mi_only_sessions(self) -> None:
        result = self._save_single_trial_session(protocol_mode="mi_only")

        with Path(result["meta_json_path"]).open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        self.assertEqual(meta["protocol_mode"], "mi_only")
        self.assertEqual(meta["session"]["protocol_mode"], "mi_only")

        with Path(result["manifest_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        self.assertEqual(manifest_rows[0]["protocol_mode"], "mi_only")

        with np.load(result["mi_epochs_path"], allow_pickle=False) as data:
            self.assertEqual(str(np.asarray(data["protocol_mode"]).reshape(-1)[0]), "mi_only")

    def test_segments_keep_actual_source_event_names(self) -> None:
        result = self._save_single_trial_session()

        with Path(result["segments_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        by_type = {row["segment_type"]: row for row in rows}
        self.assertEqual(by_type["baseline"]["start_sample"], "11")
        self.assertEqual(by_type["baseline"]["end_sample"], "511")
        self.assertEqual(by_type["baseline"]["source_start_event"], "fixation_start")
        self.assertEqual(by_type["baseline"]["source_end_event"], "baseline_end")
        self.assertEqual(by_type["cue"]["source_start_event"], "cue_start")
        self.assertEqual(by_type["cue"]["source_end_event"], "imagery_start")
        self.assertEqual(by_type["imagery"]["source_start_event"], "imagery_start")
        self.assertEqual(by_type["imagery"]["source_end_event"], "imagery_end")
        self.assertEqual(by_type["iti"]["source_start_event"], "iti_start")
        self.assertEqual(by_type["iti"]["source_end_event"], "trial_end")

    def test_multiphase_save_keeps_labels_and_phase_segments_consistent(self) -> None:
        result = self._save_multiphase_session()

        with Path(result["events_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            event_rows = list(csv.DictReader(handle))
        with Path(result["trials_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            trial_rows = list(csv.DictReader(handle))
        with Path(result["segments_csv_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
            segment_rows = list(csv.DictReader(handle))

        segment_counts: dict[str, int] = {}
        for row in segment_rows:
            segment_counts[row["segment_type"]] = segment_counts.get(row["segment_type"], 0) + 1

        self.assertEqual(len(event_rows), 81)
        self.assertEqual(len(trial_rows), 4)
        self.assertEqual(
            segment_counts,
            {
                "trial": 4,
                "baseline": 4,
                "cue": 4,
                "imagery": 4,
                "iti": 4,
                "quality_check": 1,
                "calibration": 1,
                "practice": 1,
                "run_rest": 1,
                "eyes_open_rest": 1,
                "eyes_closed_rest": 1,
                "continuous_block": 1,
                "continuous_prompt": 2,
                "idle_block": 1,
                "idle_prepare": 1,
                "artifact_block": 5,
            },
        )

        accepted_trial_ids = [int(row["trial_id"]) for row in trial_rows if int(row["accepted"]) == 1]
        self.assertEqual(accepted_trial_ids, [1, 2, 4])

        with np.load(result["mi_epochs_path"], allow_pickle=False) as data:
            self.assertEqual(tuple(data["X_mi"].shape), (3, 8, 1000))
            self.assertEqual(np.asarray(data["mi_trial_ids"], dtype=np.int64).tolist(), [1, 2, 4])
            self.assertEqual(
                np.asarray(data["y_mi"], dtype=np.int64).tolist(),
                [
                    CLASS_NAME_TO_LABEL["left_hand"],
                    CLASS_NAME_TO_LABEL["right_hand"],
                    CLASS_NAME_TO_LABEL["tongue"],
                ],
            )

        with np.load(result["gate_epochs_path"], allow_pickle=False) as data:
            gate_neg_sources = [str(item) for item in np.asarray(data["gate_neg_sources"]).tolist()]
            gate_hard_neg_sources = [str(item) for item in np.asarray(data["gate_hard_neg_sources"]).tolist()]
        self.assertTrue(
            {"baseline", "iti", "eyes_open_rest", "eyes_closed_rest", "idle_block", "idle_prepare", "continuous_no_control"}.issubset(
                set(gate_neg_sources)
            )
        )
        self.assertTrue(
            {"artifact_eye_movement", "artifact_blink", "artifact_swallow", "artifact_jaw", "artifact_head_motion", "rejected_trial"}.issubset(
                set(gate_hard_neg_sources)
            )
        )

        with np.load(result["artifact_epochs_path"], allow_pickle=False) as data:
            artifact_labels = [str(item) for item in np.asarray(data["artifact_labels"]).tolist()]
        self.assertTrue(
            {"eye_movement", "blink", "swallow", "jaw", "head_motion", "rejected_trial"}.issubset(set(artifact_labels))
        )

        with np.load(result["continuous_path"], allow_pickle=False) as data:
            self.assertEqual(tuple(data["X_continuous"].shape), (1, 8, 2501))
            self.assertEqual(np.asarray(data["continuous_event_labels"]).tolist(), ["left_hand", "no_control"])
            self.assertEqual(np.asarray(data["continuous_execution_success"], dtype=np.int8).tolist(), [1, 0])

    def test_training_loader_accepts_v2_schema(self) -> None:
        self._save_single_trial_session()

        loaded = load_custom_task_datasets(self.temp_dir)
        self.assertEqual(len(loaded["source_records"]), 1)
        record = loaded["source_records"][0]
        self.assertEqual(int(record["schema_version"]), 2)
        self.assertEqual(int(record["save_index"]), 1)
        self.assertEqual(str(record["subject_id"]), "regression")
        self.assertEqual(str(record["session_id"]), "single_trial")
        self.assertEqual(str(record["protocol_mode"]), "full")
        self.assertEqual(int(loaded["mi"]["X"].shape[0]), 1)


if __name__ == "__main__":
    unittest.main()
