import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
VIEWER_ROOT = PROJECT_ROOT / "code" / "viewer"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))
if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))

from src.mi_collection import SessionSettings, TrialRecord, make_event, save_mi_session
from mi_npz_viewer import build_stats_payload, class_rows, discover_run_bundles, load_run_bundle, prompt_rows, segment_summary_rows


class MINpzViewerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = PROJECT_ROOT / "runtime" / f"mi_npz_viewer_test_{uuid.uuid4().hex}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _save_bundle(self) -> dict[str, str]:
        sampling_rate = 250.0
        marker_row = 10
        timestamp_row = 11
        baseline = 1000
        cue = 250
        imagery = 1000
        iti = 1000
        artifact_block = 1250
        idle_block = 1250
        prompt_len = 1000
        gap = 24
        markers: dict[int, int] = {}
        events: list[dict[str, object]] = []
        trials: list[TrialRecord] = []
        cursor = 0

        def add_event(name: str, sample_index: int, **extra: object) -> None:
            self.assertNotIn(sample_index, markers)
            event = make_event(name, **extra)
            events.append(event)
            markers[sample_index] = int(event["marker_code"])

        add_event("session_start", cursor)
        cursor += 8
        add_event("calibration_start", cursor)
        add_event("eye_movement_block_start", cursor + 1)
        add_event("eye_movement_block_end", cursor + 1 + artifact_block)
        add_event("calibration_end", cursor + 1 + artifact_block + gap)
        cursor += artifact_block + gap + 25

        for trial_id, class_name in enumerate(["left_hand", "right_hand"], start=1):
            run_trial_index = trial_id
            trial_start = cursor
            baseline_start = trial_start + 1
            baseline_end = baseline_start + baseline
            cue_start = baseline_end + 1
            imagery_start = cue_start + cue
            imagery_end = imagery_start + imagery
            iti_start = imagery_end + 1
            trial_end = iti_start + iti
            add_event("trial_start", trial_start, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event("baseline_start", baseline_start, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event("baseline_end", baseline_end, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event(f"cue_{class_name}", cue_start, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event(f"imagery_{class_name}", imagery_start, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event("imagery_end", imagery_end, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event("iti_start", iti_start, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            add_event("trial_end", trial_end, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            accepted = trial_id == 1
            if not accepted:
                add_event("bad_trial_marked", trial_end + 1, trial_id=trial_id, class_name=class_name, run_index=1, run_trial_index=run_trial_index)
            trials.append(
                TrialRecord(
                    trial_id=trial_id,
                    class_name=class_name,
                    display_name=class_name,
                    run_index=1,
                    run_trial_index=run_trial_index,
                    accepted=accepted,
                    note="" if accepted else "synthetic reject",
                )
            )
            cursor = trial_end + gap

        add_event("continuous_block_start", cursor, block_index=1)
        cursor += 1
        add_event("continuous_command_no_control", cursor, class_name="no_control", block_index=1, prompt_index=1, command_duration_sec=4.0, execution_success=0)
        cursor += prompt_len
        add_event("continuous_command_end", cursor, class_name="no_control", block_index=1, prompt_index=1, command_duration_sec=4.0, execution_success=0)
        cursor += gap
        add_event("continuous_block_end", cursor, block_index=1)
        cursor += gap

        add_event("idle_block_start", cursor, block_index=1)
        cursor += idle_block
        add_event("idle_block_end", cursor, block_index=1)
        cursor += gap
        add_event("session_end", cursor)

        n_samples = cursor + 8
        data = np.zeros((12, n_samples), dtype=np.float32)
        rng = np.random.default_rng(7)
        time_axis = np.linspace(0.0, n_samples / sampling_rate, n_samples, dtype=np.float32)
        for channel_index in range(8):
            data[channel_index] = (
                8.0 * np.sin(2.0 * np.pi * (0.8 + 0.1 * channel_index) * time_axis)
                + rng.normal(0.0, 0.4, size=n_samples)
            ).astype(np.float32)
        for sample_index, marker_code in markers.items():
            data[marker_row, sample_index] = float(marker_code)
        data[timestamp_row] = np.linspace(1000.0, 1000.0 + n_samples / sampling_rate, n_samples, dtype=np.float32)

        settings = SessionSettings(
            subject_id="viewer",
            session_id="bundle",
            output_root=str(self.temp_dir),
            board_id=0,
            serial_port="COM3",
            channel_names=["C3", "Cz", "C4", "CP3", "CP4", "O1", "Oz", "O2"],
            channel_positions=list(range(8)),
            trials_per_class=1,
            baseline_sec=4.0,
            cue_sec=1.0,
            imagery_sec=4.0,
            iti_sec=4.0,
            random_seed=7,
            run_count=1,
            include_eyes_closed_rest_in_gate_neg=False,
            artifact_types=["eye_movement"],
            board_name="Synthetic",
        )
        return save_mi_session(
            brainflow_data=data,
            sampling_rate=sampling_rate,
            eeg_rows=list(range(8)),
            marker_row=marker_row,
            timestamp_row=timestamp_row,
            settings=settings,
            event_log=events,
            trial_records=trials,
        )

    def test_discover_and_load_run_bundle(self) -> None:
        result = self._save_bundle()

        discovered = discover_run_bundles(self.temp_dir)
        self.assertEqual(len(discovered), 1)
        self.assertTrue(str(discovered[0]).endswith("_session_meta.json"))

        bundle = load_run_bundle(Path(result["mi_epochs_path"]))
        self.assertEqual(bundle.run_stem, result["run_stem"])
        self.assertEqual(bundle.trial_count, 2)
        self.assertEqual(bundle.accepted_trial_count, 1)
        self.assertEqual(bundle.rejected_trial_count, 1)
        self.assertEqual(bundle.board_shape[0], 12)
        self.assertTrue(bundle.gate_summary["available"])
        self.assertTrue(bundle.artifact_summary["available"])
        self.assertTrue(bundle.continuous_summary["available"])

    def test_viewer_stats_rows_match_bundle_contents(self) -> None:
        result = self._save_bundle()
        bundle = load_run_bundle(Path(result["meta_json_path"]))

        payload = build_stats_payload(bundle)
        self.assertEqual(payload["overview"]["trial_count"], 2)
        self.assertEqual(payload["overview"]["accepted_trial_count"], 1)
        self.assertEqual(payload["overview"]["segment_count"], bundle.segment_count)

        cls_rows = class_rows(bundle)
        self.assertEqual([row["class_key"] for row in cls_rows], ["left_hand", "right_hand", "feet", "tongue"])
        self.assertEqual([row["accepted"] for row in cls_rows], [1, 0, 0, 0])
        self.assertEqual([row["total"] for row in cls_rows], [1, 1, 0, 0])

        ch_rows = bundle.board_data[bundle.eeg_rows[0], :].reshape(-1)
        channel_stats = {row["channel"]: row for row in payload["channel_stats"]}
        self.assertAlmostEqual(channel_stats["C3"]["mean_uV"], float(np.mean(ch_rows)), places=4)
        self.assertAlmostEqual(channel_stats["C3"]["abs_mean_uV"], float(np.mean(np.abs(ch_rows))), places=4)
        self.assertGreater(channel_stats["C3"]["abs_mean_uV"], 0.0)

        seg_rows = segment_summary_rows(bundle)
        seg_types = {row["segment_type"] for row in seg_rows}
        self.assertTrue({"trial", "baseline", "cue", "imagery", "iti", "artifact_block", "continuous_block", "continuous_prompt", "idle_block"}.issubset(seg_types))

        prompts = prompt_rows(bundle)
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["label"], "no_control")
        self.assertFalse(bool(prompts[0]["execution_success"]))


if __name__ == "__main__":
    unittest.main()
