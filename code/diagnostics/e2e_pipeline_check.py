# -*- coding: utf-8 -*-
"""End-to-end pipeline verification for MI project.

Flow:
1) Simulate collection and save two runs into custom_mi-like structure.
2) Verify viewer can discover/load the saved npz files.
3) Train model from those runs.
4) Run one offline realtime prediction window with the trained artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "e2e_pipeline_check"
DATASET_ROOT = RUNTIME_ROOT / "datasets" / "custom_mi"
MODEL_PATH = RUNTIME_ROOT / "models" / "e2e_realtime_model.joblib"
REPORT_PATH = RUNTIME_ROOT / "reports" / "e2e_training_summary.json"
SUMMARY_PATH = RUNTIME_ROOT / "e2e_check_summary.json"


def _prepare_env() -> None:
    """Avoid local USERPROFILE/MNE lock issues and force offscreen Qt."""
    user_profile = str(PROJECT_ROOT / "runtime" / "userprofile")
    os.environ["USERPROFILE"] = user_profile
    os.environ["HOME"] = user_profile
    if len(user_profile) >= 2 and user_profile[1] == ":":
        os.environ["HOMEDRIVE"] = user_profile[:2]
        os.environ["HOMEPATH"] = user_profile[2:] or "\\"
    mne_home = str(PROJECT_ROOT / "runtime" / "mne_home")
    os.environ["_MNE_FAKE_HOME_DIR"] = mne_home
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    Path(user_profile).mkdir(parents=True, exist_ok=True)
    Path(mne_home).mkdir(parents=True, exist_ok=True)


_prepare_env()

SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
TRAINING_ROOT = PROJECT_ROOT / "code" / "training"
VIEWER_ROOT = PROJECT_ROOT / "code" / "viewer"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(VIEWER_ROOT) not in sys.path:
    sys.path.insert(0, str(VIEWER_ROOT))

from src.mi_collection import MI_CLASSES, SessionSettings, TrialRecord, make_event, save_mi_session
from src.realtime_mi import RealtimeMIPredictor, load_realtime_model
from train_custom_dataset import train_custom_model
from mi_npz_viewer import discover_epoch_files, load_epochs_npz


@dataclass
class SimulatedRun:
    """Synthetic run generation config."""

    trials_per_class: int
    bad_trial_ids: set[int]
    seed: int


def _class_order(tpc: int) -> list[str]:
    classes = [item["key"] for item in MI_CLASSES]
    sequence: list[str] = []
    for _ in range(int(tpc)):
        sequence.extend(classes)
    return sequence


def _build_synthetic_capture(run_cfg: SimulatedRun) -> tuple[np.ndarray, list[dict], list[TrialRecord]]:
    """Create synthetic brainflow_data + event log + trial records."""
    rng = np.random.default_rng(run_cfg.seed)
    sequence = _class_order(run_cfg.trials_per_class)

    # Keep synthetic timing close to collection defaults so default training windows
    # (2.0/2.5/3.0s with offset search) remain valid.
    baseline = 500   # 2.0s @ 250Hz
    cue = 250        # 1.0s @ 250Hz
    imagery = 1000   # 4.0s @ 250Hz
    iti = 625        # 2.5s @ 250Hz
    gap = 16

    cursor = 8
    events: list[dict] = []
    trials: list[TrialRecord] = []
    markers: dict[int, int] = {}

    def add_event(name: str, sample: int, *, trial_id: int | None = None, class_name: str | None = None) -> None:
        if sample in markers:
            raise RuntimeError(f"marker sample collision at {sample}")
        event = make_event(name, trial_id=trial_id, class_name=class_name)
        events.append(event)
        markers[sample] = int(event["marker_code"])

    add_event("session_start", 0)
    for trial_idx, class_name in enumerate(sequence, start=1):
        trial = TrialRecord(
            trial_id=trial_idx,
            class_name=class_name,
            display_name=str(class_name),
            accepted=trial_idx not in run_cfg.bad_trial_ids,
        )
        if not trial.accepted:
            trial.note = "e2e synthetic bad trial"
        trials.append(trial)

        trial_start = cursor
        baseline_start = trial_start + 1
        baseline_end = baseline_start + baseline
        cue_start = baseline_end + 1
        imagery_start = cue_start + cue
        imagery_end = imagery_start + imagery
        iti_start = imagery_end + 1
        trial_end = iti_start + iti

        add_event("trial_start", trial_start, trial_id=trial_idx, class_name=class_name)
        add_event("baseline_start", baseline_start, trial_id=trial_idx, class_name=class_name)
        add_event("baseline_end", baseline_end, trial_id=trial_idx, class_name=class_name)
        add_event(f"cue_{class_name}", cue_start, trial_id=trial_idx, class_name=class_name)
        add_event(f"imagery_{class_name}", imagery_start, trial_id=trial_idx, class_name=class_name)
        add_event("imagery_end", imagery_end, trial_id=trial_idx, class_name=class_name)
        add_event("iti_start", iti_start, trial_id=trial_idx, class_name=class_name)
        add_event("trial_end", trial_end, trial_id=trial_idx, class_name=class_name)
        if trial_idx in run_cfg.bad_trial_ids:
            add_event("bad_trial_marked", trial_end + 1, trial_id=trial_idx, class_name=class_name)

        cursor = trial_end + gap

    session_end = cursor + 5
    add_event("session_end", session_end)

    n_samples = session_end + 5
    eeg_uV = (rng.normal(loc=0.0, scale=12.0, size=(8, n_samples))).astype(np.float32)
    data = np.zeros((12, n_samples), dtype=np.float32)
    data[0:8, :] = eeg_uV
    for sample, code in markers.items():
        data[10, sample] = float(code)
    data[11, :] = np.linspace(0.0, float(n_samples / 250.0), n_samples, dtype=np.float32)

    return data, events, trials


def _save_one_run(*, session_id: str, run_cfg: SimulatedRun) -> dict:
    """Call save_mi_session with synthetic payload."""
    data, events, trials = _build_synthetic_capture(run_cfg)
    settings = SessionSettings(
        subject_id="001",
        session_id=session_id,
        output_root=str(DATASET_ROOT),
        board_id=0,
        serial_port="COM3",
        channel_names=["C3", "Cz", "C4", "PO3", "PO4", "O1", "Oz", "O2"],
        channel_positions=[0, 1, 2, 3, 4, 5, 6, 7],
        trials_per_class=int(run_cfg.trials_per_class),
        baseline_sec=2.0,
        cue_sec=1.0,
        imagery_sec=4.0,
        iti_sec=2.5,
        random_seed=42,
        save_epochs_npz=True,
        operator="e2e-check",
        notes="synthetic pipeline check",
        board_name="Synthetic",
    )
    return save_mi_session(
        brainflow_data=data,
        sampling_rate=250.0,
        eeg_rows=[0, 1, 2, 3, 4, 5, 6, 7],
        marker_row=10,
        timestamp_row=11,
        settings=settings,
        event_log=events,
        trial_records=trials,
    )


def run_e2e_check() -> dict:
    """Execute full pipeline verification."""
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Collection/save simulation (two runs, different trial counts).
    run1 = _save_one_run(session_id="e2e_session", run_cfg=SimulatedRun(trials_per_class=4, bad_trial_ids=set(), seed=7))
    run2 = _save_one_run(session_id="e2e_session", run_cfg=SimulatedRun(trials_per_class=2, bad_trial_ids={1}, seed=13))

    session_dir = Path(run1["session_dir"])
    manifest_path = Path(run2["manifest_csv_path"])
    epoch_files = sorted(session_dir.glob("*_epochs.npz"))

    # 2) Viewer stage checks (data discovery + file parsing; skip GUI launch for headless stability).
    discovered = discover_epoch_files(DATASET_ROOT)
    loaded_preview = [load_epochs_npz(path) for path in discovered]

    # 3) Training stage checks.
    summary = train_custom_model(
        dataset_root=DATASET_ROOT,
        subject_filter="sub-001",
        output_model_path=MODEL_PATH,
        report_path=REPORT_PATH,
        random_state=42,
        min_class_trials=2,
    )

    # 4) Realtime offline-prediction checks.
    artifact = load_realtime_model(MODEL_PATH)
    predictor = RealtimeMIPredictor(artifact, history_len=3, confidence_threshold=0.0)
    first_epoch = loaded_preview[0]
    # viewer loads uV; training uses volt in npz, so convert back to volt here.
    test_window_volt = (first_epoch.X_uV[0] * 1e-6).astype(np.float32)
    pred = predictor.analyze_window(test_window_volt, live_sampling_rate=float(first_epoch.sampling_rate))

    metrics = dict(summary.get("metrics") or {})
    pipeline_tag = str(summary.get("artifact_type") or summary.get("selected_pipeline") or "unknown")

    result = {
        "status": "pass",
        "paths": {
            "runtime_root": str(RUNTIME_ROOT),
            "dataset_root": str(DATASET_ROOT),
            "session_dir": str(session_dir),
            "manifest_csv": str(manifest_path),
            "model_path": str(MODEL_PATH),
            "report_path": str(REPORT_PATH),
        },
        "collection": {
            "run1_stem": run1["run_stem"],
            "run2_stem": run2["run_stem"],
            "run1_trial_count": int(run1["trial_count"]),
            "run2_trial_count": int(run2["trial_count"]),
            "run1_accepted": int(run1["accepted_trial_count"]),
            "run2_accepted": int(run2["accepted_trial_count"]),
            "epoch_file_count_in_session": len(epoch_files),
        },
        "viewer": {
            "discovered_npz_count": len(discovered),
            "first_npz_shape": list(loaded_preview[0].X_uV.shape) if loaded_preview else [],
            "first_npz_sampling_rate": float(loaded_preview[0].sampling_rate) if loaded_preview else None,
        },
        "training": {
            "pipeline_tag": pipeline_tag,
            "total_trials_used": int(summary["total_trials"]),
            "source_records_count": int(len(summary["source_records"])),
            "test_acc": float(metrics.get("bank_test_acc", metrics.get("test_acc", 0.0))),
            "kappa": float(metrics.get("bank_kappa", metrics.get("kappa", 0.0))),
        },
        "realtime": {
            "expected_channels": int(predictor.expected_channel_count),
            "prediction_name": str(pred["prediction_name"]),
            "prediction_display_name": str(pred["prediction_display_name"]),
            "confidence": float(pred["confidence"]),
            "stable_prediction_name": pred["stable_prediction_name"],
        },
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def main() -> int:
    result = run_e2e_check()
    print("E2E check status:", result["status"])
    print("Summary path:", SUMMARY_PATH)
    print("Training pipeline tag:", result["training"]["pipeline_tag"])
    print("Realtime prediction:", result["realtime"]["prediction_display_name"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
