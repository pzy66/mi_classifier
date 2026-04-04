# Code README

`code/` is split by responsibility to keep collection, training, realtime, and
inspection separated.

## Modules

- `collection/`
  Collection UI, session flow control, marker emission, save triggering.
- `training/`
  Reads task-specific derivatives from `datasets/custom_mi` and trains MI /
  gate / artifact models.
- `realtime/`
  Loads trained artifacts for online inference only.
- `viewer/`
  04 viewer. Loads one full saved run bundle for post-save inspection.
- `shared/`
  Shared schema, preprocessing, model utilities, and save logic.
- `legacy/`
  Archived code, not part of the active pipeline.

## Current Schema Contract

The active codebase targets `schema_version=2`.

Canonical saved run data includes:

- `board_data.npy`
- `board_map.json`
- `events.csv`
- `trials.csv`
- `segments.csv`
- `session_meta.json`
- `quality_report.json`
- `raw.fif`

Task derivatives include:

- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`

Viewer discovery now defaults to `*_session_meta.json`, not only `*_mi_epochs.npz`.

## Recommended Entry Points

From project root:

- `run_01_collection_only.py`
- `run_02_training.py`
- `run_03_realtime_infer.py`
- `run_04_view_collected_npz.py`
- `run_05_channel_monitor.py`

## Debugging Notes

- `run_*_pycharm.py` is GUI-friendly.
- For full tracebacks, run the module entry directly:
  - `code/collection/mi_data_collector.py`
  - `code/training/train_custom_dataset.py`
  - `code/realtime/mi_realtime_infer_only.py`
  - `code/viewer/mi_npz_viewer.py`

## Change Discipline

When schema, save fields, training loaders, or viewer support changes, update:

- `collection/README.md`
- `collection/README_SAVE_NAMING.md`
- `training/README.md`
- `training/README_DATA_LOADING.md`
- `training/README_DATA_ADMISSION.md`
- `viewer/README.md`
- `datasets/custom_mi/README.md`
