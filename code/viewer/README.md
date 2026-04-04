# Viewer README

`viewer/` is now a run-bundle viewer for `schema_version=2` collection outputs.
It is no longer limited to `*_mi_epochs.npz`.

## Entry Points

Recommended from project root:

```powershell
python run_04_view_collected_npz.py
```

Equivalent entry:

```powershell
python code/viewer/run_npz_viewer_pycharm.py
```

Direct CLI:

```powershell
python code/viewer/mi_npz_viewer.py --source <session-dir-or-run-artifact>
```

`--npz` is still accepted as a backward-compatible alias, but it now means
"any run-bundle path", not only `*_mi_epochs.npz`.

## What 04 Scans

Default scan target:

- `*_session_meta.json`

You can also load any of these directly:

- session directory
- `*_session_meta.json`
- `*_mi_epochs.npz`
- `*_gate_epochs.npz`
- `*_artifact_epochs.npz`
- `*_continuous.npz`
- `*_events.csv`
- `*_trials.csv`
- `*_segments.csv`
- `*_board_data.npy`

The viewer resolves the matching run stem and then loads the whole bundle.

## What 04 Reads

For one run bundle, the viewer uses:

- `session_meta.json`
- `quality_report.json`
- `board_map.json`
- `board_data.npy`
- `events.csv`
- `trials.csv`
- `segments.csv`
- `mi_epochs.npz`
- `gate_epochs.npz`
- `artifact_epochs.npz`
- `continuous.npz`

If one derivative file is missing, the viewer still loads the rest and marks
the missing file in the Files tab.

## What 04 Shows

Overview:

- subject / session / save index
- duration, sample count, sampling rate
- board matrix shape and EEG row mapping
- trial / event / segment counts
- MI / gate / artifact / continuous derivative shapes

Tables:

- class stats from `trials.csv`
- segment summary from `segments.csv`
- detailed trials table
- detailed segments table
- continuous prompt table
- event log table
- channel quality / signal stats
- run file presence table

Plots:

- selected trial from continuous `board_data.npy`
- selected segment from continuous `board_data.npy`
- selected continuous prompt from continuous `board_data.npy`
- selected MI epoch from `mi_epochs.npz`
- MI class mean waveform
- MI class PSD

For continuous plots, the viewer can overlay:

- semantic segments
- atomic events

This makes it suitable for checking whether saved labels line up with the
continuous board signal.

## Important Interpretation Rules

- `mi_epochs.npz` still contains accepted imagery epochs only.
- rejected trials are visible through `trials.csv` and `segments.csv`.
- class stats are based on `trials.csv`, so missing / zero-count classes remain visible.
- continuous plots read EEG rows directly from `board_data.npy`, which are stored in microvolts.
- MI epoch plots read `mi_epochs.npz`; if the file stores `volt`, the viewer converts to `uV`.

## Exported Files

Exports are written next to the source run bundle:

- `<run_stem>_viewer_summary.json`
- `<run_stem>_viewer_class_stats.csv`
- `<run_stem>_viewer_segment_summary.csv`
- `<run_stem>_viewer_trials.csv`
- `<run_stem>_viewer_segments.csv`
- `<run_stem>_viewer_prompts.csv`
- `<run_stem>_viewer_channels.csv`
- `<run_stem>_viewer_files.csv`

## Good Uses

- verify trial labels against continuous saved data
- inspect rejected trials
- inspect segment boundaries and source events
- inspect continuous prompts and execution success
- inspect gate negative source composition
- inspect artifact window coverage
- inspect channel quality and board integrity after one run

## Limits

- it is still a post-save inspection tool, not a live monitor
- it does not train or infer
- it does not replace MNE/raw FIF debugging for very low-level analysis
