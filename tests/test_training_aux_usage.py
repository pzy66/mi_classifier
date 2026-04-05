import sys
from pathlib import Path
from unittest import TestCase

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "code" / "training"
SHARED_ROOT = PROJECT_ROOT / "code" / "shared"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from train_custom_dataset import (
    build_artifact_branch_dataset,
    build_gate_branch_dataset,
    compute_selection_objective_score,
    split_continuous_records_by_group_sets,
)


class TrainingAuxUsageTests(TestCase):
    def test_gate_branch_keeps_long_negatives_without_global_shortest_truncation(self) -> None:
        gate_records = [
            {
                "group": "run_a",
                "session": "ses_a",
                "X_pos": np.ones((2, 8, 1000), dtype=np.float32),
                "X_neg": np.ones((1, 8, 500), dtype=np.float32),
                "X_hard_neg": np.ones((1, 8, 920), dtype=np.float32),
                "neg_sources": np.asarray(["baseline"], dtype=object),
                "hard_neg_sources": np.asarray(["artifact"], dtype=object),
            },
            {
                "group": "run_b",
                "session": "ses_b",
                "X_pos": np.ones((1, 8, 1000), dtype=np.float32),
                "X_neg": np.ones((2, 8, 960), dtype=np.float32),
                "X_hard_neg": np.empty((0, 8, 0), dtype=np.float32),
                "neg_sources": np.asarray(["continuous_no_control", "idle_block"], dtype=object),
                "hard_neg_sources": np.asarray([], dtype=object),
            },
        ]

        result = build_gate_branch_dataset(gate_records, required_samples=800, channel_count=8)

        self.assertEqual(result["X_pos"].shape, (3, 8, 800))
        self.assertEqual(result["X_neg"].shape, (2, 8, 800))
        self.assertEqual(result["X_hard_neg"].shape, (1, 8, 800))
        self.assertListEqual(result["neg_sources"].tolist(), ["continuous_no_control", "idle_block"])

    def test_artifact_branch_filters_per_record_instead_of_inheriting_short_gate_run(self) -> None:
        artifact_records = [
            {
                "group": "run_art",
                "session": "ses_art",
                "X_artifact": np.ones((2, 8, 920), dtype=np.float32),
                "artifact_labels": np.asarray(["blink", "jaw"], dtype=object),
            }
        ]
        gate_records = [
            {
                "group": "run_short",
                "session": "ses_short",
                "X_pos": np.ones((1, 8, 1000), dtype=np.float32),
                "X_neg": np.ones((1, 8, 500), dtype=np.float32),
                "X_hard_neg": np.empty((0, 8, 0), dtype=np.float32),
                "neg_sources": np.asarray(["baseline"], dtype=object),
                "hard_neg_sources": np.asarray([], dtype=object),
            },
            {
                "group": "run_long",
                "session": "ses_long",
                "X_pos": np.ones((1, 8, 1000), dtype=np.float32),
                "X_neg": np.ones((1, 8, 940), dtype=np.float32),
                "X_hard_neg": np.empty((0, 8, 0), dtype=np.float32),
                "neg_sources": np.asarray(["continuous_no_control"], dtype=object),
                "hard_neg_sources": np.asarray([], dtype=object),
            },
        ]

        result = build_artifact_branch_dataset(
            artifact_records,
            gate_records,
            required_samples=900,
            channel_count=8,
        )

        self.assertEqual(result["X_artifact"].shape, (2, 8, 900))
        self.assertEqual(result["X_clean_negative"].shape, (3, 8, 900))

    def test_continuous_records_follow_group_then_session_assignment(self) -> None:
        records = [
            {"run_stem": "run_train", "session_id": "ses_train"},
            {"run_stem": "run_val", "session_id": "ses_val"},
            {"run_stem": "", "session_id": "ses_test"},
        ]

        result = split_continuous_records_by_group_sets(
            records,
            train_groups={"run_train"},
            val_groups={"run_val"},
            test_groups={"run_test"},
            train_sessions={"ses_train"},
            val_sessions={"ses_val"},
            test_sessions={"ses_test"},
        )

        self.assertEqual(len(result["train"]), 1)
        self.assertEqual(len(result["val"]), 1)
        self.assertEqual(len(result["test"]), 1)
        self.assertEqual(result["assignments"][0]["matched_by"], "group")
        self.assertEqual(result["assignments"][2]["matched_by"], "session")

    def test_selection_objective_uses_selection_metrics_not_bank_test_metrics(self) -> None:
        objective = compute_selection_objective_score(
            metrics={
                "selection_val_kappa": 0.8,
                "selection_val_macro_f1": 0.7,
                "bank_kappa": 0.1,
                "bank_macro_f1": 0.1,
            },
            continuous_summary={"available": False},
        )

        self.assertAlmostEqual(objective["score"], 0.76, places=6)
        self.assertAlmostEqual(objective["components"]["selection_val_kappa"], 0.8, places=6)
        self.assertAlmostEqual(objective["components"]["selection_val_macro_f1"], 0.7, places=6)
