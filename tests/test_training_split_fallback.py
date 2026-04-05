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

from train_custom_dataset import _split_train_val_with_groups


class TrainingSplitFallbackTests(TestCase):
    def test_group_inner_fallback_returns_none_when_stratify_is_impossible(self) -> None:
        y = np.asarray([0, 1, 2, 3], dtype=np.int64)
        groups = np.asarray(["run_a", "run_b", "run_c", "run_d"], dtype=object)
        trainval_idx = np.arange(y.shape[0], dtype=np.int64)

        result = _split_train_val_with_groups(
            y,
            groups,
            trainval_idx,
            random_state=42,
            class_count=4,
            max_group_attempts=2,
        )

        self.assertIsNone(result)
