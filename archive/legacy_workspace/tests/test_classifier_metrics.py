#!/usr/bin/env python3
"""Tests for threshold tuning guardrails in classifier metrics."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.classifier_metrics import (
    evaluate_threshold_candidates,
    select_best_threshold_metrics,
    tune_closed_threshold,
)


class ClassifierMetricTests(unittest.TestCase):
    """Check that threshold tuning avoids collapsed all-closed solutions."""

    def setUp(self) -> None:
        # Mirrors the real collapse seen in the report artifacts:
        # - threshold 0.3 predicts every sample as closed
        # - threshold 0.4 predicts almost every sample as closed
        # - threshold 0.5 is the first reasonably balanced operating point
        negatives = [0.35] * 1 + [0.45] * 88 + [0.55] * 46
        positives = [0.35] * 1 + [0.45] * 51 + [0.55] * 83
        self.targets = np.array([0] * len(negatives) + [1] * len(positives), dtype=np.int64)
        self.closed_probabilities = np.array(negatives + positives, dtype=np.float32)
        self.thresholds = [0.3, 0.4, 0.5, 0.6]

    def test_raw_f1_selection_prefers_degenerate_threshold_without_guardrail(self) -> None:
        rows = evaluate_threshold_candidates(
            targets=self.targets,
            closed_probabilities=self.closed_probabilities,
            thresholds=self.thresholds,
        )
        best = max(rows, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"])))

        self.assertEqual(float(best["threshold"]), 0.3)
        self.assertEqual(float(best["predicted_positive_rate"]), 1.0)

    def test_guarded_selection_prefers_balanced_threshold(self) -> None:
        rows = evaluate_threshold_candidates(
            targets=self.targets,
            closed_probabilities=self.closed_probabilities,
            thresholds=self.thresholds,
        )
        best = select_best_threshold_metrics(
            rows,
            objective="f1",
            min_positive_rate=0.05,
            max_positive_rate=0.95,
        )

        self.assertEqual(float(best["threshold"]), 0.5)
        self.assertGreater(float(best["accuracy"]), 0.6)
        self.assertLess(float(best["predicted_positive_rate"]), 0.95)

    def test_tune_closed_threshold_uses_guardrail(self) -> None:
        best_threshold, best_metrics = tune_closed_threshold(
            targets=self.targets,
            closed_probabilities=self.closed_probabilities,
            thresholds=self.thresholds,
            objective="f1",
            min_positive_rate=0.05,
            max_positive_rate=0.95,
        )

        self.assertEqual(best_threshold, 0.5)
        self.assertEqual(float(best_metrics["threshold"]), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
