import unittest

import numpy as np
import torch

from src.phase10_model_error_analysis import (
    _binary_metrics,
    _ground_truth_components,
)


class ModelErrorAnalysisTests(unittest.TestCase):
    def test_components_distinguish_missed_from_partial_segmentation(self):
        target = np.zeros((1, 1, 20, 20), dtype=np.float32)
        target[0, 0, 2:4, 2:4] = 1
        target[0, 0, 10:12, 10:13] = 1
        predictions = {
            "Phase 5": np.zeros((1, 20, 20), dtype=bool),
            "Extended best": np.zeros((1, 20, 20), dtype=bool),
        }
        predictions["Phase 5"][0, 2:3, 2:4] = True
        predictions["Extended best"][0, 10, 10:12] = True

        rows = _ground_truth_components(
            torch.from_numpy(target),
            predictions,
            ["example.png"],
            {"Phase 5": 0.165, "Extended best": 0.285},
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [row["Phase 5_component_status"] for row in rows],
            ["partial", "missed"],
        )
        self.assertEqual(
            [row["Extended best_component_status"] for row in rows],
            ["missed", "partial"],
        )

    def test_tiny_recall_uses_macro_average_over_tiny_images(self):
        targets = torch.zeros((2, 1, 20, 20))
        targets[0, 0, 1:3, 1:3] = 1
        targets[1, 0, 10:12, 10:12] = 1
        predictions = torch.zeros_like(targets)
        predictions[0, 0, 1:3, 1:3] = 0.9
        predictions[1, 0, 10, 10] = 0.9

        metrics = _binary_metrics(predictions, targets, threshold=0.5)

        self.assertAlmostEqual(metrics["tiny_defect_dice"], 0.7)
        self.assertAlmostEqual(metrics["tiny_defect_recall"], 0.625)


if __name__ == "__main__":
    unittest.main()
