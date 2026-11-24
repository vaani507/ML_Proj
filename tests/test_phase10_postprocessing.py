import unittest

import numpy as np

from src.phase10_postprocessing import _apply_variant


class Phase10PostProcessingTests(unittest.TestCase):
    def test_area_filter_preserves_a_small_but_thin_connected_crack(self):
        probability = np.zeros((32, 32), dtype=np.float32)
        probability[4:24, 8] = 0.9
        gray = np.full((32, 32), 128, dtype=np.uint8)

        result = _apply_variant(
            probability,
            gray,
            threshold=0.5,
            strategy="min_area_16",
        )

        self.assertEqual(int(result.sum()), 20)
        self.assertTrue(result[4:24, 8].all())

    def test_compact_shape_filter_does_not_remove_elongated_microcrack(self):
        probability = np.zeros((32, 32), dtype=np.float32)
        probability[4:24, 8] = 0.9
        probability[10:13, 18:21] = 0.9
        gray = np.full((32, 32), 128, dtype=np.uint8)

        result = _apply_variant(
            probability,
            gray,
            threshold=0.5,
            strategy="compact_area32_aspect2",
        )

        self.assertEqual(int(result.sum()), 20)
        self.assertTrue(result[4:24, 8].all())
        self.assertFalse(result[10:13, 18:21].any())

    def test_threshold_only_matches_strict_probability_thresholding(self):
        probability = np.array([[0.5, 0.5001], [0.1, 0.9]], dtype=np.float32)
        gray = np.full((2, 2), 128, dtype=np.uint8)

        result = _apply_variant(
            probability,
            gray,
            threshold=0.5,
            strategy="none",
        )

        np.testing.assert_array_equal(
            result,
            np.array([[False, True], [False, True]]),
        )


if __name__ == "__main__":
    unittest.main()
