import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.phase13_patch_training import (
    SolarCellPatchDataset,
    SolarCellFullResolutionDataset,
    PHASE5_THRESHOLD,
    _raw_pairs,
    _threshold_scores,
    _tile_starts,
    predict_tiled,
)


class PatchTrainingTests(unittest.TestCase):
    def test_tile_starts_cover_full_image_without_resizing(self):
        self.assertEqual(_tile_starts(448), [0, 224])
        self.assertEqual(_tile_starts(500), [0, 224, 276])
        with self.assertRaisesRegex(ValueError, "smaller than"):
            _tile_starts(223)

    def test_threshold_sweep_includes_locked_phase5_threshold(self):
        prediction = torch.full((1, 1, 16, 16), 0.2)
        target = torch.zeros_like(prediction)
        target[0, 0, 2:4, 3:5] = 1

        scores = _threshold_scores(prediction, target)

        self.assertIn(PHASE5_THRESHOLD, scores)
        self.assertGreaterEqual(scores[PHASE5_THRESHOLD]["dice"], 0.0)

    def test_patch_dataset_extracts_native_resolution_binary_tiles(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "cell.png"
            mask_path = root / "cell_mask.png"
            pixels = np.zeros((448, 448, 3), dtype=np.uint8)
            pixels[:224, :224] = (255, 0, 0)
            mask = np.zeros((448, 448), dtype=np.uint8)
            mask[10, 12] = 255
            Image.fromarray(pixels).save(image_path)
            Image.fromarray(mask).save(mask_path)

            dataset = SolarCellPatchDataset([(image_path, mask_path)])
            image, target, label = dataset[0]
            self.assertEqual(len(dataset), 4)
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertEqual(tuple(target.shape), (1, 224, 224))
            self.assertEqual(float(target.sum()), 1.0)
            self.assertEqual(float(label), 1.0)
            self.assertEqual(float(dataset[1][1].sum()), 0.0)

    def test_raw_pairing_matches_common_mask_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_images = root / "data" / "SolarCells" / "train" / "defect"
            raw_masks = root / "data" / "SolarCells" / "train" / "label"
            raw_images.mkdir(parents=True)
            raw_masks.mkdir(parents=True)
            image_path = raw_images / "cell.jpg"
            mask_path = raw_masks / "cell_mask.jpg"
            Image.new("RGB", (448, 448)).save(image_path)
            Image.new("L", (448, 448)).save(mask_path)

            class ProcessedDataset:
                samples = [(Path("cell.png"), Path("cell.png"))]

                def __len__(self):
                    return 1

            self.assertEqual(
                _raw_pairs(root, ProcessedDataset()),
                [(image_path, mask_path)],
            )

    def test_rotating_sampler_visits_each_tile_once_per_four_epochs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "cell.png"
            mask_path = root / "cell_mask.png"
            pixels = np.zeros((448, 448, 3), dtype=np.uint8)
            mask = np.zeros((448, 448), dtype=np.uint8)
            for tile_index, (top, left) in enumerate(
                ((0, 0), (0, 224), (224, 0), (224, 224))
            ):
                pixels[top : top + 224, left : left + 224] = (
                    40 + tile_index * 40,
                    0,
                    0,
                )
                mask[top + 10, left + 12] = 255
            Image.fromarray(pixels).save(image_path)
            Image.fromarray(mask).save(mask_path)

            dataset = SolarCellPatchDataset(
                [(image_path, mask_path)],
                one_patch_per_image=True,
                seed=1,
            )
            observed_red_values = set()
            for epoch in range(4):
                dataset.set_epoch(epoch)
                image, target, _ = dataset[0]
                observed_red_values.add(round(float(image[0, 0, 0]), 5))
                self.assertEqual(float(target.sum()), 1.0)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(observed_red_values), 4)

    def test_tiled_prediction_restores_full_image_shape_and_alignment(self):
        class FirstChannel(nn.Module):
            def forward(self, inputs):
                return inputs[:, :1]

        image = torch.zeros((3, 448, 448), dtype=torch.float32)
        image[0, :224, :224] = 0.1
        image[0, :224, 224:] = 0.2
        image[0, 224:, :224] = 0.3
        image[0, 224:, 224:] = 0.4
        prediction = predict_tiled(FirstChannel(), image, tile_batch_size=3)

        self.assertEqual(tuple(prediction.shape), (1, 1, 448, 448))
        self.assertTrue(torch.all(prediction[0, 0, :224, :224] == 0.1))
        self.assertTrue(torch.all(prediction[0, 0, :224, 224:] == 0.2))
        self.assertTrue(torch.all(prediction[0, 0, 224:, :224] == 0.3))
        self.assertTrue(torch.all(prediction[0, 0, 224:, 224:] == 0.4))

    def test_full_resolution_validation_dataset_keeps_original_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "cell.png"
            mask_path = root / "cell_mask.png"
            Image.new("RGB", (448, 448), (10, 20, 30)).save(image_path)
            mask = np.zeros((448, 448), dtype=np.uint8)
            mask[5:7, 9:11] = 255
            Image.fromarray(mask).save(mask_path)

            image, target, name = SolarCellFullResolutionDataset(
                [(image_path, mask_path)]
            )[0]
            self.assertEqual(tuple(image.shape), (3, 448, 448))
            self.assertEqual(tuple(target.shape), (1, 448, 448))
            self.assertEqual(float(target.sum()), 4.0)
            self.assertEqual(name, "cell.png")


if __name__ == "__main__":
    unittest.main()
