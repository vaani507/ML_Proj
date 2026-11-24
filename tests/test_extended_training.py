import unittest

import torch
from torch import nn

from src.train import BoundaryAwareBCEDiceLoss
from src.train_resnet_unet import _scheduler, _validation_scores


class ExtendedTrainingTests(unittest.TestCase):
    def test_scheduler_reduces_learning_rate_after_validation_plateau(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = _scheduler(optimizer)

        for score in (0.8, 0.8, 0.8, 0.8):
            scheduler.step(score)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-4)

    def test_validation_reports_tiny_dice_and_pixel_false_positives(self):
        class ProbabilityFromImage(nn.Module):
            def forward(self, images):
                return images[:, :1]

        images = torch.full((1, 3, 16, 16), 0.05)
        images[0, 0, 4, 7] = 0.95
        masks = torch.zeros((1, 1, 16, 16))
        masks[0, 0, 4, 7] = 1
        loader = [(images, masks, torch.ones(1))]
        scores, validation_loss = _validation_scores(
            ProbabilityFromImage(),
            loader,
            torch.device("cpu"),
            BoundaryAwareBCEDiceLoss(negative_weight=1.1, boundary_weight=0),
        )

        self.assertTrue(torch.isfinite(torch.tensor(validation_loss)))
        self.assertEqual(scores[0.1]["tiny_defect_dice"], 1.0)
        self.assertEqual(scores[0.1]["false_positive_pixels"], 0)
        self.assertEqual(scores[0.1]["false_negative_pixels"], 0)
        self.assertGreater(len(scores), 35)


if __name__ == "__main__":
    unittest.main()
