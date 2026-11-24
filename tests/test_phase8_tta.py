import unittest

import torch
import torch.nn as nn

from src.phase8_tta import TTA_CONFIGURATIONS, predict_with_tta


class FirstChannel(nn.Module):
    def forward(self, inputs):
        return inputs[:, :1]


class TestTimeAugmentationTests(unittest.TestCase):
    def test_inverse_transforms_preserve_spatial_alignment(self):
        images = torch.linspace(0, 1, 3 * 7 * 11).reshape(1, 3, 7, 11)
        expected = images[:, :1]
        model = FirstChannel()
        for transforms in TTA_CONFIGURATIONS.values():
            with self.subTest(transforms=transforms):
                result = predict_with_tta(model, images, transforms)
                self.assertEqual(result.shape, expected.shape)
                self.assertTrue(torch.allclose(result, expected))
        self.assertFalse(model.training)

    def test_empty_transform_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            predict_with_tta(FirstChannel(), torch.zeros(1, 3, 4, 4), ())


if __name__ == "__main__":
    unittest.main()
