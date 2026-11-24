import unittest

import torch

from src.train import BCEDiceLoss, BoundaryAwareBCEDiceLoss
from src.phase7_analysis import _boundary_f1


class BoundaryAwareBCEDiceLossTests(unittest.TestCase):
    def test_zero_boundary_weight_matches_existing_loss_exactly(self):
        prediction = torch.tensor(
            [[[[0.1, 0.8], [0.4, 0.9]]]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[[[0.0, 1.0], [0.0, 1.0]]]],
            dtype=torch.float32,
        )
        baseline = BCEDiceLoss(negative_weight=1.1)(prediction, target)
        zero_weight = BoundaryAwareBCEDiceLoss(
            negative_weight=1.1,
            boundary_weight=0,
        )(prediction, target)
        self.assertTrue(torch.equal(baseline, zero_weight))

    def test_empty_masks_have_finite_loss_and_gradients(self):
        prediction = torch.full((2, 1, 16, 16), 0.2, requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = BoundaryAwareBCEDiceLoss(
            negative_weight=1.1,
            boundary_weight=0.1,
        )(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(prediction.grad.shape, prediction.shape)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_saturated_probability_inputs_have_finite_loss_and_gradients(self):
        prediction = torch.tensor(
            [[[[0.0, 1.0], [1.0, 0.0]]]],
            requires_grad=True,
        )
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        loss = BoundaryAwareBCEDiceLoss(
            negative_weight=1.1,
            boundary_weight=0.1,
        )(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_thin_defect_has_finite_nonzero_boundary_gradients(self):
        prediction = torch.full((1, 1, 16, 16), 0.2)
        target = torch.zeros_like(prediction)
        target[0, 0, 7, 3:13] = 1
        prediction[0, 0, 7, 4:8] = 0.7
        prediction.requires_grad_()
        baseline_gradient = torch.autograd.grad(
            BCEDiceLoss(negative_weight=1.1)(prediction, target),
            prediction,
            retain_graph=True,
        )[0]
        loss = BoundaryAwareBCEDiceLoss(
            negative_weight=1.1,
            boundary_weight=0.1,
        )(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(torch.count_nonzero(prediction.grad).item(), 0)
        self.assertGreater(
            torch.count_nonzero(
                (prediction.grad - baseline_gradient).abs() > 1e-8
            ).item(),
            0,
        )
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_boundary_f1_is_one_for_identical_thin_masks(self):
        target = torch.zeros((1, 1, 16, 16))
        target[0, 0, 7, 3:13] = 1
        self.assertEqual(_boundary_f1(target, target, threshold=0.5), 1.0)

    def test_prediction_and_target_shapes_must_match(self):
        prediction = torch.full((1, 1, 8, 8), 0.5)
        target = torch.zeros((1, 1, 8, 7))
        with self.assertRaisesRegex(ValueError, "same NCHW shape"):
            BoundaryAwareBCEDiceLoss()(prediction, target)


if __name__ == "__main__":
    unittest.main()
