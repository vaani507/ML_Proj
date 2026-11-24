import unittest

import torch
from torch import nn

from src.models import ResNet18UNet
from src.train import BoundaryAwareBCEDiceLoss
from src.train_resnet_unet import _optimizer, _validation_scores


class FinalBlockFineTuningTests(unittest.TestCase):
    def test_only_layer4_and_decoder_are_trainable_with_batch_norm_frozen(self):
        model = ResNet18UNet(pretrained=False)
        model.unfreeze_final_block()
        model.train()

        for name, parameter in model.encoder.named_parameters():
            if name.startswith("layer4."):
                self.assertTrue(parameter.requires_grad, name)
            else:
                self.assertFalse(parameter.requires_grad, name)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.decoder.parameters())
        )

        self.assertTrue(model.encoder.layer4.training)
        for module in model.encoder.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertFalse(module.training)
                self.assertTrue(module.weight.requires_grad)
                self.assertTrue(module.bias.requires_grad)

    def test_fine_tuning_optimizer_uses_requested_learning_rates(self):
        model = ResNet18UNet(pretrained=False)
        model.unfreeze_final_block()

        optimizer = _optimizer(model, fine_tune=True)

        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [5e-5, 1e-5],
        )
        self.assertEqual(
            list(optimizer.param_groups[0]["params"]),
            list(model.decoder.parameters()),
        )
        self.assertEqual(
            list(optimizer.param_groups[1]["params"]),
            list(model.encoder.layer4.parameters()),
        )

    def test_validation_scoring_returns_finite_loss(self):
        class ConstantModel(nn.Module):
            def forward(self, images):
                return torch.full(
                    (len(images), 1, *images.shape[-2:]),
                    0.2,
                )

        images = torch.zeros((2, 3, 8, 8))
        masks = torch.zeros((2, 1, 8, 8))
        loader = [(images, masks, torch.zeros(2))]
        scores, validation_loss = _validation_scores(
            ConstantModel(),
            loader,
            torch.device("cpu"),
            BoundaryAwareBCEDiceLoss(negative_weight=1.1, boundary_weight=0),
        )

        self.assertIn(0.1, scores)
        self.assertTrue(torch.isfinite(torch.tensor(validation_loss)))


if __name__ == "__main__":
    unittest.main()
