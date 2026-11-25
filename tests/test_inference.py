import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.inference import (
    create_heatmap_image,
    create_mask_image,
    create_overlay_image,
    decode_uploaded_image,
    encode_png,
    load_model,
    model_checkpoint_path,
    preprocess_image,
    run_inference,
    summarize_mask,
    threshold_probability_map,
)


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FixedProbabilityModel(nn.Module):
    def __init__(self, probability=0.2):
        super().__init__()
        self.probability = probability
        self.last_input = None

    def forward(self, inputs):
        self.last_input = inputs.detach().clone()
        return inputs.new_full(
            (inputs.shape[0], 1, *inputs.shape[-2:]),
            self.probability,
        )


class InferenceTests(unittest.TestCase):
    def test_phase5_loader_checks_checkpoint_in_requested_project_dir(self):
        with TemporaryDirectory() as project_dir:
            checkpoint = model_checkpoint_path("phase5_resnet18", project_dir)
            self.assertEqual(
                checkpoint,
                Path(project_dir) / "results" / "resnet18_unet_phase5_negw_best.pth",
            )
            checkpoint.parent.mkdir()
            model = nn.Conv2d(3, 1, kernel_size=1)
            torch.save(model.state_dict(), checkpoint)
            with patch("src.inference._load_state") as deserialize:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    load_model("phase5_resnet18", project_dir)
                deserialize.assert_not_called()

            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with (
                patch("src.inference.PHASE5_CHECKPOINT_SHA256", digest),
                patch("src.inference.ResNet18UNet", return_value=model),
            ):
                self.assertIs(load_model("phase5_resnet18", project_dir), model)
                self.assertFalse(model.training)

    def test_preprocessing_matches_rgb_resize_and_imagenet_normalization(self):
        image = Image.new("RGB", (320, 160), (255, 127, 0))
        tensor = preprocess_image(image)

        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))
        self.assertAlmostEqual(
            float(tensor[0, 0, 0, 0]),
            (1.0 - 0.485) / 0.229,
            places=5,
        )
        self.assertAlmostEqual(
            float(tensor[0, 1, 0, 0]),
            (127 / 255.0 - 0.456) / 0.224,
            places=5,
        )
        self.assertAlmostEqual(
            float(tensor[0, 2, 0, 0]),
            (0.0 - 0.406) / 0.225,
            places=5,
        )

    def test_upload_validation_and_inference_preserve_original_dimensions(self):
        image_bytes = png_bytes(Image.new("RGB", (320, 160), (80, 120, 160)))
        image, source_format = decode_uploaded_image(image_bytes, "cell.PNG")
        self.assertEqual(source_format, "PNG")
        self.assertEqual(image.size, (320, 160))

        model = FixedProbabilityModel(0.2)
        result = run_inference(
            model,
            image_bytes,
            "cell.PNG",
            model_key="phase5_resnet18",
        )

        self.assertEqual(tuple(model.last_input.shape), (1, 3, 224, 224))
        self.assertEqual(model.last_input.device.type, "cpu")
        self.assertFalse(model.training)
        self.assertEqual(result.probability_map.shape, (160, 320))
        self.assertEqual(result.predicted_mask.shape, (160, 320))
        self.assertTrue(np.all(result.predicted_mask == 1))
        self.assertTrue(
            np.all(threshold_probability_map(result.probability_map, 0.25) == 0)
        )

    def test_invalid_images_and_thresholds_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "PNG, JPG, or JPEG"):
            decode_uploaded_image(b"not an image", "cell.tiff")
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            decode_uploaded_image(b"not an image", "cell.png")
        image_bytes = png_bytes(Image.new("RGB", (20, 10)))
        with patch("src.inference.MAX_IMAGE_PIXELS", 100):
            with self.assertRaisesRegex(ValueError, "too large"):
                decode_uploaded_image(image_bytes, "cell.png")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            threshold_probability_map(np.zeros((2, 2)), 1.1)

    def test_region_summary_and_visual_downloads(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        mask[10:15, 20:24] = 1
        summary = summarize_mask(mask)

        self.assertTrue(summary["defect_pixels_detected"])
        self.assertEqual(summary["connected_regions"], 2)
        self.assertEqual(summary["predicted_defect_pixels"], 24)
        self.assertEqual(summary["largest_region_pixels"], 20)
        self.assertAlmostEqual(summary["predicted_image_area_percent"], 4.0)
        self.assertEqual(summarize_mask(np.zeros((5, 5)))["connected_regions"], 0)

        original = Image.new("RGB", (30, 20), (100, 100, 100))
        probability = mask.astype(np.float32) * 0.8
        outputs = (
            create_mask_image(mask),
            create_overlay_image(original, mask),
            create_heatmap_image(probability),
        )
        for output in outputs:
            self.assertEqual(output.size, original.size)
            encoded = encode_png(output)
            self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(BytesIO(encoded)) as decoded:
                decoded.verify()


if __name__ == "__main__":
    unittest.main()
