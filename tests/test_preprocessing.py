from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from src.preprocessing import EXIF_ORIENTATION, load_aligned_pair


class PreprocessingTests(unittest.TestCase):
    def test_untagged_mask_rotates_with_the_image(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "cell.png"
            mask_path = Path(directory) / "cell_mask.png"
            image = Image.new("RGB", (3, 2))
            image.putpixel((0, 0), (255, 255, 255))
            exif = image.getexif()
            exif[EXIF_ORIENTATION] = 6
            image.save(image_path, exif=exif)
            mask = image.convert("L")
            mask.save(mask_path)

            aligned_image, aligned_mask = load_aligned_pair(image_path, mask_path)

            self.assertEqual(aligned_image.size, (2, 3))
            self.assertEqual(aligned_mask.size, aligned_image.size)
            self.assertEqual(aligned_mask.getpixel((1, 0)), 255)
            self.assertEqual(aligned_mask.tobytes(), aligned_image.convert("L").tobytes())

    def test_pair_rejects_mismatched_dimensions_and_orientation(self):
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "cell.png"
            mask_path = Path(directory) / "cell_mask.png"
            Image.new("RGB", (3, 2)).save(image_path)
            Image.new("L", (2, 3)).save(mask_path)
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                load_aligned_pair(image_path, mask_path)

            mask = Image.new("L", (3, 2))
            exif = mask.getexif()
            exif[EXIF_ORIENTATION] = 6
            mask.save(mask_path, exif=exif)
            with self.assertRaisesRegex(ValueError, "Conflicting EXIF"):
                load_aligned_pair(image_path, mask_path)
