from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.transforms as T


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SolarCellDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, augmentation=False):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.augmentation = augmentation
        self.color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15)

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_paths = sorted(
            path
            for path in self.img_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        image_stems = [path.stem for path in image_paths]
        if len(image_stems) != len(set(image_stems)):
            raise ValueError(f"Duplicate image filename stems in {self.img_dir}")

        mask_paths = [
            path
            for path in self.mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        mask_stems = [path.stem for path in mask_paths]
        if len(mask_stems) != len(set(mask_stems)):
            raise ValueError(f"Duplicate mask filename stems in {self.mask_dir}")
        masks_by_stem = {path.stem: path for path in mask_paths}
        image_stems = set(image_stems)
        missing_masks = sorted(image_stems - masks_by_stem.keys())
        missing_images = sorted(masks_by_stem.keys() - image_stems)
        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired image/mask files in {self.img_dir} and {self.mask_dir}; "
                f"images without masks: {missing_masks[:10]}, "
                f"masks without images: {missing_images[:10]}"
            )
        if not image_paths:
            raise ValueError(f"No supported image files found in {self.img_dir}")

        self.samples = [
            (image_path, masks_by_stem[image_path.stem])
            for image_path in image_paths
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")

        if image.size != mask.size:
            raise ValueError(
                f"Image and mask sizes differ for {image_path.name}: "
                f"{image.size} versus {mask.size}"
            )

        if self.augmentation:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rotations = random.randrange(4)
            for _ in range(rotations):
                image = image.transpose(Image.Transpose.ROTATE_90)
                mask = mask.transpose(Image.Transpose.ROTATE_90)
            image = self.color_jitter(image)

        mask_arr = np.asarray(mask, dtype=np.uint8)
        mask_values = np.unique(mask_arr)
        if not np.isin(mask_values, (0, 1)).all():
            raise ValueError(
                f"Expected a binary mask containing only 0 and 1: {mask_path}"
            )

        cls_label = 1.0 if np.sum(mask_arr) > 0 else 0.0

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        mask_tensor = torch.from_numpy(mask_arr.astype(np.float32)).unsqueeze(0)
        cls_tensor = torch.tensor(cls_label, dtype=torch.float32)

        return image, mask_tensor, cls_tensor


def get_dataloaders(
    root_processed_dir,
    batch_size=8,
    validation_fraction=0.2,
    seed=42,
    augment_training=False,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root_processed_dir = Path(root_processed_dir)
    train_dir = root_processed_dir / "train"
    test_dir = root_processed_dir / "test"

    train_dataset = SolarCellDataset(
        train_dir / "images",
        train_dir / "masks",
        transform=transform,
        augmentation=augment_training,
    )
    validation_dataset = SolarCellDataset(
        train_dir / "images",
        train_dir / "masks",
        transform=transform,
    )
    test_dataset = SolarCellDataset(
        test_dir / "images",
        test_dir / "masks",
        transform=transform,
    )

    if len(train_dataset) < 2:
        raise ValueError("At least two training samples are needed for a validation split")

    validation_size = max(1, round(len(train_dataset) * validation_fraction))
    training_size = len(train_dataset) - validation_size
    if training_size == 0:
        raise ValueError("The validation split leaves no samples for training")

    training_indices, validation_indices = random_split(
        train_dataset,
        (training_size, validation_size),
        generator=torch.Generator().manual_seed(seed),
    )
    training_subset = Subset(train_dataset, training_indices.indices)
    validation_subset = Subset(validation_dataset, validation_indices.indices)
    train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader