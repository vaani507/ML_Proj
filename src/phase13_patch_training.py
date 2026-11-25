import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.dataset import IMAGE_EXTENSIONS, SolarCellDataset
from src.models import ResNet18UNet
from src.preprocessing import MASK_THRESHOLD, load_aligned_pair
from src.train import BoundaryAwareBCEDiceLoss, compute_detailed_metrics
from src.train_resnet_unet import (
    EARLY_STOPPING_PATIENCE,
    THRESHOLD_CANDIDATES,
    _scheduler,
    _threshold_grid,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CHECKPOINT = PROJECT_ROOT / "results" / "resnet18_unet_phase5_negw_best.pth"
PHASE5_THRESHOLD = 0.165
PATCH_SIZE = 224
MAX_EPOCHS = 50
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_CPU_THREADS = 6
NEGATIVE_BCE_WEIGHT = 1.1
DECODER_LEARNING_RATE = 1e-3
MASK_SUFFIXES = ("_mask", "_label", "_groundtruth", "_gt")
PHASE5_CHECKPOINT_SHA256 = (
    "518a4de1485a8828f6bbeea89da19bafe5db86ae65054e5dbc6cadbbab4a8cd8"
)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_stem(path, is_label=False):
    stem = Path(path).stem.casefold()
    if is_label:
        for suffix in MASK_SUFFIXES:
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
    return stem


def _indexed_files(folder, is_label=False):
    indexed = {}
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = _canonical_stem(path, is_label=is_label)
        if key in indexed:
            raise ValueError(f"Duplicate raw data stem {key!r} in {folder}")
        indexed[key] = path
    return indexed


def _raw_pairs(project_root, processed_dataset):
    raw_root = Path(project_root) / "data" / "SolarCells" / "train"
    raw_images = _indexed_files(raw_root / "defect")
    raw_masks = _indexed_files(raw_root / "label", is_label=True)
    if raw_images.keys() != raw_masks.keys():
        raise ValueError(
            "Raw training images and masks do not pair one-to-one; "
            f"missing masks={sorted(raw_images.keys() - raw_masks.keys())[:10]}, "
            f"missing images={sorted(raw_masks.keys() - raw_images.keys())[:10]}"
        )

    pairs = []
    for processed_image, _ in processed_dataset.samples:
        key = _canonical_stem(processed_image)
        if key not in raw_images:
            raise ValueError(f"No raw training image found for {processed_image.name}")
        pairs.append((raw_images[key], raw_masks[key]))
    if len(pairs) != len(processed_dataset):
        raise ValueError("Raw and processed training datasets have different lengths")
    return pairs


def _tile_starts(length, tile_size=PATCH_SIZE):
    if length < tile_size:
        raise ValueError(
            f"Image dimension {length} is smaller than the {tile_size}-pixel patch"
        )
    starts = list(range(0, length - tile_size + 1, tile_size))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _image_tensor(image):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    normalized = (pixels - IMAGE_MEAN) / IMAGE_STD
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy())


def _load_pair(image_path, mask_path):
    image, mask = load_aligned_pair(image_path, mask_path)
    mask_array = np.asarray(mask, dtype=np.uint8)
    mask = Image.fromarray((mask_array > MASK_THRESHOLD).astype(np.uint8) * 255)
    return image, mask


class SolarCellPatchDataset(Dataset):
    """Native-resolution tiles, optionally rotating one tile per image each epoch."""

    def __init__(
        self,
        pairs,
        augment=False,
        patch_size=PATCH_SIZE,
        one_patch_per_image=False,
        seed=42,
    ):
        self.pairs = list(pairs)
        self.augment = augment
        self.patch_size = patch_size
        self.one_patch_per_image = one_patch_per_image
        self.color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15)
        self.tiles_by_image = []
        for image_index, (image_path, _) in enumerate(self.pairs):
            with Image.open(image_path) as source:
                width, height = ImageOps.exif_transpose(source).size
            image_tiles = []
            for top in _tile_starts(height, patch_size):
                for left in _tile_starts(width, patch_size):
                    image_tiles.append((left, top))
            self.tiles_by_image.append(image_tiles)
        self.tile_offsets = [
            random.Random(seed + image_index).randrange(len(image_tiles))
            for image_index, image_tiles in enumerate(self.tiles_by_image)
        ]
        self.epoch = 0

    def __len__(self):
        if self.one_patch_per_image:
            return len(self.pairs)
        return sum(len(tiles) for tiles in self.tiles_by_image)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must not be negative")
        self.epoch = epoch

    def __getitem__(self, index):
        if self.one_patch_per_image:
            image_index = index
            image_tiles = self.tiles_by_image[image_index]
            tile_index = (self.tile_offsets[image_index] + self.epoch) % len(
                image_tiles
            )
            left, top = image_tiles[tile_index]
        else:
            for image_index, image_tiles in enumerate(self.tiles_by_image):
                if index < len(image_tiles):
                    left, top = image_tiles[index]
                    break
                index -= len(image_tiles)
            else:
                raise IndexError("Patch index is out of range")
        image_path, mask_path = self.pairs[image_index]
        image, mask = _load_pair(image_path, mask_path)
        box = (
            left,
            top,
            left + self.patch_size,
            top + self.patch_size,
        )
        image = image.crop(box)
        mask = mask.crop(box)

        if self.augment:
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

        mask_array = np.asarray(mask, dtype=np.uint8) > 127
        mask_tensor = torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0)
        return (
            _image_tensor(image),
            mask_tensor,
            torch.tensor(float(mask_array.any()), dtype=torch.float32),
        )


class SolarCellFullResolutionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image, mask = _load_pair(image_path, mask_path)
        mask_array = np.asarray(mask, dtype=np.uint8) > 127
        return (
            _image_tensor(image),
            torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0),
            image_path.name,
        )


def predict_tiled(model, image, tile_batch_size=DEFAULT_BATCH_SIZE):
    """Predict an entire image by stitching aligned native-resolution tiles."""
    if image.ndim != 3:
        raise ValueError("Expected an image tensor with shape CxHxW")
    _, height, width = image.shape
    tile_origins = [
        (left, top)
        for top in _tile_starts(height)
        for left in _tile_starts(width)
    ]
    probability_sum = torch.zeros((1, 1, height, width), dtype=torch.float32)
    overlap_count = torch.zeros_like(probability_sum)
    tiles = torch.stack(
        [
            image[:, top : top + PATCH_SIZE, left : left + PATCH_SIZE]
            for left, top in tile_origins
        ]
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tiles), tile_batch_size):
            batch = tiles[start : start + tile_batch_size]
            output = model(batch)
            expected_shape = (len(batch), 1, PATCH_SIZE, PATCH_SIZE)
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected_shape:
                raise RuntimeError(
                    f"Expected tiled output shape {expected_shape}, got "
                    f"{getattr(output, 'shape', type(output).__name__)}"
                )
            if not torch.isfinite(output).all():
                raise RuntimeError("The model returned non-finite tile probabilities")
            for tile_index, (left, top) in enumerate(
                tile_origins[start : start + len(batch)]
            ):
                tile = output[tile_index : tile_index + 1]
                probability_sum[
                    :, :, top : top + PATCH_SIZE, left : left + PATCH_SIZE
                ] += tile
                overlap_count[
                    :, :, top : top + PATCH_SIZE, left : left + PATCH_SIZE
                ] += 1
    if torch.any(overlap_count == 0):
        raise RuntimeError("Tiled inference did not cover every image pixel")
    return probability_sum / overlap_count


def _full_resolution_predictions(model, dataset, tiled):
    predictions = []
    targets = []
    names = []
    model.eval()
    for image, mask, name in dataset:
        if tiled:
            prediction = predict_tiled(model, image)
        else:
            resized = Image.fromarray(
                np.clip(
                    (image.permute(1, 2, 0).numpy() * IMAGE_STD + IMAGE_MEAN)
                    * 255.0,
                    0,
                    255,
                ).round().astype(np.uint8)
            ).resize(
                (PATCH_SIZE, PATCH_SIZE),
                Image.Resampling.BILINEAR,
            )
            with torch.inference_mode():
                low_resolution = model(_image_tensor(resized).unsqueeze(0))
            prediction = F.interpolate(
                low_resolution,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        predictions.append(prediction.cpu())
        targets.append(mask.unsqueeze(0))
        names.append(name)
    if not predictions:
        raise ValueError("Validation dataset is empty")
    return torch.cat(predictions), torch.cat(targets), names


def _threshold_scores(predictions, targets):
    defect_fractions = targets.sum(dim=(1, 2, 3)) / targets[0, 0].numel()
    tiny_indices = (defect_fractions > 0) & (defect_fractions <= 0.02)
    scores = {}
    thresholds = set(THRESHOLD_CANDIDATES)
    thresholds.add(PHASE5_THRESHOLD)
    for threshold in THRESHOLD_CANDIDATES:
        prediction = predictions > threshold
        target = targets > 0.5
        metrics = compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        metrics["tiny_defect_dice"] = (
            compute_detailed_metrics(
                predictions[tiny_indices],
                targets[tiny_indices],
                threshold=threshold,
            )["dice"]
            if tiny_indices.any()
            else None
        )
        metrics["false_positive_pixels"] = int((prediction & ~target).sum())
        metrics["false_negative_pixels"] = int((~prediction & target).sum())
        scores[threshold] = metrics
    thresholds.update(_threshold_grid(scores))
    for threshold in sorted(thresholds - set(scores)):
        prediction = predictions > threshold
        target = targets > 0.5
        metrics = compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        metrics["tiny_defect_dice"] = (
            compute_detailed_metrics(
                predictions[tiny_indices],
                targets[tiny_indices],
                threshold=threshold,
            )["dice"]
            if tiny_indices.any()
            else None
        )
        metrics["false_positive_pixels"] = int((prediction & ~target).sum())
        metrics["false_negative_pixels"] = int((~prediction & target).sum())
        scores[threshold] = metrics
    return scores


def _selected_threshold(scores):
    return max(scores, key=lambda threshold: scores[threshold]["dice"])


def _save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_history(history):
    rows = []
    for record in history:
        row = {
            key: value
            for key, value in record.items()
            if key not in {"validation"}
        }
        row.update(
            {
                f"validation_{key}": value
                for key, value in record["validation"].items()
            }
        )
        rows.append(row)
    return rows


def _training_pairs(project_root, seed=42):
    processed = SolarCellDataset(
        Path(project_root) / "data" / "processed" / "SolarCells" / "train" / "images",
        Path(project_root) / "data" / "processed" / "SolarCells" / "train" / "masks",
    )
    validation_size = max(1, round(len(processed) * 0.2))
    training_size = len(processed) - validation_size
    if training_size <= 0:
        raise ValueError("The training split must contain at least one image")
    train_subset, validation_subset = random_split(
        processed,
        (training_size, validation_size),
        generator=torch.Generator().manual_seed(seed),
    )
    pairs = _raw_pairs(project_root, processed)
    return (
        [pairs[index] for index in train_subset.indices],
        [pairs[index] for index in validation_subset.indices],
    )


def _load_phase5_model(device):
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Phase 5 checkpoint not found: {BASELINE_CHECKPOINT}")
    state, metadata = _load_state(BASELINE_CHECKPOINT)
    if (
        metadata.get("phase") != "decoder"
        or metadata.get("negative_bce_weight") != NEGATIVE_BCE_WEIGHT
        or float(metadata.get("boundary_weight") or 0.0) != 0.0
    ):
        raise ValueError("The checkpoint is not the selected Phase 5 model")
    model = ResNet18UNet(pretrained=False)
    model.load_state_dict(state)
    model.to(device)
    model.freeze_encoder()
    return model, metadata


def _per_image_rows(names, targets, phase5_predictions, patch_predictions, baseline_threshold, patch_threshold):
    rows = []
    for index, name in enumerate(names):
        target = targets[index : index + 1]
        defect_fraction = float(target.mean())
        is_tiny = 0 < defect_fraction <= 0.02
        values = {
            "phase5_locked": (phase5_predictions, PHASE5_THRESHOLD),
            "phase5_validation_selected": (
                phase5_predictions,
                baseline_threshold,
            ),
            "patch_locked": (patch_predictions, PHASE5_THRESHOLD),
            "patch_validation_selected": (patch_predictions, patch_threshold),
        }
        row = {
            "image": name,
            "ground_truth_defect_fraction": defect_fraction,
            "tiny_defect_image": is_tiny,
        }
        for label, (predictions, threshold) in values.items():
            metrics = compute_detailed_metrics(
                predictions[index : index + 1],
                target,
                threshold=threshold,
            )
            row[f"{label}_dice"] = metrics["dice"]
            row[f"{label}_iou"] = metrics["iou"]
            row[f"{label}_precision"] = metrics["precision"]
            row[f"{label}_recall"] = metrics["recall"]
            predicted = predictions[index : index + 1] > threshold
            actual = target > 0.5
            row[f"{label}_false_positive_pixels"] = int(
                (predicted & ~actual).sum()
            )
            row[f"{label}_false_negative_pixels"] = int(
                (~predicted & actual).sum()
            )
        rows.append(row)
    return rows


def _save_examples(path, dataset, names, targets, phase5_predictions, patch_predictions, baseline_threshold, patch_threshold):
    import matplotlib.pyplot as plt

    per_image = []
    for index in range(len(names)):
        target = targets[index : index + 1]
        baseline_dice = compute_detailed_metrics(
            phase5_predictions[index : index + 1],
            target,
            threshold=baseline_threshold,
        )["dice"]
        patch_dice = compute_detailed_metrics(
            patch_predictions[index : index + 1],
            target,
            threshold=patch_threshold,
        )["dice"]
        per_image.append(
            (
                index,
                patch_dice - baseline_dice,
                patch_dice,
                float(target.mean()),
            )
        )

    chosen = []
    for index, _, _, _ in sorted(per_image, key=lambda value: value[2]):
        if index not in chosen:
            chosen.append(index)
            break
    for index, delta, _, _ in sorted(per_image, key=lambda value: value[1]):
        if delta > 0 and index not in chosen:
            chosen.append(index)
            break
    for index, _, _, _ in sorted(per_image, key=lambda value: value[3]):
        if index not in chosen:
            chosen.append(index)
            break
    for index, _, _, _ in sorted(per_image, key=lambda value: value[1], reverse=True):
        if index not in chosen:
            chosen.append(index)
            break

    figure, axes = plt.subplots(
        len(chosen),
        5,
        figsize=(16, max(3.2, 3.2 * len(chosen))),
        squeeze=False,
    )
    headings = (
        "Original EL",
        "Ground truth",
        "Phase 5 (0.165)",
        f"Patch model ({patch_threshold:.3f})",
        "Patch TP / FP / FN",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row_index, sample_index in enumerate(chosen):
        image, target, _ = dataset[sample_index]
        rgb = np.clip(
            (image.permute(1, 2, 0).numpy() * IMAGE_STD + IMAGE_MEAN),
            0,
            1,
        )
        target_mask = targets[sample_index, 0].numpy() > 0.5
        baseline_mask = (
            phase5_predictions[sample_index, 0].numpy() > baseline_threshold
        )
        patch_mask = (
            patch_predictions[sample_index, 0].numpy() > patch_threshold
        )
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 1].imshow(target_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(baseline_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(patch_mask, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*target_mask.shape, 4), dtype=np.float32)
        overlay[target_mask & patch_mask] = (0.1, 1.0, 0.1, 0.8)
        overlay[~target_mask & patch_mask] = (1.0, 0.1, 0.1, 0.85)
        overlay[target_mask & ~patch_mask] = (0.1, 0.35, 1.0, 0.85)
        axes[row_index, 4].imshow(rgb)
        axes[row_index, 4].imshow(overlay)
        axes[row_index, 0].set_ylabel(names[sample_index])
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _evaluate_candidates(model, validation_dataset, criterion, tiled):
    predictions, targets, names = _full_resolution_predictions(
        model,
        validation_dataset,
        tiled=tiled,
    )
    scores = _threshold_scores(predictions, targets)
    threshold = _selected_threshold(scores)
    validation_loss = criterion(predictions, targets).item()
    return predictions, targets, names, scores, threshold, validation_loss


def _output_paths(output_prefix):
    output_prefix = Path(output_prefix)
    return {
        "best": output_prefix.with_name(output_prefix.name + "_best.pth"),
        "last": output_prefix.with_name(output_prefix.name + "_last.pth"),
        "history": output_prefix.with_name(output_prefix.name + "_training_history.csv"),
        "thresholds": output_prefix.with_name(output_prefix.name + "_validation_thresholds.csv"),
        "per_image": output_prefix.with_name(output_prefix.name + "_validation_per_image.csv"),
        "summary": output_prefix.with_name(output_prefix.name + "_validation_summary.json"),
        "examples": output_prefix.with_name(output_prefix.name + "_validation_examples.png"),
        "probabilities": output_prefix.with_name(output_prefix.name + "_validation_probabilities.npz"),
    }


def finalize_patch_run(
    project_root=PROJECT_ROOT,
    output_prefix=None,
    batch_size=DEFAULT_BATCH_SIZE,
    num_threads=DEFAULT_CPU_THREADS,
    seed=42,
):
    project_root = Path(project_root)
    if output_prefix is None:
        output_prefix = project_root / "results" / "resnet18_unet_phase13_patch"
    output_prefix = Path(output_prefix)
    output_paths = _output_paths(output_prefix)
    if output_paths["summary"].exists():
        raise FileExistsError(
            f"Phase 13 validation report already exists: {output_paths['summary']}"
        )
    if not output_paths["best"].is_file() or not output_paths["last"].is_file():
        raise FileNotFoundError(
            "Both Phase 13 best and resumable checkpoints are required to "
            "finalize the saved experiment."
        )

    torch.set_num_threads(num_threads)
    train_pairs, validation_pairs = _training_pairs(project_root, seed=seed)
    training_dataset = SolarCellPatchDataset(
        train_pairs,
        augment=False,
        one_patch_per_image=True,
        seed=seed,
    )
    validation_dataset = SolarCellFullResolutionDataset(validation_pairs)
    last_checkpoint = torch.load(
        output_paths["last"],
        map_location="cpu",
        weights_only=False,
    )
    history = last_checkpoint.get("history", [])
    if not history:
        raise ValueError("The saved Phase 13 run contains no training history")
    training_seconds = sum(
        record["training_seconds"] for record in history
    )
    validation_seconds = sum(
        record["validation_seconds"] for record in history
    )

    phase5_hash_before = _file_sha256(BASELINE_CHECKPOINT)
    if phase5_hash_before != PHASE5_CHECKPOINT_SHA256:
        raise RuntimeError("The frozen Phase 5 checkpoint hash does not match")
    device = torch.device("cpu")
    criterion = BoundaryAwareBCEDiceLoss(
        negative_weight=NEGATIVE_BCE_WEIGHT,
        boundary_weight=0.0,
    )
    phase5_model, _ = _load_phase5_model(device)
    phase5_predictions, targets, names, baseline_scores, baseline_threshold, _ = (
        _evaluate_candidates(
            phase5_model,
            validation_dataset,
            criterion,
            tiled=False,
        )
    )
    patch_model = ResNet18UNet(pretrained=False)
    best_checkpoint = torch.load(
        output_paths["best"],
        map_location=device,
        weights_only=False,
    )
    patch_model.load_state_dict(best_checkpoint["model_state"])
    patch_model.freeze_encoder()
    patch_model.to(device)
    patch_predictions, patch_targets, patch_names, patch_scores, patch_threshold, _ = (
        _evaluate_candidates(
            patch_model,
            validation_dataset,
            criterion,
            tiled=True,
        )
    )
    if names != patch_names or not torch.equal(targets, patch_targets):
        raise RuntimeError("Phase 5 and patch validation records are not aligned")

    _write_csv(output_paths["history"], _flatten_history(history))
    _write_csv(
        output_paths["thresholds"],
        [
            {"model": model_name, "threshold": threshold, **metrics}
            for model_name, scores in (
                ("phase5", baseline_scores),
                ("phase13_patch", patch_scores),
            )
            for threshold, metrics in sorted(scores.items())
        ],
    )
    _write_csv(
        output_paths["per_image"],
        _per_image_rows(
            names,
            targets,
            phase5_predictions,
            patch_predictions,
            baseline_threshold,
            patch_threshold,
        ),
    )
    _save_examples(
        output_paths["examples"],
        validation_dataset,
        names,
        targets,
        phase5_predictions,
        patch_predictions,
        PHASE5_THRESHOLD,
        patch_threshold,
    )
    np.savez_compressed(
        output_paths["probabilities"],
        file_names=np.asarray(names),
        targets=targets.numpy(),
        phase5_probabilities=phase5_predictions.numpy(),
        patch_probabilities=patch_predictions.numpy(),
    )

    summary = {
        "experiment": "Phase 13 native-resolution patch-based segmentation",
        "test_evaluated": False,
        "training_images": len(train_pairs),
        "validation_images": len(validation_pairs),
        "training_patches_per_epoch": len(training_dataset),
        "candidate_training_patches": sum(
            len(tiles) for tiles in training_dataset.tiles_by_image
        ),
        "patch_sampling": "one rotating native-resolution tile per training image per epoch",
        "patch_size": PATCH_SIZE,
        "epochs_completed": len(history),
        "best_epoch": best_checkpoint.get("epoch"),
        "batch_size": batch_size,
        "cpu_threads": num_threads,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "phase5_checkpoint": str(BASELINE_CHECKPOINT),
        "phase5_checkpoint_sha256_before": phase5_hash_before,
        "phase5_checkpoint_sha256_after": _file_sha256(BASELINE_CHECKPOINT),
        "phase5_locked_threshold": PHASE5_THRESHOLD,
        "phase5_validation_selected_threshold": baseline_threshold,
        "patch_validation_selected_threshold": patch_threshold,
        "phase5_locked_validation_metrics": baseline_scores[PHASE5_THRESHOLD],
        "phase5_validation_selected_metrics": baseline_scores[baseline_threshold],
        "patch_locked_validation_metrics": patch_scores[PHASE5_THRESHOLD],
        "patch_validation_selected_metrics": patch_scores[patch_threshold],
        "selected_patch_checkpoint": str(output_paths["best"]),
        "artifacts": {
            key: str(path)
            for key, path in output_paths.items()
        },
    }
    if summary["phase5_checkpoint_sha256_after"] != phase5_hash_before:
        raise RuntimeError("The protected Phase 5 checkpoint changed during evaluation")
    output_paths["summary"].write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Best patch checkpoint: {output_paths['best']}")
    print(f"Patch validation Dice: {patch_scores[patch_threshold]['dice']:.4f}")
    print(f"Patch validation tiny Dice: {patch_scores[patch_threshold]['tiny_defect_dice']}")
    print(f"Training time: {training_seconds:.1f} seconds")
    print(f"Artifacts: {output_prefix.parent}")
    return summary


def train_patch_model(
    project_root=PROJECT_ROOT,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    num_threads=DEFAULT_CPU_THREADS,
    seed=42,
    output_prefix=None,
):
    if epochs <= 0 or epochs > MAX_EPOCHS:
        raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}")
    if batch_size <= 0 or num_threads <= 0:
        raise ValueError("batch_size and num_threads must be positive")
    project_root = Path(project_root)
    if output_prefix is None:
        output_prefix = project_root / "results" / "resnet18_unet_phase13_patch"
    output_prefix = Path(output_prefix)
    output_paths = _output_paths(output_prefix)
    existing = [path for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Phase 13 artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    torch.set_num_threads(num_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_pairs, validation_pairs = _training_pairs(project_root, seed=seed)
    training_dataset = SolarCellPatchDataset(
        train_pairs,
        augment=True,
        one_patch_per_image=True,
        seed=seed,
    )
    validation_dataset = SolarCellFullResolutionDataset(validation_pairs)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    device = torch.device("cpu")
    model, _ = _load_phase5_model(device)
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("The Phase 5 encoder must remain frozen in this experiment")

    criterion = BoundaryAwareBCEDiceLoss(
        negative_weight=NEGATIVE_BCE_WEIGHT,
        boundary_weight=0.0,
    )
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=DECODER_LEARNING_RATE,
    )
    scheduler = _scheduler(optimizer)
    history = []
    best_dice = -1.0
    best_threshold = THRESHOLD_CANDIDATES[0]
    epochs_without_improvement = 0
    training_seconds = 0.0
    validation_seconds = 0.0
    baseline_hash_before = _file_sha256(BASELINE_CHECKPOINT)
    if baseline_hash_before != PHASE5_CHECKPOINT_SHA256:
        raise RuntimeError("The frozen Phase 5 checkpoint hash does not match")

    for epoch in range(epochs):
        training_dataset.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        training_started = time.perf_counter()
        total_loss = 0.0
        sample_count = 0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(images)
            loss = criterion(probabilities, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
            sample_count += len(images)
        if sample_count == 0:
            raise ValueError("Training loader produced no patches")
        epoch_training_seconds = time.perf_counter() - training_started
        training_seconds += epoch_training_seconds

        validation_started = time.perf_counter()
        predictions, targets, _, scores, selected_threshold, validation_loss = (
            _evaluate_candidates(model, validation_dataset, criterion, tiled=True)
        )
        epoch_validation_seconds = time.perf_counter() - validation_started
        validation_seconds += epoch_validation_seconds
        selected_metrics = scores[selected_threshold]
        scheduler.step(selected_metrics["dice"])
        improved = selected_metrics["dice"] > best_dice
        if improved:
            best_dice = selected_metrics["dice"]
            best_threshold = selected_threshold
            epochs_without_improvement = 0
            _save_checkpoint(
                output_paths["best"],
                {
                    "model_state": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "phase": "phase13_patch_decoder",
                    "selected_from": str(BASELINE_CHECKPOINT),
                    "best_validation_dice": best_dice,
                    "best_threshold": best_threshold,
                    "validation_metrics": selected_metrics,
                    "input_patch_size": PATCH_SIZE,
                    "negative_bce_weight": NEGATIVE_BCE_WEIGHT,
                    "boundary_weight": 0.0,
                    "optimizer_learning_rate": DECODER_LEARNING_RATE,
                    "batch_size": batch_size,
                    "cpu_threads": num_threads,
                    "epoch": epoch + 1,
                },
            )
        else:
            epochs_without_improvement += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / sample_count,
            "validation_loss": validation_loss,
            "threshold": selected_threshold,
            "validation": selected_metrics,
            "training_seconds": epoch_training_seconds,
            "validation_seconds": epoch_validation_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        _save_checkpoint(
            output_paths["last"],
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "best_validation_dice": best_dice,
                "best_threshold": best_threshold,
                "epochs_without_improvement": epochs_without_improvement,
            },
        )
        print(
            f"patch epoch {epoch + 1}/{epochs} | "
            f"loss={record['train_loss']:.4f} | "
            f"val_dice={selected_metrics['dice']:.4f} | "
            f"val_iou={selected_metrics['iou']:.4f} | "
            f"tiny_dice={selected_metrics['tiny_defect_dice']} | "
            f"fp={selected_metrics['false_positive_pixels']} | "
            f"fn={selected_metrics['false_negative_pixels']} | "
            f"threshold={selected_threshold:.3f} | "
            f"seconds={time.perf_counter() - epoch_started:.1f}"
        )
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping after {EARLY_STOPPING_PATIENCE} epochs "
                "without validation Dice improvement."
            )
            break

    return finalize_patch_run(
        project_root=project_root,
        output_prefix=output_prefix,
        batch_size=batch_size,
        num_threads=num_threads,
        seed=seed,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the Phase 5 ResNet-18 U-Net on native-resolution EL patches."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Create validation reports from an already-trained Phase 13 checkpoint.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results" / "resnet18_unet_phase13_patch",
    )
    args = parser.parse_args()
    if args.epochs <= 0 or args.epochs > MAX_EPOCHS:
        parser.error(f"--epochs must be between 1 and {MAX_EPOCHS}")
    if args.batch_size <= 0 or args.num_threads <= 0:
        parser.error("--batch-size and --num-threads must be positive")
    if args.finalize_existing:
        finalize_patch_run(
            output_prefix=args.output_prefix,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            seed=args.seed,
        )
        return
    train_patch_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        seed=args.seed,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()
