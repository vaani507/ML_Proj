import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.dataset import get_dataloaders
from src.models import ResNet18UNet
from src.phase7_analysis import _boundary_f1
from src.train import compute_detailed_metrics


SIZE_GROUPS = ("tiny", "medium", "large", "empty")
TTA_CONFIGURATIONS = {
    "A_original": ((),),
    "B_original_hflip": ((), (-1,)),
    "C_original_hv_flip": ((), (-1,), (-2,)),
    "D_original_hv_180": ((), (-1,), (-2,), (-2, -1)),
}


def predict_with_tta(model, images, transforms):
    if not transforms:
        raise ValueError("At least one inference transform is required")
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for dimensions in transforms:
            transformed = (
                torch.flip(images, dims=dimensions)
                if dimensions
                else images
            )
            prediction = model(transformed)
            if prediction.shape != (images.shape[0], 1, *images.shape[-2:]):
                raise ValueError(
                    "Model output must have shape [batch, 1, height, width] "
                    "matching the input spatial dimensions"
                )
            aligned = (
                torch.flip(prediction, dims=dimensions)
                if dimensions
                else prediction
            )
            probabilities.append(aligned)
    return torch.stack(probabilities, dim=0).mean(dim=0)


def _size_group(fraction):
    if fraction == 0:
        return "empty"
    if fraction <= 0.02:
        return "tiny"
    if fraction <= 0.05:
        return "medium"
    return "large"


def _select_threshold(predictions, targets):
    coarse = [round(0.1 + step * 0.025, 3) for step in range(35)]
    scores = {
        threshold: compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        for threshold in coarse
    }
    center = max(coarse, key=lambda threshold: scores[threshold]["dice"])
    fine = {
        round(center - 0.025 + step * 0.005, 3)
        for step in range(11)
    }
    for threshold in fine:
        if threshold not in scores:
            scores[threshold] = compute_detailed_metrics(
                predictions,
                targets,
                threshold=threshold,
            )
    selected = max(scores, key=lambda threshold: scores[threshold]["dice"])
    return selected, scores


def _predict_dataset(model, images, transforms, batch_size):
    predictions = []
    started = time.perf_counter()
    for offset in range(0, len(images), batch_size):
        predictions.append(
            predict_with_tta(
                model,
                images[offset : offset + batch_size],
                transforms,
            ).cpu()
        )
    elapsed = time.perf_counter() - started
    if not predictions:
        raise ValueError("Cannot infer an empty validation dataset")
    return torch.cat(predictions), elapsed / len(images)


def _per_image_rows(name, predictions, targets, threshold, image_names):
    rows = []
    for index, image_name in enumerate(image_names):
        target = targets[index, 0].numpy() > 0.5
        predicted = predictions[index, 0].numpy() > threshold
        true_positive = int((predicted & target).sum())
        false_positive = int((predicted & ~target).sum())
        false_negative = int((~predicted & target).sum())
        ground_truth_pixels = int(target.sum())
        component_count = 0
        if predicted.any():
            component_count = int(
                cv2.connectedComponents(
                    predicted.astype(np.uint8),
                    connectivity=8,
                )[0]
                - 1
            )
        rows.append(
            {
                "configuration": name,
                "image": image_name,
                "size_group": _size_group(ground_truth_pixels / target.size),
                "ground_truth_pixels": ground_truth_pixels,
                "true_positive_pixels": true_positive,
                "false_positive_pixels": false_positive,
                "false_negative_pixels": false_negative,
                "predicted_defect_pixels": true_positive + false_positive,
                "predicted_components": component_count,
                "dice": 2
                * true_positive
                / max(
                    1,
                    2 * true_positive + false_positive + false_negative,
                ),
                "iou": true_positive
                / max(
                    1,
                    true_positive + false_positive + false_negative,
                ),
                "precision": true_positive
                / max(1, true_positive + false_positive),
                "recall": true_positive
                / max(1, true_positive + false_negative),
                "completely_missed": int(
                    ground_truth_pixels > 0 and true_positive == 0
                ),
            }
        )
    return rows


def _group_rows(configuration, per_image):
    rows = []
    for group in SIZE_GROUPS:
        selected = [
            row
            for row in per_image
            if row["configuration"] == configuration
            and row["size_group"] == group
        ]
        result = {
            "configuration": configuration,
            "images": len(selected),
            "size_group": group,
        }
        for metric in ("dice", "iou", "precision", "recall"):
            result[f"mean_{metric}"] = (
                float(np.mean([row[metric] for row in selected]))
                if selected
                else ""
            )
        result["false_positive_pixels"] = sum(
            row["false_positive_pixels"] for row in selected
        )
        result["false_negative_pixels"] = sum(
            row["false_negative_pixels"] for row in selected
        )
        result["completely_missed_defects"] = sum(
            row["completely_missed"] for row in selected
        )
        rows.append(result)
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _qualitative_examples(baseline_rows):
    positives = [
        row for row in baseline_rows if row["ground_truth_pixels"] > 0
    ]
    tiny = [row for row in positives if row["size_group"] == "tiny"]
    small_or_medium = [
        row
        for row in positives
        if row["size_group"] in ("tiny", "medium")
    ]
    selected = []
    if positives:
        selected.append(
            (
                min(positives, key=lambda row: row["ground_truth_pixels"]),
                "Smallest validation defect",
            )
        )
    if tiny:
        selected.append(
            (min(tiny, key=lambda row: row["dice"]), "Lowest tiny-defect Dice")
        )
    if small_or_medium:
        selected.append(
            (
                min(small_or_medium, key=lambda row: row["recall"]),
                "Lowest small/thin recall",
            )
        )
    examples = []
    labels = []
    for row, label in selected:
        if row["image"] not in examples:
            examples.append(row["image"])
            labels.append(label)
    return examples, labels


def _save_qualitative(
    path,
    project_dir,
    image_names,
    targets,
    predictions,
    thresholds,
    per_image,
):
    baseline_rows = [
        row for row in per_image if row["configuration"] == "A_original"
    ]
    examples, labels = _qualitative_examples(baseline_rows)
    report_path = (
        project_dir
        / "results"
        / "preprocessing"
        / "preprocessing_report.csv"
    )
    with report_path.open(newline="", encoding="utf-8-sig") as source:
        original_images = {
            Path(row["image"]).stem: row["image"]
            for row in csv.DictReader(source)
            if row["split"] == "train" and row["status"] == "ok"
        }

    configurations = tuple(TTA_CONFIGURATIONS)
    rows = []
    for image_name, label in zip(examples, labels):
        for configuration in configurations:
            rows.append((image_name, label, configuration))
    figure, axes = plt.subplots(
        len(rows),
        8,
        figsize=(25, 3.1 * len(rows)),
        squeeze=False,
    )
    headings = (
        "Original EL image",
        "Ground truth",
        "Baseline probability",
        "Baseline mask",
        "Averaged probability",
        "TTA mask",
        "Baseline errors",
        "TTA errors",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)

    for row_index, (image_name, reason, configuration) in enumerate(rows):
        index = image_names.index(image_name)
        raw_relative = original_images.get(Path(image_name).stem)
        if raw_relative is None:
            raise FileNotFoundError(
                f"No original image recorded for {image_name}"
            )
        raw_path = project_dir / "data" / "SolarCells" / Path(raw_relative)
        with Image.open(raw_path) as source_image:
            original = np.asarray(
                ImageOps.exif_transpose(source_image).convert("RGB")
            )
        target = targets[index, 0].numpy() > 0.5
        baseline_probability = predictions["A_original"][index, 0].numpy()
        tta_probability = predictions[configuration][index, 0].numpy()
        baseline_mask = baseline_probability > thresholds["A_original"]
        tta_mask = tta_probability > thresholds[configuration]
        height, width = original.shape[:2]

        def display_mask(mask):
            return np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                )
            ) > 0

        target_display = display_mask(target)
        baseline_display = display_mask(baseline_mask)
        tta_display = display_mask(tta_mask)
        axes[row_index, 0].imshow(original)
        axes[row_index, 1].imshow(
            target_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 2].imshow(
            baseline_probability,
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 3].imshow(
            baseline_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 4].imshow(
            tta_probability,
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        axes[row_index, 5].imshow(
            tta_display,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        for column, predicted in ((6, baseline_display), (7, tta_display)):
            true_positive = target_display & predicted
            false_positive = ~target_display & predicted
            false_negative = target_display & ~predicted
            overlay = np.zeros((*target_display.shape, 4), dtype=np.float32)
            overlay[true_positive] = (0.1, 1.0, 0.1, 0.8)
            overlay[false_positive] = (1.0, 0.1, 0.1, 0.85)
            overlay[false_negative] = (0.1, 0.75, 1.0, 0.85)
            axes[row_index, column].imshow(original)
            axes[row_index, column].imshow(overlay)

        axes[row_index, 0].set_ylabel(
            f"{image_name}: {reason}\n{configuration}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.axis("off")

    figure.suptitle("Green: TP | Red: FP | Blue: FN", y=1.002)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_tta(
    project_dir,
    checkpoint_path,
    output_dir,
    batch_size=8,
    num_threads=6,
):
    if batch_size <= 0 or num_threads <= 0:
        raise ValueError("batch_size and num_threads must be greater than zero")
    torch.set_num_threads(num_threads)
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    validation_subset = validation_loader.dataset
    image_names = [
        validation_subset.dataset.samples[index][0].name
        for index in validation_subset.indices
    ]
    all_images = []
    all_targets = []
    for images, masks, _ in validation_loader:
        all_images.append(images)
        all_targets.append(masks)
    images = torch.cat(all_images)
    targets = torch.cat(all_targets)
    if len(image_names) != len(images):
        raise RuntimeError("Validation image order does not match the data loader")

    state, checkpoint_metadata = _load_state(checkpoint_path)
    if (
        checkpoint_metadata.get("negative_bce_weight") != 1.1
        or float(checkpoint_metadata.get("boundary_weight") or 0.0) != 0.0
        or (
            checkpoint_metadata.get("sampling") or {}
        ).get("strategy", "shuffle")
        != "shuffle"
    ):
        raise ValueError("The selected checkpoint is not the Phase 5 checkpoint")
    model = ResNet18UNet(pretrained=False).cpu()
    model.load_state_dict(state)
    model.eval()
    if model.training:
        raise RuntimeError("Model did not enter evaluation mode")

    predictions_by_configuration = {}
    summary_rows = []
    per_image = []
    threshold_rows = []
    group_rows = []
    for configuration, transforms in TTA_CONFIGURATIONS.items():
        warmup = images[: min(batch_size, len(images))]
        predict_with_tta(model, warmup, transforms)
        predictions, seconds_per_image = _predict_dataset(
            model,
            images,
            transforms,
            batch_size,
        )
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"{configuration} predictions are misaligned with targets: "
                f"{predictions.shape} != {targets.shape}"
            )
        predictions_by_configuration[configuration] = predictions
        threshold, scores = _select_threshold(predictions, targets)
        metrics = compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        rows = _per_image_rows(
            configuration,
            predictions,
            targets,
            threshold,
            image_names,
        )
        per_image.extend(rows)
        groups = {
            row["size_group"]: row
            for row in _group_rows(configuration, rows)
        }
        summary_rows.append(
            {
                "configuration": configuration,
                "transforms": json.dumps(transforms),
                "selected_threshold": threshold,
                "validation_dice": metrics["dice"],
                "validation_iou": metrics["iou"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_boundary_f1_tolerance_1px": _boundary_f1(
                    predictions,
                    targets,
                    threshold,
                ),
                "tiny_images": groups["tiny"]["images"],
                "tiny_dice": groups["tiny"]["mean_dice"],
                "medium_dice": groups["medium"]["mean_dice"],
                "large_dice": groups["large"]["mean_dice"],
                "false_positive_pixels": sum(
                    row["false_positive_pixels"] for row in rows
                ),
                "false_negative_pixels": sum(
                    row["false_negative_pixels"] for row in rows
                ),
                "completely_missed_defects": sum(
                    row["completely_missed"] for row in rows
                ),
                "inference_seconds_per_image": seconds_per_image,
                "checkpoint": str(checkpoint_path),
            }
        )
        for candidate, candidate_metrics in sorted(scores.items()):
            threshold_rows.append(
                {
                    "configuration": configuration,
                    "threshold": candidate,
                    **candidate_metrics,
                    "selected": candidate == threshold,
                }
            )
        group_rows.extend(_group_rows(configuration, rows))
        print(
            f"{configuration}: threshold={threshold:.3f}, "
            f"Dice={metrics['dice']:.4f}, tiny Dice={groups['tiny']['mean_dice']}, "
            f"Boundary F1={summary_rows[-1]['validation_boundary_f1_tolerance_1px']:.4f}, "
            f"inference={seconds_per_image:.4f}s/image"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "phase8_validation_tta_comparison.csv", summary_rows)
    _write_csv(output_dir / "phase8_validation_tta_thresholds.csv", threshold_rows)
    _write_csv(output_dir / "phase8_validation_tta_size_groups.csv", group_rows)
    _write_csv(output_dir / "phase8_validation_tta_per_image.csv", per_image)
    _save_qualitative(
        output_dir / "phase8_validation_tta_qualitative.png",
        project_dir,
        image_names,
        targets,
        predictions_by_configuration,
        {
            row["configuration"]: row["selected_threshold"]
            for row in summary_rows
        },
        per_image,
    )
    print(f"Saved Phase 8 validation TTA outputs under {output_dir}")
    return summary_rows


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate flip-based TTA on the validation split only."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    evaluate_tta(
        project_dir,
        args.checkpoint,
        args.output_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
