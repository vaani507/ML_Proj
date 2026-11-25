import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.compare_models import IMAGE_MEAN, IMAGE_STD
from src.dataset import get_dataloaders
from src.inference import (
    DEFAULT_THRESHOLD,
    PHASE5_CHECKPOINT_SHA256,
    load_model,
    model_checkpoint_path,
)
from src.train import compute_detailed_metrics


EXAMPLE_COUNT = 8
OUTPUT_NAMES = (
    "phase12_final_test_metrics.json",
    "phase12_final_test_per_image.csv",
    "phase12_final_test_examples.csv",
    "phase12_final_test_examples.png",
)


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _select_examples(rows, count=EXAMPLE_COUNT):
    if not rows:
        raise ValueError("No test predictions are available for example selection")
    ordered = [
        (min(rows, key=lambda row: row["dice"]), "Lowest Dice"),
        (min(rows, key=lambda row: row["recall"]), "Lowest recall"),
        (
            max(rows, key=lambda row: row["false_positive_pixels"]),
            "Most false-positive pixels",
        ),
        (
            min(rows, key=lambda row: row["ground_truth_pixels"]),
            "Smallest ground-truth defect",
        ),
        (
            sorted(rows, key=lambda row: row["dice"])[len(rows) // 2],
            "Median Dice",
        ),
        (max(rows, key=lambda row: row["dice"]), "Highest Dice"),
        (
            max(rows, key=lambda row: row["false_negative_pixels"]),
            "Most false-negative pixels",
        ),
        (
            max(rows, key=lambda row: row["ground_truth_pixels"]),
            "Largest ground-truth defect",
        ),
    ]
    selected = []
    seen = set()
    for row, label in ordered:
        if row["image"] in seen:
            continue
        selected.append((row, label))
        seen.add(row["image"])
        if len(selected) == count:
            break
    return selected


def _error_overlay(image, prediction, target):
    colors = np.zeros_like(image)
    colors[prediction & target] = (0.0, 1.0, 0.0)
    colors[prediction & ~target] = (1.0, 0.0, 0.0)
    colors[~prediction & target] = (0.0, 0.35, 1.0)
    selected = prediction | target
    overlay = image.copy()
    overlay[selected] = 0.45 * image[selected] + 0.55 * colors[selected]
    return overlay


def _save_examples(path, images, targets, predictions, examples):
    figure, axes = plt.subplots(
        len(examples),
        4,
        figsize=(15, 3.25 * len(examples)),
        squeeze=False,
    )
    headings = (
        "Original EL image",
        "Ground truth",
        f"Phase 5 prediction (threshold {DEFAULT_THRESHOLD:.3f})",
        "Errors: green TP | red FP | blue FN",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)

    for row_index, (row, label) in enumerate(examples):
        image_index = row["index"]
        normalized = images[image_index].numpy().transpose(1, 2, 0)
        rgb = np.clip(normalized * IMAGE_STD + IMAGE_MEAN, 0, 1)
        target = targets[image_index, 0].numpy() > 0.5
        prediction = predictions[image_index, 0].numpy() > DEFAULT_THRESHOLD
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(
            _error_overlay(rgb, prediction, target),
            vmin=0,
            vmax=1,
        )
        axes[row_index, 0].set_ylabel(
            f"{row['image']} | {label}\n"
            f"Dice {row['dice']:.3f} | Recall {row['recall']:.3f}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.axis("off")

    figure.tight_layout(rect=(0, 0, 1, 0.99))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _collect_test_predictions(model, test_loader, threshold):
    if len(test_loader) == 0:
        raise ValueError("The test data loader is empty")
    predictions = []
    targets = []
    images = []
    elapsed = 0.0
    image_count = 0
    model.eval()
    with torch.inference_mode():
        for batch_images, batch_masks, _ in test_loader:
            started = time.perf_counter()
            probabilities = model(batch_images)
            elapsed += time.perf_counter() - started
            image_count += len(batch_images)
            predictions.append(probabilities.cpu())
            targets.append(batch_masks.cpu())
            images.append(batch_images.cpu())
    if image_count == 0:
        raise ValueError("The test data loader produced no samples")
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    image_tensor = torch.cat(images)
    binary = prediction_tensor[:, 0].numpy() > threshold
    truth = target_tensor[:, 0].numpy() > 0.5
    per_image = []
    for index, (prediction, target) in enumerate(zip(binary, truth)):
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        tn = int(np.count_nonzero(~prediction & ~target))
        per_image.append(
            {
                "index": index,
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "true_negative_pixels": tn,
                "ground_truth_pixels": int(target.sum()),
                "predicted_pixels": int(prediction.sum()),
                "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "iou": tp / max(1, tp + fp + fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
            }
        )
    metrics = compute_detailed_metrics(
        prediction_tensor,
        target_tensor,
        threshold=threshold,
    )
    true_positive = int(np.count_nonzero(binary & truth))
    false_positive = int(np.count_nonzero(binary & ~truth))
    false_negative = int(np.count_nonzero(~binary & truth))
    true_negative = int(np.count_nonzero(~binary & ~truth))
    metrics.update(
        {
            "true_positive_pixels": true_positive,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": false_negative,
            "true_negative_pixels": true_negative,
            "ground_truth_defect_pixels": int(truth.sum()),
            "predicted_defect_pixels": int(binary.sum()),
            "micro_dice": 2 * true_positive
            / max(1, 2 * true_positive + false_positive + false_negative),
            "inference_seconds_per_image": elapsed / image_count,
        }
    )
    return prediction_tensor, target_tensor, image_tensor, per_image, metrics


def _read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def _recover_partial_outputs(project_dir, output_paths, batch_size):
    partial_paths = {output_paths[1], output_paths[2]}
    if not all(path.is_file() for path in partial_paths):
        raise FileExistsError(
            "Some final test outputs already exist, but the saved per-image "
            "metrics and example selection are unavailable. Refusing to "
            "evaluate the test set again."
        )
    unexpected = [
        path
        for path in (output_paths[0], output_paths[3])
        if path.exists()
    ]
    if unexpected:
        raise FileExistsError(
            "Final test outputs are already present; refusing to overwrite "
            "or evaluate again."
        )

    raw_per_image = _read_csv(output_paths[1])
    if not raw_per_image:
        raise ValueError("The saved test per-image results are empty")
    numeric_fields = (
        "true_positive_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
        "true_negative_pixels",
        "ground_truth_pixels",
        "predicted_pixels",
        "dice",
        "iou",
        "precision",
        "recall",
    )
    per_image = []
    for raw_row in raw_per_image:
        row = dict(raw_row)
        for field in numeric_fields:
            row[field] = (
                float(row[field])
                if field in {"dice", "iou", "precision", "recall"}
                else int(row[field])
            )
        per_image.append(row)

    sums = {
        field: sum(row[field] for row in per_image)
        for field in numeric_fields[:6]
    }
    mean_metrics = {
        field: float(np.mean([row[field] for row in per_image]))
        for field in ("dice", "iou", "precision", "recall")
    }
    mean_metrics["f1"] = mean_metrics["dice"]
    mean_metrics.update(
        {
            "true_positive_pixels": sums["true_positive_pixels"],
            "false_positive_pixels": sums["false_positive_pixels"],
            "false_negative_pixels": sums["false_negative_pixels"],
            "true_negative_pixels": sums["true_negative_pixels"],
            "ground_truth_defect_pixels": sums["ground_truth_pixels"],
            "predicted_defect_pixels": sums["predicted_pixels"],
            "micro_dice": 2 * sums["true_positive_pixels"]
            / max(
                1,
                2 * sums["true_positive_pixels"]
                + sums["false_positive_pixels"]
                + sums["false_negative_pixels"],
            ),
        }
    )
    raw_examples = _read_csv(output_paths[2])
    if not 5 <= len(raw_examples) <= 10:
        raise ValueError("Saved test example list must contain 5-10 images")
    examples = []
    for raw_example in raw_examples:
        example = dict(raw_example)
        for field in ("dice", "iou", "precision", "recall"):
            example[field] = float(example[field])
        for field in ("ground_truth_pixels", "predicted_pixels"):
            example[field] = int(example[field])
        examples.append(example)

    checkpoint_path = model_checkpoint_path("phase5_resnet18", project_dir)
    model = load_model("phase5_resnet18", project_dir)
    if next(model.parameters()).device.type != "cpu" or model.training:
        raise RuntimeError("The application model must load in CPU eval mode")
    _, _, test_loader = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    dataset = test_loader.dataset
    sample_indices = {
        image_path.name: index
        for index, (image_path, _) in enumerate(dataset.samples)
    }
    selected_images = []
    selected_targets = []
    selected_rows = []
    for index, example in enumerate(examples):
        image_name = example["image"]
        if image_name not in sample_indices:
            raise RuntimeError(f"Saved example is not in the test split: {image_name}")
        image, mask, _ = dataset[sample_indices[image_name]]
        selected_images.append(image)
        selected_targets.append(mask)
        selected_rows.append(
            {
                **example,
                "index": index,
            }
        )
    batch_images = torch.stack(selected_images)
    with torch.inference_mode():
        selected_predictions = model(batch_images).cpu()
    selected_targets_tensor = torch.stack(selected_targets)
    _save_examples(
        output_paths[3],
        batch_images,
        selected_targets_tensor,
        selected_predictions,
        [
            (row, row["selection_reason"])
            for row in selected_rows
        ],
    )

    result = {
        "model": "Phase 5 ResNet-18 U-Net",
        "checkpoint": str(checkpoint_path.relative_to(project_dir)),
        "checkpoint_sha256": PHASE5_CHECKPOINT_SHA256,
        "threshold": DEFAULT_THRESHOLD,
        "threshold_source": "Fixed validation-selected Phase 5 threshold; no test tuning",
        "test_images": len(per_image),
        "metric_aggregation": (
            "Per-image macro Dice, IoU, precision, recall; pixel totals and "
            "micro Dice are also reported"
        ),
        "metrics": mean_metrics,
        "example_images": examples,
        "test_set_evaluated": True,
        "full_test_inference_passes": 1,
        "qualitative_recovery_images": len(examples),
    }
    output_paths[0].write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def run_evaluation(project_dir, batch_size=8, num_threads=6):
    if batch_size <= 0 or num_threads <= 0:
        raise ValueError("batch_size and num_threads must be greater than zero")
    torch.set_num_threads(num_threads)
    project_dir = Path(project_dir).resolve()
    output_dir = project_dir / "results"
    output_paths = [output_dir / name for name in OUTPUT_NAMES]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        if all(path in existing for path in (output_paths[1], output_paths[2])):
            return _recover_partial_outputs(
                project_dir,
                output_paths,
                batch_size,
            )
        raise FileExistsError(
            "Final test outputs already exist; refusing to evaluate the test "
            f"set again: {', '.join(str(path) for path in existing)}"
        )

    checkpoint_path = model_checkpoint_path("phase5_resnet18", project_dir)
    model = load_model("phase5_resnet18", project_dir)
    if next(model.parameters()).device.type != "cpu" or model.training:
        raise RuntimeError("The application model must load in CPU eval mode")

    _, _, test_loader = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    dataset = test_loader.dataset
    if not hasattr(dataset, "samples"):
        raise TypeError("Expected the test loader dataset to expose paired samples")
    names = [image_path.name for image_path, _ in dataset.samples]

    predictions, targets, images, per_image, metrics = _collect_test_predictions(
        model,
        test_loader,
        DEFAULT_THRESHOLD,
    )
    if len(names) != len(per_image):
        raise RuntimeError("Test image names do not align with model predictions")
    for image_name, row in zip(names, per_image):
        row["image"] = image_name

    examples = _select_examples(per_image)
    if len(examples) < 5:
        raise RuntimeError("Expected at least five distinct test examples")
    example_rows = [
        {
            "image": row["image"],
            "selection_reason": label,
            "dice": row["dice"],
            "iou": row["iou"],
            "precision": row["precision"],
            "recall": row["recall"],
            "ground_truth_pixels": row["ground_truth_pixels"],
            "predicted_pixels": row["predicted_pixels"],
        }
        for row, label in examples
    ]
    result = {
        "model": "Phase 5 ResNet-18 U-Net",
        "checkpoint": str(checkpoint_path.relative_to(project_dir)),
        "checkpoint_sha256": PHASE5_CHECKPOINT_SHA256,
        "threshold": DEFAULT_THRESHOLD,
        "threshold_source": "Fixed validation-selected Phase 5 threshold; no test tuning",
        "test_images": len(names),
        "metric_aggregation": (
            "Per-image macro Dice, IoU, precision, recall; pixel totals and "
            "micro Dice are also reported"
        ),
        "metrics": metrics,
        "example_images": example_rows,
        "test_set_evaluated": True,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_per_image = [
        {key: value for key, value in row.items() if key != "index"}
        for row in per_image
    ]
    _write_csv(output_paths[1], csv_per_image)
    _write_csv(output_paths[2], example_rows)
    _save_examples(output_paths[3], images, targets, predictions, examples)
    output_paths[0].write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="One fixed-threshold final test evaluation of Phase 5."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    run_evaluation(args.project_dir, args.batch_size, args.num_threads)


if __name__ == "__main__":
    main()
