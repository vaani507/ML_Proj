import argparse
import csv
import json
from pathlib import Path
import time

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import Subset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.dataset import get_dataloaders
from src.models import ResNet18UNet
from src.phase5_analysis import _dark_linear_structure
from src.phase7_analysis import _boundary_f1
from src.phase8_tta import _select_threshold
from src.train import compute_detailed_metrics


def _image_names(loader):
    subset = loader.dataset
    if not isinstance(subset, Subset):
        raise TypeError("Expected the validation loader to use a dataset subset")
    return [
        subset.dataset.samples[index][0].name
        for index in subset.indices
    ]


def _predict(model, loader):
    predictions = []
    targets = []
    images_for_visualization = []
    elapsed = 0.0
    image_count = 0
    model.eval()
    with torch.inference_mode():
        for images, masks, _ in loader:
            started = time.perf_counter()
            outputs = model(images)
            elapsed += time.perf_counter() - started
            image_count += len(images)
            predictions.append(outputs.cpu())
            targets.append(masks)
            images_for_visualization.append(images)
    if image_count == 0:
        raise ValueError("The validation data loader is empty")
    return (
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(images_for_visualization),
        elapsed / image_count,
    )


def _load_model(checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = ResNet18UNet(pretrained=False)
    state, metadata = _load_state(checkpoint_path)
    model.load_state_dict(state)
    model.eval()
    return model, metadata


def _size_group(fraction):
    if fraction == 0:
        return "empty"
    if fraction <= 0.02:
        return "tiny"
    if fraction <= 0.05:
        return "medium"
    return "large"


def _per_image_rows(
    model_name,
    predictions,
    targets,
    images,
    threshold,
    image_names,
):
    rows = []
    for index, image_name in enumerate(image_names):
        target = targets[index, 0].numpy() > 0.5
        prediction = predictions[index, 0].numpy()
        predicted = prediction > threshold
        true_positive = int((predicted & target).sum())
        false_positive_mask = predicted & ~target
        false_positive = int(false_positive_mask.sum())
        false_negative = int((~predicted & target).sum())
        ground_truth_pixels = int(target.sum())

        rgb = images[index].numpy().transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN
        rgb = np.clip(rgb, 0, 1)
        gray = cv2.cvtColor(
            np.round(rgb * 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        linear_proxy = _dark_linear_structure(gray)
        proxy_false_positives = int((false_positive_mask & linear_proxy).sum())
        rows.append(
            {
                "model": model_name,
                "image": image_name,
                "size_group": _size_group(ground_truth_pixels / target.size),
                "threshold": threshold,
                "ground_truth_pixels": ground_truth_pixels,
                "true_positive_pixels": true_positive,
                "false_positive_pixels": false_positive,
                "false_negative_pixels": false_negative,
                "predicted_defect_pixels": true_positive + false_positive,
                "predicted_to_ground_truth_area_ratio": (
                    (true_positive + false_positive) / ground_truth_pixels
                    if ground_truth_pixels
                    else ""
                ),
                "dice": 2 * true_positive
                / max(1, 2 * true_positive + false_positive + false_negative),
                "iou": true_positive
                / max(1, true_positive + false_positive + false_negative),
                "precision": true_positive / max(1, true_positive + false_positive),
                "recall": true_positive / max(1, true_positive + false_negative),
                "completely_missed": int(
                    ground_truth_pixels > 0 and true_positive == 0
                ),
                "fp_on_dark_linear_proxy": proxy_false_positives,
                "fp_on_proxy_fraction": (
                    proxy_false_positives / false_positive
                    if false_positive
                    else 0.0
                ),
                "mean_probability_on_defect": (
                    float(prediction[target].mean()) if target.any() else ""
                ),
                "mean_probability_on_background": float(prediction[~target].mean()),
            }
        )
    return rows


def _group_rows(per_image_rows):
    rows = []
    for model_name in dict.fromkeys(row["model"] for row in per_image_rows):
        model_rows = [row for row in per_image_rows if row["model"] == model_name]
        for group in ("tiny", "medium", "large", "empty"):
            selected = [row for row in model_rows if row["size_group"] == group]
            row = {
                "model": model_name,
                "size_group": group,
                "images": len(selected),
            }
            for metric in ("dice", "iou", "precision", "recall"):
                row[f"mean_{metric}"] = (
                    float(np.mean([item[metric] for item in selected]))
                    if selected
                    else ""
                )
            for metric in (
                "true_positive_pixels",
                "false_positive_pixels",
                "false_negative_pixels",
                "fp_on_dark_linear_proxy",
                "completely_missed",
            ):
                row[metric] = sum(item[metric] for item in selected)
            area_ratios = [
                item["predicted_to_ground_truth_area_ratio"]
                for item in selected
                if item["predicted_to_ground_truth_area_ratio"] != ""
            ]
            row["mean_predicted_to_ground_truth_area_ratio"] = (
                float(np.mean(area_ratios)) if area_ratios else ""
            )
            rows.append(row)
    return rows


def _probability_rows(model_name, predictions, targets):
    rows = []
    probability_values = predictions[:, 0].numpy()
    target_values = targets[:, 0].numpy() > 0.5
    for class_name, selected in (
        ("ground_truth_defect", target_values),
        ("ground_truth_background", ~target_values),
    ):
        values = probability_values[selected]
        if not values.size:
            raise ValueError(f"No pixels available for {class_name}")
        quantiles = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))
        rows.append(
            {
                "model": model_name,
                "pixel_class": class_name,
                "pixel_count": int(values.size),
                "mean": float(values.mean()),
                "q00": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "q50": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "q90": float(quantiles[4]),
                "q99": float(quantiles[5]),
                "q100": float(quantiles[6]),
            }
        )
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _overall_row(
    model_name,
    predictions,
    targets,
    threshold,
    inference_seconds,
    metadata,
    per_image_rows,
):
    metrics = compute_detailed_metrics(predictions, targets, threshold)
    binary = predictions > threshold
    target = targets > 0.5
    tiny_indices = [
        index
        for index, row in enumerate(per_image_rows)
        if row["size_group"] == "tiny"
    ]
    tiny_dice = (
        compute_detailed_metrics(
            predictions[tiny_indices],
            targets[tiny_indices],
            threshold,
        )["dice"]
        if tiny_indices
        else None
    )
    false_positive = int((binary & ~target).sum())
    predicted_pixels = int(binary.sum())
    ground_truth_pixels = int(target.sum())
    return {
        "model": model_name,
        "threshold": threshold,
        **metrics,
        "tiny_defect_dice": tiny_dice,
        "boundary_f1_tolerance_1px": _boundary_f1(
            predictions,
            targets,
            threshold,
            tolerance=1,
        ),
        "true_positive_pixels": int((binary & target).sum()),
        "false_positive_pixels": false_positive,
        "false_negative_pixels": int((~binary & target).sum()),
        "predicted_to_ground_truth_area_ratio": (
            predicted_pixels / ground_truth_pixels
            if ground_truth_pixels
            else ""
        ),
        "fp_on_dark_linear_proxy": sum(
            row["fp_on_dark_linear_proxy"] for row in per_image_rows
        ),
        "fp_on_proxy_fraction": (
            sum(row["fp_on_dark_linear_proxy"] for row in per_image_rows)
            / false_positive
            if false_positive
            else 0.0
        ),
        "training_seconds": metadata.get("training_seconds", ""),
        "inference_seconds_per_image": inference_seconds,
    }


def _qualitative_examples(rows):
    baseline = {row["image"]: row for row in rows if row["model"] == "Phase 5"}
    fine_tuned = {
        row["image"]: row for row in rows if row["model"] == "Phase 9 layer4"
    }
    candidates = []
    positive = [row for row in baseline.values() if row["ground_truth_pixels"]]
    if positive:
        candidates.append(
            (min(positive, key=lambda row: row["ground_truth_pixels"]), "Smallest defect")
        )
        candidates.append((min(positive, key=lambda row: row["dice"]), "Lowest baseline Dice"))
        changes = [
            (fine_tuned[row["image"]]["dice"] - row["dice"], row)
            for row in positive
        ]
        candidates.append((max(changes, key=lambda item: item[0])[1], "Largest Dice gain"))
        candidates.append((min(changes, key=lambda item: item[0])[1], "Largest Dice loss"))
    candidates.append(
        (
            max(
                baseline.values(),
                key=lambda row: row["fp_on_dark_linear_proxy"],
            ),
            "Most baseline FP on linear proxy",
        )
    )
    chosen = []
    for row, label in candidates:
        if row["image"] not in [item[0] for item in chosen]:
            chosen.append((row["image"], label))
    return chosen[:5]


def _error_overlay(image, prediction, target, threshold):
    predicted = prediction > threshold
    target = target > 0.5
    overlay = image.copy()
    colors = np.zeros_like(overlay)
    colors[predicted & target] = (0.0, 1.0, 0.0)
    colors[predicted & ~target] = (1.0, 0.0, 0.0)
    colors[~predicted & target] = (0.0, 0.35, 1.0)
    selected = predicted | target
    overlay[selected] = 0.45 * overlay[selected] + 0.55 * colors[selected]
    return overlay


def _save_qualitative(
    path,
    image_names,
    images,
    targets,
    predictions,
    thresholds,
    per_image,
):
    examples = _qualitative_examples(per_image)
    if not examples:
        raise ValueError("No validation images are available for visualization")
    figure, axes = plt.subplots(
        len(examples),
        8,
        figsize=(27, 3.8 * len(examples)),
        squeeze=False,
    )
    headings = (
        "EL input",
        "Ground truth",
        "Phase 5 probability",
        "Phase 5 mask",
        "Fine-tuned probability",
        "Fine-tuned mask",
        "Phase 5 errors",
        "Fine-tuned errors",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)

    for row_index, (image_name, reason) in enumerate(examples):
        index = image_names.index(image_name)
        image = np.clip(
            images[index].numpy().transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN,
            0,
            1,
        )
        target = targets[index, 0].numpy()
        phase5_probability = predictions["Phase 5"][index, 0].numpy()
        fine_probability = predictions["Phase 9 layer4"][index, 0].numpy()
        phase5_mask = phase5_probability > thresholds["Phase 5"]
        fine_mask = fine_probability > thresholds["Phase 9 layer4"]
        panels = (
            image,
            target,
            phase5_probability,
            phase5_mask,
            fine_probability,
            fine_mask,
            _error_overlay(
                image,
                phase5_probability,
                target,
                thresholds["Phase 5"],
            ),
            _error_overlay(
                image,
                fine_probability,
                target,
                thresholds["Phase 9 layer4"],
            ),
        )
        cmaps = (
            None,
            "gray",
            "magma",
            "gray",
            "magma",
            "gray",
            None,
            None,
        )
        for column, (panel, cmap) in enumerate(zip(panels, cmaps)):
            axes[row_index, column].imshow(panel, cmap=cmap, vmin=0, vmax=1)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(reason, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_history(history, output_dir):
    if not history:
        raise ValueError("Fine-tuning checkpoint has no training history")
    rows = [
        {
            "phase": item["phase"],
            "epoch": item["epoch"],
            "train_loss": item["train_loss"],
            "validation_loss": item["validation_loss"],
            "validation_dice": item["validation"]["dice"],
            "validation_iou": item["validation"]["iou"],
            "validation_precision": item["validation"]["precision"],
            "validation_recall": item["validation"]["recall"],
            "threshold": item["threshold"],
            "epoch_seconds": item["epoch_seconds"],
        }
        for item in history
    ]
    _write_csv(output_dir / "phase9_layer4_training_history.csv", rows)
    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label="train")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in rows],
        marker="o",
        label="validation",
    )
    axes[0].set_title("BCE + Dice loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["validation_dice"] for row in rows],
        marker="o",
    )
    axes[1].set_title("Validation Dice (per-epoch coarse threshold)")
    axes[1].set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(output_dir / "phase9_layer4_training_curves.png", dpi=160)
    plt.close(figure)


def run_analysis(
    project_dir,
    baseline_path,
    fine_tuned_path,
    batch_size=8,
    num_threads=6,
):
    torch.set_num_threads(num_threads)
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    image_names = _image_names(validation_loader)
    targets = None
    images = None
    prediction_sets = {}
    metadata_by_model = {}
    inference_times = {}
    checkpoint_paths = (
        ("Phase 5", baseline_path),
        ("Phase 9 layer4", fine_tuned_path),
    )
    for model_name, checkpoint_path in checkpoint_paths:
        model, metadata = _load_model(checkpoint_path)
        predictions, model_targets, model_images, inference_time = _predict(
            model,
            validation_loader,
        )
        if targets is None:
            targets = model_targets
            images = model_images
        elif not torch.equal(targets, model_targets):
            raise RuntimeError("The evaluated checkpoints used different targets")
        prediction_sets[model_name] = predictions
        metadata_by_model[model_name] = metadata
        inference_times[model_name] = inference_time

    output_dir = project_dir / "results"
    overall_rows = []
    threshold_rows = []
    per_image = []
    distributions = []
    thresholds = {}
    for model_name, predictions in prediction_sets.items():
        threshold, scores = _select_threshold(predictions, targets)
        thresholds[model_name] = threshold
        threshold_rows.extend(
            {
                "model": model_name,
                "threshold": candidate,
                **metrics,
                "selected": int(candidate == threshold),
            }
            for candidate, metrics in sorted(scores.items())
        )
        model_rows = _per_image_rows(
            model_name,
            predictions,
            targets,
            images,
            threshold,
            image_names,
        )
        per_image.extend(model_rows)
        overall_rows.append(
            _overall_row(
                model_name,
                predictions,
                targets,
                threshold,
                inference_times[model_name],
                metadata_by_model[model_name],
                model_rows,
            )
        )
        distributions.extend(_probability_rows(model_name, predictions, targets))

    _write_csv(output_dir / "phase9_layer4_validation_comparison.csv", overall_rows)
    _write_csv(output_dir / "phase9_layer4_validation_thresholds.csv", threshold_rows)
    _write_csv(output_dir / "phase9_layer4_validation_per_image.csv", per_image)
    _write_csv(output_dir / "phase9_layer4_validation_size_groups.csv", _group_rows(per_image))
    _write_csv(
        output_dir / "phase9_layer4_probability_distributions.csv",
        distributions,
    )
    _save_qualitative(
        output_dir / "phase9_layer4_validation_qualitative.png",
        image_names,
        images,
        targets,
        prediction_sets,
        thresholds,
        per_image,
    )
    _save_history(metadata_by_model["Phase 9 layer4"].get("history"), output_dir)
    summary = {
        "validation_images": len(image_names),
        "threshold_selection": "validation Dice; 0.1-0.95 coarse sweep plus 0.005 refinement",
        "models": overall_rows,
        "checkpoints": {
            "Phase 5": str(baseline_path),
            "Phase 9 layer4": str(fine_tuned_path),
        },
        "test_split_evaluated": False,
    }
    with (output_dir / "phase9_layer4_validation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(summary, output_file, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare Phase 5 and Phase 9 layer4 models on validation data."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument(
        "--fine-tuned",
        type=Path,
        default=project_dir
        / "results"
        / "resnet18_unet_phase9_layer4_finetuned_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    run_analysis(
        project_dir,
        args.baseline,
        args.fine_tuned,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
