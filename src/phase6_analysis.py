import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.dataset import get_dataloaders
from src.models import ResNet18UNet
from src.train import compute_detailed_metrics


SIZE_GROUPS = ("tiny", "medium", "large", "empty")


def _size_group(fraction):
    if fraction == 0:
        return "empty"
    if fraction <= 0.02:
        return "tiny"
    if fraction <= 0.05:
        return "medium"
    return "large"


def _infer(model, loader):
    predictions = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for images, masks, _ in loader:
            predictions.append(model(images).cpu())
            targets.append(masks)
    if not predictions:
        raise ValueError("The validation data loader is empty")
    return torch.cat(predictions), torch.cat(targets)


def _select_threshold(predictions, targets):
    coarse = [round(0.1 + step * 0.025, 3) for step in range(35)]
    coarse_scores = {
        threshold: compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        for threshold in coarse
    }
    center = max(coarse, key=lambda threshold: coarse_scores[threshold]["dice"])
    fine = {
        round(center - 0.025 + step * 0.005, 3)
        for step in range(11)
    }
    scores = {
        **coarse_scores,
        **{
            threshold: compute_detailed_metrics(
                predictions,
                targets,
                threshold=threshold,
            )
            for threshold in fine
        },
    }
    selected = max(scores, key=lambda threshold: scores[threshold]["dice"])
    return selected, scores


def _per_image_rows(name, predictions, targets, threshold, image_names):
    rows = []
    for index, image_name in enumerate(image_names):
        target = targets[index, 0].numpy() > 0.5
        probability = predictions[index, 0].numpy()
        predicted = probability > threshold
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        area = int(target.sum())
        fraction = area / target.size
        rows.append(
            {
                "model": name,
                "image": image_name,
                "size_group": _size_group(fraction),
                "ground_truth_pixels": area,
                "ground_truth_fraction": fraction,
                "threshold": threshold,
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "iou": tp / max(1, tp + fp + fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "completely_missed": int(area > 0 and tp == 0),
            }
        )
    return rows


def _group_rows(per_image_rows):
    rows = []
    for model in dict.fromkeys(row["model"] for row in per_image_rows):
        model_rows = [row for row in per_image_rows if row["model"] == model]
        for group in SIZE_GROUPS:
            selected = [row for row in model_rows if row["size_group"] == group]
            rows.append(
                {
                    "model": model,
                    "size_group": group,
                    "images": len(selected),
                    "mean_dice": (
                        float(np.mean([row["dice"] for row in selected]))
                        if selected
                        else ""
                    ),
                    "mean_iou": (
                        float(np.mean([row["iou"] for row in selected]))
                        if selected
                        else ""
                    ),
                    "mean_precision": (
                        float(np.mean([row["precision"] for row in selected]))
                        if selected
                        else ""
                    ),
                    "mean_recall": (
                        float(np.mean([row["recall"] for row in selected]))
                        if selected
                        else ""
                    ),
                    "completely_missed_defects": sum(
                        row["completely_missed"] for row in selected
                    ),
                    "false_positive_pixels": sum(
                        row["false_positive_pixels"] for row in selected
                    ),
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


def _visual_indices(per_image_rows):
    baseline = [row for row in per_image_rows if row["model"] == "Existing best"]
    positive = [row for row in baseline if row["ground_truth_pixels"] > 0]
    tiny = [row for row in positive if row["size_group"] == "tiny"]
    hard_thin = [
        row for row in positive if row["ground_truth_fraction"] <= 0.05
    ]
    candidates = []
    if positive:
        candidates.append(
            (min(positive, key=lambda row: row["ground_truth_pixels"]), "Smallest defect")
        )
    if tiny:
        candidates.append((min(tiny, key=lambda row: row["dice"]), "Lowest tiny-defect Dice"))
    if hard_thin:
        candidates.append(
            (min(hard_thin, key=lambda row: row["recall"]), "Lowest thin/small recall")
        )
    selected = []
    labels = []
    for row, label in candidates:
        if row["image"] not in selected:
            selected.append(row["image"])
            labels.append(label)
    return selected, labels


def _save_visualization(path, project_dir, image_names, targets, predictions, thresholds, per_image):
    chosen, labels = _visual_indices(per_image)
    with (project_dir / "results" / "preprocessing" / "preprocessing_report.csv").open(
        newline="",
        encoding="utf-8-sig",
    ) as source:
        raw_paths = {
            Path(row["image"]).stem: row["image"]
            for row in csv.DictReader(source)
            if row["split"] == "train" and row["status"] == "ok"
        }

    fig, axes = plt.subplots(
        len(chosen),
        7,
        figsize=(21, 3.8 * len(chosen)),
        squeeze=False,
    )
    headings = (
        "Original EL image",
        "Ground truth",
        "Existing probability",
        "Existing prediction",
        "New probability",
        "New prediction",
        "New-model errors: TP green / FP red / FN blue",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)

    for row_index, (image_name, label) in enumerate(zip(chosen, labels)):
        index = image_names.index(image_name)
        stem = Path(image_name).stem
        raw_relative = raw_paths.get(stem)
        if raw_relative is None:
            raise FileNotFoundError(f"No original image recorded for {image_name}")
        raw_image_path = project_dir / "data" / "SolarCells" / Path(raw_relative)
        with Image.open(raw_image_path) as source_image:
            original = np.asarray(
                ImageOps.exif_transpose(source_image).convert("RGB")
            )
        target = targets[index, 0].numpy() > 0.5
        old_probability = predictions["Existing best"][index, 0].numpy()
        new_probability = predictions["Phase 6 sampling"][index, 0].numpy()
        old_mask = old_probability > thresholds["Existing best"]
        new_mask = new_probability > thresholds["Phase 6 sampling"]
        height, width = original.shape[:2]

        def display_mask(mask):
            return np.asarray(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                )
            ) > 0

        target_display = display_mask(target)
        old_display = display_mask(old_mask)
        new_display = display_mask(new_mask)
        axes[row_index, 0].imshow(original)
        axes[row_index, 1].imshow(target_display, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].imshow(old_probability, cmap="inferno", vmin=0, vmax=1)
        axes[row_index, 3].imshow(old_display, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 4].imshow(new_probability, cmap="inferno", vmin=0, vmax=1)
        axes[row_index, 5].imshow(new_display, cmap="gray", vmin=0, vmax=1)

        true_positive = target_display & new_display
        false_positive = ~target_display & new_display
        false_negative = target_display & ~new_display
        overlay = np.zeros((*target_display.shape, 4), dtype=np.float32)
        overlay[true_positive] = (0.1, 1.0, 0.1, 0.8)
        overlay[false_positive] = (1.0, 0.1, 0.1, 0.85)
        overlay[false_negative] = (0.1, 0.75, 1.0, 0.85)
        axes[row_index, 6].imshow(original)
        axes[row_index, 6].imshow(overlay)
        axes[row_index, 0].set_ylabel(f"{image_name}: {label}", fontsize=9)
        for axis in axes[row_index]:
            axis.axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_validation(project_dir, baseline_checkpoint, phase6_checkpoint, output_dir):
    torch.set_num_threads(6)
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=8,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    subset = validation_loader.dataset
    image_names = [
        subset.dataset.samples[index][0].name for index in subset.indices
    ]
    model_specs = (
        ("Existing best", baseline_checkpoint),
        ("Phase 6 sampling", phase6_checkpoint),
    )
    all_predictions = {}
    all_targets = None
    threshold_rows = []
    per_image = []
    summary = []

    for name, checkpoint_path in model_specs:
        model = ResNet18UNet(pretrained=False).cpu()
        state, metadata = _load_state(checkpoint_path)
        model.load_state_dict(state)
        predictions, targets = _infer(model, validation_loader)
        if all_targets is None:
            all_targets = targets
        elif not torch.equal(all_targets, targets):
            raise RuntimeError("Validation targets differ between model runs")
        threshold, scores = _select_threshold(predictions, targets)
        all_predictions[name] = predictions
        rows = _per_image_rows(
            name,
            predictions,
            targets,
            threshold,
            image_names,
        )
        per_image.extend(rows)
        groups = _group_rows(rows)
        overall_metrics = compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        small = next(row for row in groups if row["size_group"] == "tiny")
        summary.append(
            {
                "model": name,
                "checkpoint": str(checkpoint_path),
                "validation_threshold": threshold,
                "validation_dice": overall_metrics["dice"],
                "validation_iou": overall_metrics["iou"],
                "validation_precision": overall_metrics["precision"],
                "validation_recall": overall_metrics["recall"],
                "tiny_images": small["images"],
                "tiny_dice": small["mean_dice"],
                "tiny_iou": small["mean_iou"],
                "tiny_precision": small["mean_precision"],
                "tiny_recall": small["mean_recall"],
                "tiny_completely_missed": small["completely_missed_defects"],
                "false_positive_pixels": sum(
                    row["false_positive_pixels"] for row in rows
                ),
                "training_seconds": metadata.get("training_seconds", "Not recorded"),
                "training_sampling": metadata.get("sampling", {"strategy": "unknown"}),
            }
        )
        for candidate, metrics in sorted(scores.items()):
            threshold_rows.append(
                {
                    "model": name,
                    "threshold": candidate,
                    **metrics,
                    "selected": candidate == threshold,
                }
            )
        del model

    group_rows = []
    for name, _ in model_specs:
        group_rows.extend(
            row for row in _group_rows(
                [item for item in per_image if item["model"] == name]
            )
        )
    _write_csv(output_dir / "phase6_validation_comparison.csv", summary)
    _write_csv(output_dir / "phase6_validation_size_groups.csv", group_rows)
    _write_csv(output_dir / "phase6_validation_per_image.csv", per_image)
    _write_csv(output_dir / "phase6_validation_thresholds.csv", threshold_rows)
    _save_visualization(
        output_dir / "phase6_validation_qualitative.png",
        project_dir,
        image_names,
        all_targets,
        all_predictions,
        {row["model"]: row["validation_threshold"] for row in summary},
        per_image,
    )
    print(f"Saved Phase 6 validation comparison under {output_dir}")
    for row in summary:
        print(
            f"{row['model']}: threshold={row['validation_threshold']:.3f}, "
            f"Dice={row['validation_dice']:.4f}, tiny Dice={row['tiny_dice']}, "
            f"tiny recall={row['tiny_recall']}, "
            f"tiny missed={row['tiny_completely_missed']}, "
            f"FP={row['false_positive_pixels']}, "
            f"training_seconds={row['training_seconds']}"
        )


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare Phase 6 checkpoints on validation data only."
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument(
        "--phase6-checkpoint",
        type=Path,
        default=project_dir
        / "results"
        / "resnet18_unet_phase6_small_sampling_best.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results")
    args = parser.parse_args()
    analyze_validation(
        project_dir,
        args.baseline_checkpoint,
        args.phase6_checkpoint,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
