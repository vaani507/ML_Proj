import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def _boundary_f1(predictions, targets, threshold, tolerance=1):
    predicted = (predictions > threshold).float()
    target = (targets > 0.5).float()
    kernel_size = 3

    def boundary(mask):
        dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=1)
        eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=1)
        return (dilated - eroded) > 0.5

    predicted_boundary = boundary(predicted)
    target_boundary = boundary(target)
    if tolerance:
        target_neighborhood = F.max_pool2d(
            target_boundary.float(),
            kernel_size=2 * tolerance + 1,
            stride=1,
            padding=tolerance,
        ) > 0
        predicted_neighborhood = F.max_pool2d(
            predicted_boundary.float(),
            kernel_size=2 * tolerance + 1,
            stride=1,
            padding=tolerance,
        ) > 0
    else:
        target_neighborhood = target_boundary
        predicted_neighborhood = predicted_boundary

    predicted_count = predicted_boundary.sum(dim=(1, 2, 3)).float()
    target_count = target_boundary.sum(dim=(1, 2, 3)).float()
    matched_prediction = (
        predicted_boundary & target_neighborhood
    ).sum(dim=(1, 2, 3)).float()
    matched_target = (
        target_boundary & predicted_neighborhood
    ).sum(dim=(1, 2, 3)).float()
    precision = matched_prediction / predicted_count.clamp_min(1)
    recall = matched_target / target_count.clamp_min(1)
    score = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    both_empty = (predicted_count == 0) & (target_count == 0)
    score[both_empty] = 1.0
    return score.mean().item()


def _per_image_rows(name, predictions, targets, threshold, image_names):
    rows = []
    for index, image_name in enumerate(image_names):
        target = targets[index : index + 1] > 0.5
        predicted = predictions[index : index + 1] > threshold
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        gt_pixels = int(target.sum())
        rows.append(
            {
                "model": name,
                "image": image_name,
                "size_group": _size_group(gt_pixels / target[0, 0].numel()),
                "ground_truth_pixels": gt_pixels,
                "threshold": threshold,
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "predicted_defect_pixels": tp + fp,
                "predicted_to_ground_truth_area_ratio": (
                    (tp + fp) / gt_pixels if gt_pixels else ""
                ),
                "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "iou": tp / max(1, tp + fp + fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "completely_missed": int(gt_pixels > 0 and tp == 0),
            }
        )
    return rows


def _group_rows(per_image_rows):
    rows = []
    for model in dict.fromkeys(row["model"] for row in per_image_rows):
        model_rows = [row for row in per_image_rows if row["model"] == model]
        for group in SIZE_GROUPS:
            selected = [row for row in model_rows if row["size_group"] == group]
            row = {
                "model": model,
                "size_group": group,
                "images": len(selected),
            }
            for metric in ("dice", "iou", "precision", "recall"):
                row[f"mean_{metric}"] = (
                    float(np.mean([item[metric] for item in selected]))
                    if selected
                    else ""
                )
            row["completely_missed_defects"] = sum(
                item["completely_missed"] for item in selected
            )
            row["false_positive_pixels"] = sum(
                item["false_positive_pixels"] for item in selected
            )
            row["false_negative_pixels"] = sum(
                item["false_negative_pixels"] for item in selected
            )
            row["mean_predicted_to_ground_truth_area_ratio"] = (
                float(
                    np.mean(
                        [
                            item["predicted_to_ground_truth_area_ratio"]
                            for item in selected
                        ]
                    )
                )
                if selected
                else ""
            )
            rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _visual_examples(per_image):
    baseline = [row for row in per_image if row["model"] == "Phase 5"]
    positive = [row for row in baseline if row["ground_truth_pixels"] > 0]
    tiny = [row for row in positive if row["size_group"] == "tiny"]
    small_or_medium = [
        row for row in positive if row["size_group"] in ("tiny", "medium")
    ]
    examples = []
    if positive:
        examples.append(
            (
                min(positive, key=lambda row: row["ground_truth_pixels"]),
                "Smallest defect",
            )
        )
    if tiny:
        examples.append((min(tiny, key=lambda row: row["dice"]), "Lowest tiny Dice"))
    if small_or_medium:
        examples.append(
            (
                min(small_or_medium, key=lambda row: row["recall"]),
                "Lowest small/thin recall",
            )
        )
    selected = []
    labels = []
    for row, label in examples:
        if row["image"] not in selected:
            selected.append(row["image"])
            labels.append(label)
    return selected, labels


def _save_visualization(path, project_dir, image_names, targets, predictions, thresholds, per_image):
    chosen, labels = _visual_examples(per_image)
    report_path = project_dir / "results" / "preprocessing" / "preprocessing_report.csv"
    with report_path.open(newline="", encoding="utf-8-sig") as source:
        raw_paths = {
            Path(row["image"]).stem: row["image"]
            for row in csv.DictReader(source)
            if row["split"] == "train" and row["status"] == "ok"
        }

    figure, axes = plt.subplots(
        len(chosen),
        6,
        figsize=(19, 3.8 * len(chosen)),
        squeeze=False,
    )
    headings = (
        "Original EL image",
        "Ground truth",
        "Phase 5 prediction",
        "Phase 7 prediction",
        "Phase 5 errors",
        "Phase 7 errors",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)

    for row_index, (image_name, label) in enumerate(zip(chosen, labels)):
        index = image_names.index(image_name)
        stem = Path(image_name).stem
        raw_relative = raw_paths.get(stem)
        if raw_relative is None:
            raise FileNotFoundError(f"No original image recorded for {image_name}")
        image_path = project_dir / "data" / "SolarCells" / Path(raw_relative)
        with Image.open(image_path) as source_image:
            original = np.asarray(
                ImageOps.exif_transpose(source_image).convert("RGB")
            )
        target = targets[index, 0].numpy() > 0.5
        old_mask = (
            predictions["Phase 5"][index, 0].numpy()
            > thresholds["Phase 5"]
        )
        new_mask = (
            predictions["Phase 7 boundary"][index, 0].numpy()
            > thresholds["Phase 7 boundary"]
        )
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
        axes[row_index, 2].imshow(old_display, cmap="gray", vmin=0, vmax=1)
        axes[row_index, 3].imshow(new_display, cmap="gray", vmin=0, vmax=1)
        for column, predicted in ((4, old_display), (5, new_display)):
            tp = target_display & predicted
            fp = ~target_display & predicted
            fn = target_display & ~predicted
            overlay = np.zeros((*target_display.shape, 4), dtype=np.float32)
            overlay[tp] = (0.1, 1.0, 0.1, 0.8)
            overlay[fp] = (1.0, 0.1, 0.1, 0.85)
            overlay[fn] = (0.1, 0.75, 1.0, 0.85)
            axes[row_index, column].imshow(original)
            axes[row_index, column].imshow(overlay)
        axes[row_index, 0].set_ylabel(f"{image_name}: {label}", fontsize=9)
        for axis in axes[row_index]:
            axis.axis("off")

    figure.suptitle("TP green | FP red | FN blue", y=1.01)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze_validation(project_dir, baseline_checkpoint, phase7_checkpoint, output_dir):
    torch.set_num_threads(6)
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=8,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    validation_subset = validation_loader.dataset
    image_names = [
        validation_subset.dataset.samples[index][0].name
        for index in validation_subset.indices
    ]
    model_specs = (
        ("Phase 5", baseline_checkpoint),
        ("Phase 7 boundary", phase7_checkpoint),
    )
    all_predictions = {}
    all_targets = None
    summary_rows = []
    per_image = []
    threshold_rows = []

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
        model_rows = _per_image_rows(
            name,
            predictions,
            targets,
            threshold,
            image_names,
        )
        per_image.extend(model_rows)
        metrics = compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        groups = {
            row["size_group"]: row
            for row in _group_rows(model_rows)
        }
        summary_rows.append(
            {
                "model": name,
                "checkpoint": str(checkpoint_path),
                "validation_threshold": threshold,
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
                "tiny_iou": groups["tiny"]["mean_iou"],
                "tiny_precision": groups["tiny"]["mean_precision"],
                "tiny_recall": groups["tiny"]["mean_recall"],
                "tiny_completely_missed": groups["tiny"][
                    "completely_missed_defects"
                ],
                "medium_dice": groups["medium"]["mean_dice"],
                "large_dice": groups["large"]["mean_dice"],
                "false_positive_pixels": sum(
                    row["false_positive_pixels"] for row in model_rows
                ),
                "false_negative_pixels": sum(
                    row["false_negative_pixels"] for row in model_rows
                ),
                "predicted_to_ground_truth_area_ratio": (
                    sum(row["predicted_defect_pixels"] for row in model_rows)
                    / max(1, sum(row["ground_truth_pixels"] for row in model_rows))
                ),
                "training_seconds": metadata.get("training_seconds", "Not recorded"),
                "boundary_weight": metadata.get("boundary_weight", 0.0),
            }
        )
        for candidate, candidate_metrics in sorted(scores.items()):
            threshold_rows.append(
                {
                    "model": name,
                    "threshold": candidate,
                    **candidate_metrics,
                    "selected": candidate == threshold,
                }
            )
        del model

    size_rows = []
    for name, _ in model_specs:
        size_rows.extend(
            _group_rows([row for row in per_image if row["model"] == name])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "phase7_validation_comparison.csv", summary_rows)
    _write_csv(output_dir / "phase7_validation_size_groups.csv", size_rows)
    _write_csv(output_dir / "phase7_validation_per_image.csv", per_image)
    _write_csv(output_dir / "phase7_validation_thresholds.csv", threshold_rows)
    _save_visualization(
        output_dir / "phase7_validation_qualitative.png",
        project_dir,
        image_names,
        all_targets,
        all_predictions,
        {row["model"]: row["validation_threshold"] for row in summary_rows},
        per_image,
    )
    print(f"Saved Phase 7 validation analysis under {output_dir}")
    for row in summary_rows:
        print(
            f"{row['model']}: threshold={row['validation_threshold']:.3f}, "
            f"Dice={row['validation_dice']:.4f}, tiny Dice={row['tiny_dice']}, "
            f"medium Dice={row['medium_dice']}, large Dice={row['large_dice']}, "
            f"boundary F1={row['validation_boundary_f1_tolerance_1px']:.4f}, "
            f"FP={row['false_positive_pixels']}, "
            f"FN={row['false_negative_pixels']}"
        )


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare Phase 5 and Phase 7 on validation data only."
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument(
        "--phase7-checkpoint",
        type=Path,
        default=project_dir
        / "results"
        / "resnet18_unet_phase7_boundary_best.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results")
    args = parser.parse_args()
    analyze_validation(
        project_dir,
        args.baseline_checkpoint,
        args.phase7_checkpoint,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
