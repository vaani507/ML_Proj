import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.dataset import get_dataloaders
from src.models import ResNet18UNet, UNet
from src.train import compute_detailed_metrics


def _infer(model, loader, device):
    predictions = []
    targets = []
    images = []
    elapsed = 0.0
    count = 0
    model.eval()
    with torch.inference_mode():
        for batch_images, batch_masks, _ in loader:
            batch_images = batch_images.to(device)
            started = time.perf_counter()
            batch_predictions = model(batch_images)
            elapsed += time.perf_counter() - started
            predictions.append(batch_predictions.cpu())
            targets.append(batch_masks)
            images.append(batch_images.cpu())
            count += len(batch_images)
    if count == 0:
        raise ValueError("Cannot analyze an empty data loader")
    return torch.cat(predictions), torch.cat(targets), torch.cat(images), elapsed / count


def _confusion(predictions, targets, threshold):
    predicted = predictions > threshold
    target = targets > 0.5
    true_positive = int((predicted & target).sum().item())
    false_positive = int((predicted & ~target).sum().item())
    false_negative = int((~predicted & target).sum().item())
    true_negative = int((~predicted & ~target).sum().item())
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_negative_pixels": true_negative,
        "ground_truth_defect_pixels": true_positive + false_negative,
        "predicted_defect_pixels": true_positive + false_positive,
    }


def _dark_linear_structure(gray):
    """Approximate dark elongated image structures; this is not a semantic grid label."""
    horizontal = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)),
    )
    vertical = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31)),
    )
    linear_features = np.maximum(horizontal, vertical) >= 24
    return cv2.dilate(
        linear_features.astype(np.uint8),
        np.ones((7, 7), dtype=np.uint8),
    ).astype(bool)


def _image_rgb(normalized):
    return np.clip(
        normalized.transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN,
        0,
        1,
    )


def _dataset_records(loader, split):
    fractions = []
    components = []
    empty_masks = 0
    total_pixels = 0
    defect_pixels = 0
    for _, masks, _ in loader:
        binary_masks = masks[:, 0].numpy().astype(np.uint8)
        areas = binary_masks.reshape(len(binary_masks), -1).sum(axis=1)
        fractions.extend((areas / (224 * 224)).tolist())
        empty_masks += int((areas == 0).sum())
        defect_pixels += int(areas.sum())
        total_pixels += int(binary_masks.size)
        for mask in binary_masks:
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask,
                connectivity=8,
            )
            components.extend(stats[1:count, cv2.CC_STAT_AREA].tolist())

    area_fractions = np.asarray(fractions)
    component_areas = np.asarray(components)
    row = {
        "split": split,
        "images": len(fractions),
        "defect_pixels": defect_pixels,
        "total_pixels": total_pixels,
        "defect_pixel_fraction": defect_pixels / total_pixels,
        "empty_masks": empty_masks,
        "median_defect_fraction_per_image": float(np.median(area_fractions)),
        "min_defect_fraction_per_image": float(area_fractions.min()),
        "max_defect_fraction_per_image": float(area_fractions.max()),
        "images_with_defect_fraction_le_0_5_percent": int(
            (area_fractions <= 0.005).sum()
        ),
        "images_with_defect_fraction_le_1_percent": int(
            (area_fractions <= 0.01).sum()
        ),
        "images_with_defect_fraction_le_2_percent": int(
            (area_fractions <= 0.02).sum()
        ),
        "images_with_defect_fraction_le_5_percent": int(
            (area_fractions <= 0.05).sum()
        ),
        "foreground_components": len(component_areas),
        "components_under_10_pixels": int((component_areas < 10).sum()),
        "components_under_20_pixels": int((component_areas < 20).sum()),
        "smallest_component_pixels": (
            int(component_areas.min()) if len(component_areas) else 0
        ),
    }
    return row


def _raw_component_retention(project_dir):
    report_path = project_dir / "results" / "preprocessing" / "preprocessing_report.csv"
    if not report_path.is_file():
        return {}

    retained = {}
    with report_path.open(newline="", encoding="utf-8-sig") as source:
        rows = csv.DictReader(source)
        for row in rows:
            if row["status"] != "ok":
                continue
            split = row["split"]
            raw_path = project_dir / "data" / "SolarCells" / Path(row["mask"])
            processed_path = (
                project_dir
                / "data"
                / "processed"
                / "SolarCells"
                / split
                / "masks"
                / f"{Path(row['image']).stem}.png"
            )
            raw = (np.asarray(Image.open(raw_path).convert("L")) > 127).astype(
                np.uint8
            )
            resized = (
                np.asarray(
                    Image.fromarray(raw * 255).resize(
                        (224, 224),
                        Image.Resampling.NEAREST,
                    )
                )
                > 127
            )
            processed = np.asarray(Image.open(processed_path).convert("L")) > 0
            if not np.array_equal(resized, processed):
                raise ValueError(
                    f"Preprocessed mask differs from expected nearest resize: "
                    f"{processed_path}"
                )

            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                raw,
                connectivity=8,
            )
            sample_y = sample_x = None
            for y_offset in (0, 1):
                for x_offset in (0, 1):
                    source_sample = raw[y_offset::2, x_offset::2]
                    if source_sample.shape == (224, 224) and np.array_equal(
                        source_sample.astype(bool),
                        resized,
                    ):
                        sample_y, sample_x = y_offset, x_offset
                        break
                if sample_y is not None:
                    break
            if sample_y is None:
                raise ValueError(f"Unable to map nearest resize samples for {raw_path}")

            sampled_labels = labels[sample_y::2, sample_x::2]
            surviving_components = set(
                np.unique(sampled_labels[resized]).tolist()
            ) - {0}
            lost_small = 0
            total_small = 0
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] <= 64:
                    total_small += 1
                    lost_small += int(component not in surviving_components)
            counts = retained.setdefault(split, [0, 0])
            counts[0] += lost_small
            counts[1] += total_small

    return {
        split: {
            "small_components_lost": counts[0],
            "small_components_total": counts[1],
        }
        for split, counts in retained.items()
    }


def _select_visual_examples(images, targets, per_image_results):
    target_masks = targets[:, 0].numpy().astype(bool)
    areas = target_masks.reshape(len(target_masks), -1).sum(axis=1)
    model_names = tuple(
        name for name in per_image_results if name != "image_names"
    )
    latest_model = model_names[-1]
    best = int(np.argmax(per_image_results[latest_model]["dice"]))
    worst = int(np.argmin(per_image_results[latest_model]["dice"]))
    smallest = int(np.argmin(areas))
    grid_index = int(
        np.argmax(
            per_image_results[latest_model]["false_positive_near_dark_line"]
        )
    )

    selected = []
    labels = []
    for index, label in (
        (best, "Best ResNet-18 Dice"),
        (worst, "Lowest ResNet-18 Dice"),
        (smallest, "Smallest ground-truth mask"),
        (grid_index, "Most FP near dark linear structures"),
    ):
        if index not in selected:
            selected.append(index)
            labels.append(label)
    return selected, labels


def _save_error_figure(path, images, targets, predictions, per_image_results):
    indices, labels = _select_visual_examples(images, targets, per_image_results)
    model_names = tuple(predictions)
    figure, axes = plt.subplots(
        len(indices),
        2 + len(model_names),
        figsize=(4.0 * (2 + len(model_names)), 3.5 * len(indices)),
        squeeze=False,
    )
    headings = ("EL image", "Ground truth") + tuple(
        f"{name}\nTP green | FP red | FN blue" for name in model_names
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)

    for row, (index, label) in enumerate(zip(indices, labels)):
        rgb = _image_rgb(images[index].numpy())
        target = targets[index, 0].numpy().astype(bool)
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        for column, name in enumerate(model_names, start=2):
            predicted = predictions[name][index, 0].numpy().astype(bool)
            true_positive = target & predicted
            false_positive = ~target & predicted
            false_negative = target & ~predicted
            overlay = np.zeros((*target.shape, 4), dtype=np.float32)
            overlay[true_positive] = (0.1, 1.0, 0.1, 0.8)
            overlay[false_positive] = (1.0, 0.1, 0.1, 0.85)
            overlay[false_negative] = (0.1, 0.75, 1.0, 0.85)
            axes[row, column].imshow(rgb)
            axes[row, column].imshow(overlay)
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=9)
        axes[row, 0].set_xlabel(
            Path(per_image_results["image_names"][index]).stem,
            fontsize=8,
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return indices, labels


def analyze_phase5(
    project_dir,
    custom_checkpoint,
    initial_resnet_checkpoint,
    long_resnet_checkpoint,
    phase5_resnet_checkpoint,
    output_dir,
    batch_size=8,
    num_threads=6,
):
    if batch_size <= 0 or num_threads <= 0:
        raise ValueError("batch_size and num_threads must be greater than zero")
    torch.set_num_threads(num_threads)
    device = torch.device("cpu")
    train_loader, validation_loader, test_loader = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    dataset_rows = [
        _dataset_records(train_loader, "training subset"),
        _dataset_records(validation_loader, "validation"),
        _dataset_records(test_loader, "test"),
    ]
    raw_retention = _raw_component_retention(project_dir)

    model_specs = (
        ("Custom U-Net", UNet, custom_checkpoint),
        (
            "ResNet-18 U-Net (5 epochs)",
            ResNet18UNet,
            initial_resnet_checkpoint,
        ),
        (
            "ResNet-18 U-Net (20-epoch run)",
            ResNet18UNet,
            long_resnet_checkpoint,
        ),
        (
            "ResNet-18 U-Net (Phase 5)",
            ResNet18UNet,
            phase5_resnet_checkpoint,
        ),
    )
    threshold_grids = {
        "Custom U-Net": tuple(np.arange(0.4, 0.6001, 0.025)),
        "ResNet-18 U-Net (5 epochs)": tuple(np.arange(0.5, 0.7501, 0.025)),
        "ResNet-18 U-Net (20-epoch run)": tuple(np.arange(0.6, 0.9501, 0.025)),
        "ResNet-18 U-Net (Phase 5)": tuple(np.arange(0.1, 0.9501, 0.025)),
    }
    validation_results = {}
    test_results = {}
    test_predictions = {}
    metadata = {}
    test_targets = None
    test_images = None
    image_names = [
        image_path.name for image_path, _ in test_loader.dataset.samples
    ]

    for name, model_type, checkpoint_path in model_specs:
        model = model_type() if model_type is UNet else model_type(pretrained=False)
        state, checkpoint_metadata = _load_state(checkpoint_path)
        model.load_state_dict(state)
        model.to(device).eval()
        metadata[name] = checkpoint_metadata

        validation_pred, validation_mask, _, validation_inference = _infer(
            model,
            validation_loader,
            device,
        )
        thresholds = threshold_grids[name]
        threshold_scores = {
            float(threshold): compute_detailed_metrics(
                validation_pred,
                validation_mask,
                threshold=float(threshold),
            )
            for threshold in thresholds
        }
        selected_threshold = max(
            threshold_scores,
            key=lambda threshold: threshold_scores[threshold]["dice"],
        )
        validation_results[name] = {
            "threshold": selected_threshold,
            "metrics": threshold_scores[selected_threshold],
            "scores": threshold_scores,
            "inference_seconds_per_image": validation_inference,
        }

        test_pred, current_test_mask, current_test_images, test_inference = _infer(
            model,
            test_loader,
            device,
        )
        if test_targets is None:
            test_targets = current_test_mask
            test_images = current_test_images
        elif not torch.equal(test_targets, current_test_mask):
            raise RuntimeError("Test masks differed between model evaluations")

        final_binary = test_pred > selected_threshold
        test_results[name] = {
            "probabilities": test_pred,
            "binary": final_binary,
            "metrics": compute_detailed_metrics(
                test_pred,
                current_test_mask,
                threshold=selected_threshold,
            ),
            "counts": _confusion(
                test_pred,
                current_test_mask,
                selected_threshold,
            ),
            "inference_seconds_per_image": test_inference,
        }
        print(
            f"{name}: validation threshold={selected_threshold:.3f}, "
            f"validation Dice={validation_results[name]['metrics']['dice']:.5f}; "
            f"test Dice={test_results[name]['metrics']['dice']:.5f}"
        )
        test_predictions[name] = final_binary

    dark_line_zones = []
    for normalized_image in test_images.numpy():
        rgb = _image_rgb(normalized_image)
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        dark_line_zones.append(_dark_linear_structure(gray))

    per_image = {
        name: {"dice": [], "fp_near": [], "fp_total": []}
        for name, _, _ in model_specs
    }
    per_image_rows = []
    for index, image_name in enumerate(image_names):
        target = test_targets[index, 0].numpy().astype(bool)
        dark_zone = dark_line_zones[index]
        for name, _, _ in model_specs:
            predicted = test_predictions[name][index, 0].numpy()
            true_positive = int((predicted & target).sum())
            false_positive = int((predicted & ~target).sum())
            false_negative = int((~predicted & target).sum())
            denominator = 2 * true_positive + false_positive + false_negative
            dice = 2 * true_positive / denominator if denominator else 1.0
            fp_near = int((predicted & ~target & dark_zone).sum())
            fp_outside = false_positive - fp_near
            background_near = int((~target & dark_zone).sum())
            background_outside = int((~target & ~dark_zone).sum())
            per_image[name]["dice"].append(dice)
            per_image[name]["fp_near"].append(fp_near)
            per_image[name]["fp_total"].append(false_positive)
            per_image_rows.append(
                {
                    "model": name,
                    "image": image_name,
                    "threshold": validation_results[name]["threshold"],
                    "true_positive_pixels": true_positive,
                    "false_positive_pixels": false_positive,
                    "false_negative_pixels": false_negative,
                    "ground_truth_defect_pixels": int(target.sum()),
                    "predicted_defect_pixels": int(predicted.sum()),
                    "dice": dice,
                    "false_positive_near_dark_linear_proxy": fp_near,
                    "false_positive_outside_dark_linear_proxy": fp_outside,
                    "background_pixels_near_dark_linear_proxy": background_near,
                    "background_pixels_outside_dark_linear_proxy": background_outside,
                }
            )

    summary_rows = []
    for name, _, _ in model_specs:
        validation = validation_results[name]
        test = test_results[name]
        counts = test["counts"]
        fp_near = sum(per_image[name]["fp_near"])
        fp_total = sum(per_image[name]["fp_total"])
        background_near = sum(
            row["background_pixels_near_dark_linear_proxy"]
            for row in per_image_rows
            if row["model"] == name
        )
        background_outside = sum(
            row["background_pixels_outside_dark_linear_proxy"]
            for row in per_image_rows
            if row["model"] == name
        )
        fp_outside = fp_total - fp_near
        summary_rows.append(
            {
                "model": name,
                "validation_threshold": validation["threshold"],
                "validation_dice": validation["metrics"]["dice"],
                "validation_iou": validation["metrics"]["iou"],
                "validation_precision": validation["metrics"]["precision"],
                "validation_recall": validation["metrics"]["recall"],
                "test_dice": test["metrics"]["dice"],
                "test_iou": test["metrics"]["iou"],
                "test_precision": test["metrics"]["precision"],
                "test_recall": test["metrics"]["recall"],
                **counts,
                "test_micro_dice": 2
                * counts["true_positive_pixels"]
                / max(
                    1,
                    2 * counts["true_positive_pixels"]
                    + counts["false_positive_pixels"]
                    + counts["false_negative_pixels"],
                ),
                "training_seconds": metadata[name].get(
                    "training_seconds",
                    "Not recorded",
                ),
                "test_inference_seconds_per_image": test[
                    "inference_seconds_per_image"
                ],
                "false_positive_near_dark_linear_proxy": fp_near,
                "false_positive_fraction_near_dark_linear_proxy": fp_near
                / max(1, fp_total),
                "false_positive_rate_near_dark_linear_proxy": fp_near
                / max(1, background_near),
                "false_positive_rate_outside_dark_linear_proxy": fp_outside
                / max(1, background_outside),
            }
        )
        print(
            f"  test pixel counts TP={counts['true_positive_pixels']} "
            f"FP={counts['false_positive_pixels']} "
            f"FN={counts['false_negative_pixels']}; "
            f"FP near dark-line proxy={fp_near}/{fp_total} "
            f"({fp_near / max(1, fp_total):.1%})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "phase5_model_comparison.csv",
        summary_rows,
    )
    _write_csv(
        output_dir / "phase5_per_image_errors.csv",
        per_image_rows,
    )
    _write_csv(
        output_dir / "phase5_validation_thresholds.csv",
        [
            {
                "model": name,
                "threshold": threshold,
                **scores,
                "selected": threshold == validation_results[name]["threshold"],
            }
            for name in validation_results
            for threshold, scores in validation_results[name]["scores"].items()
        ],
    )
    _write_csv(
        output_dir / "phase5_dataset_statistics.csv",
        [
            {
                **row,
                        "raw_training_split_components_under_64_lost": raw_retention.get(
                            "train", {}
                        ).get("small_components_lost", "not measured"),
                        "raw_training_split_components_under_64_total": raw_retention.get(
                            "train", {}
                        ).get("small_components_total", "not measured"),
                        "raw_test_split_components_under_64_lost": raw_retention.get(
                            "test", {}
                        ).get("small_components_lost", "not measured"),
                        "raw_test_split_components_under_64_total": raw_retention.get(
                            "test", {}
                        ).get("small_components_total", "not measured"),
            }
            for row in dataset_rows
        ],
    )

    figure_results = {
        name: {
            "dice": per_image[name]["dice"],
            "false_positive_near_dark_line": per_image[name]["fp_near"],
        }
        for name, _, _ in model_specs
    }
    figure_results["image_names"] = image_names
    indices, labels = _save_error_figure(
        output_dir / "phase5_qualitative_errors.png",
        test_images,
        test_targets,
        test_predictions,
        figure_results,
    )
    _write_csv(
        output_dir / "phase5_visual_examples.csv",
        [
            {"image": image_names[index], "selection_reason": label}
            for index, label in zip(indices, labels)
        ],
    )
    print(f"Saved Phase 5 analyses under {output_dir}")


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Phase 5 validation calibration and held-out segmentation "
            "errors without training."
        )
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument(
        "--custom-checkpoint",
        type=Path,
        default=project_dir / "results" / "unet_solar_defect.pth",
    )
    parser.add_argument(
        "--initial-resnet-checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_best.pth",
    )
    parser.add_argument(
        "--long-resnet-checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_20ep_best.pth",
    )
    parser.add_argument(
        "--phase5-resnet-checkpoint",
        type=Path,
        default=project_dir
        / "results"
        / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results",
    )
    args = parser.parse_args()
    analyze_phase5(
        project_dir=project_dir,
        custom_checkpoint=args.custom_checkpoint,
        initial_resnet_checkpoint=args.initial_resnet_checkpoint,
        long_resnet_checkpoint=args.long_resnet_checkpoint,
        phase5_resnet_checkpoint=args.phase5_resnet_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
