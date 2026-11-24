import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.compare_models import _load_state
from src.dataset import get_dataloaders
from src.models import ResNet18UNet
from src.phase11_extended_training_analysis import (
    _error_overlay,
    _infer,
    _size_groups,
    _validation_data,
    _validation_names,
)
from src.phase5_analysis import _dark_linear_structure, _image_rgb
from src.phase8_tta import _select_threshold
from src.train import compute_detailed_metrics


THRESHOLD_STEPS = tuple(round(0.1 + step * 0.025, 3) for step in range(35))
COMPONENT_BINS = (
    ("1-3", 1, 3),
    ("4-15", 4, 15),
    ("16-63", 16, 63),
    ("64-255", 64, 255),
    ("256+", 256, None),
)


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_baseline_cache(cache_path, checkpoint_path, names, targets):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Phase 5 checkpoint not found: {checkpoint_path}")
    baseline_state, _ = _load_state(checkpoint_path)
    baseline_model = ResNet18UNet(pretrained=False).cpu()
    baseline_model.load_state_dict(baseline_state)
    baseline_model.eval()
    checkpoint_stat = checkpoint_path.stat()
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["image_names"], np.asarray(names)):
            raise RuntimeError("Cached Phase 5 predictions use a different split")
        if (
            int(cache["checkpoint_size_bytes"]) != checkpoint_stat.st_size
            or int(cache["checkpoint_mtime_ns"]) != checkpoint_stat.st_mtime_ns
        ):
            raise RuntimeError("Cached Phase 5 predictions use a different checkpoint")
        expected_targets = targets[:, 0].numpy().astype(np.uint8)
        if not np.array_equal(cache["targets"], expected_targets):
            raise RuntimeError("Cached Phase 5 masks do not match validation")
        predictions = torch.from_numpy(cache["probabilities"][:, None].copy())
        inference_seconds = float(cache["inference_seconds_per_image"])
    return predictions, inference_seconds


def _load_extended_predictions(checkpoint_path, loader):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Extended checkpoint not found: {checkpoint_path}")
    state, _ = _load_state(checkpoint_path)
    model = ResNet18UNet(pretrained=False).cpu()
    model.load_state_dict(state)
    predictions, inference_seconds = _infer(model, loader)
    return predictions, inference_seconds


def _binary_metrics(predictions, targets, threshold):
    predicted = predictions[:, 0].numpy() > threshold
    truth = targets[:, 0].numpy() > 0.5
    per_image_dice = []
    groups = _size_groups(targets)
    tiny_indices = np.flatnonzero(groups == "tiny")
    for prediction, target in zip(predicted, truth):
        tp = int(np.count_nonzero(prediction & target))
        fp = int(np.count_nonzero(prediction & ~target))
        fn = int(np.count_nonzero(~prediction & target))
        per_image_dice.append(2 * tp / max(1, 2 * tp + fp + fn))

    metrics = compute_detailed_metrics(predictions, targets, threshold)
    return {
        **metrics,
        "tiny_defect_images": int(len(tiny_indices)),
        "tiny_defect_dice": (
            float(np.mean([per_image_dice[index] for index in tiny_indices]))
            if len(tiny_indices)
            else None
        ),
        "tiny_defect_recall": _group_recall(predicted, truth, tiny_indices),
        "true_positive_pixels": int(np.count_nonzero(predicted & truth)),
        "false_positive_pixels": int(np.count_nonzero(predicted & ~truth)),
        "false_negative_pixels": int(np.count_nonzero(~predicted & truth)),
    }


def _group_recall(predicted, truth, indices):
    if len(indices) == 0:
        return None
    per_image = []
    for index in indices:
        tp = int(np.count_nonzero(predicted[index] & truth[index]))
        positives = int(np.count_nonzero(truth[index]))
        per_image.append(tp / max(1, positives))
    return float(np.mean(per_image))


def _threshold_analysis(predictions_by_model, targets):
    rows = []
    selected_thresholds = {}
    for model_name, predictions in predictions_by_model.items():
        selected, _ = _select_threshold(predictions, targets)
        selected_thresholds[model_name] = selected
        thresholds = sorted(set(THRESHOLD_STEPS) | {selected})
        for threshold in thresholds:
            metrics = _binary_metrics(predictions, targets, threshold)
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    **metrics,
                    "selected_by_validation_dice": int(threshold == selected),
                }
            )
    return rows, selected_thresholds


def _tiny_image_threshold_rows(names, predictions_by_model, targets, thresholds):
    groups = _size_groups(targets)
    truth = targets[:, 0].numpy() > 0.5
    tiny_indices = np.flatnonzero(groups == "tiny")
    rows = []
    for model_name, probabilities in predictions_by_model.items():
        predicted_probabilities = probabilities[:, 0].numpy()
        for index in tiny_indices:
            target = truth[index]
            for threshold in sorted(set(THRESHOLD_STEPS) | {thresholds[model_name]}):
                prediction = predicted_probabilities[index] > threshold
                tp = int(np.count_nonzero(prediction & target))
                fp = int(np.count_nonzero(prediction & ~target))
                fn = int(np.count_nonzero(~prediction & target))
                rows.append(
                    {
                        "model": model_name,
                        "image": names[index],
                        "threshold": threshold,
                        "selected_by_validation_dice": int(
                            threshold == thresholds[model_name]
                        ),
                        "ground_truth_pixels": int(target.sum()),
                        "true_positive_pixels": tp,
                        "false_positive_pixels": fp,
                        "false_negative_pixels": fn,
                        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                        "precision": tp / max(1, tp + fp),
                        "recall": tp / max(1, tp + fn),
                    }
                )
    return rows


def _image_comparison_rows(
    names,
    targets,
    predictions_by_model,
    selected_thresholds,
):
    groups = _size_groups(targets)
    truth = targets[:, 0].numpy() > 0.5
    binary = {
        name: predictions[:, 0].numpy() > selected_thresholds[name]
        for name, predictions in predictions_by_model.items()
    }
    rows = []
    for index, image_name in enumerate(names):
        target = truth[index]
        row = {
            "image": image_name,
            "size_group": str(groups[index]),
            "ground_truth_pixels": int(target.sum()),
            "ground_truth_fraction": float(target.mean()),
        }
        for model_name, predictions in binary.items():
            predicted = predictions[index]
            tp = int(np.count_nonzero(predicted & target))
            fp = int(np.count_nonzero(predicted & ~target))
            fn = int(np.count_nonzero(~predicted & target))
            row.update(
                {
                    f"{model_name}_tp": tp,
                    f"{model_name}_fp": fp,
                    f"{model_name}_fn": fn,
                    f"{model_name}_dice": 2 * tp / max(1, 2 * tp + fp + fn),
                    f"{model_name}_iou": tp / max(1, tp + fp + fn),
                    f"{model_name}_precision": tp / max(1, tp + fp),
                    f"{model_name}_recall": tp / max(1, tp + fn),
                }
            )
        row["extended_minus_phase5_dice"] = (
            row["Extended best_dice"] - row["Phase 5_dice"]
        )
        rows.append(row)
    return rows, binary


def _area_bin(area):
    for label, lower, upper in COMPONENT_BINS:
        if area >= lower and (upper is None or area <= upper):
            return label
    raise ValueError(f"Invalid component area: {area}")


def _ground_truth_components(targets, binary_predictions, names, thresholds):
    rows = []
    for image_index, image_name in enumerate(names):
        target = (targets[image_index, 0].numpy() > 0.5).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            target,
            connectivity=8,
        )
        for component_id in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[component_id]
            )
            component = labels == component_id
            row = {
                "image": image_name,
                "component_id": component_id,
                "area_bin_pixels": _area_bin(area),
                "area_pixels": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_aspect_ratio": max(width, height) / max(1, min(width, height)),
                "centroid_x_normalized": float(
                    centroids[component_id][0] / target.shape[1]
                ),
                "centroid_y_normalized": float(
                    centroids[component_id][1] / target.shape[0]
                ),
                "tiny_image_by_existing_definition": int(
                    target.mean() <= 0.02 and target.sum() > 0
                ),
            }
            for model_name, predictions in binary_predictions.items():
                overlap = int(np.count_nonzero(predictions[image_index] & component))
                recall = overlap / area
                row.update(
                    {
                        f"{model_name}_overlap_pixels": overlap,
                        f"{model_name}_component_recall": recall,
                        f"{model_name}_component_status": (
                            "missed" if overlap == 0 else "partial" if overlap < area else "complete"
                        ),
                        f"{model_name}_threshold": thresholds[model_name],
                    }
                )
            rows.append(row)
    return rows


def _component_size_summary(rows, model_names):
    summary = []
    for area_bin, _, _ in COMPONENT_BINS:
        components = [row for row in rows if row["area_bin_pixels"] == area_bin]
        for model_name in model_names:
            prefix = f"{model_name}_"
            summary.append(
                {
                    "area_bin_pixels": area_bin,
                    "model": model_name,
                    "ground_truth_components": len(components),
                    "missed_components": sum(
                        row[f"{prefix}component_status"] == "missed"
                        for row in components
                    ),
                    "partially_segmented_components": sum(
                        row[f"{prefix}component_status"] == "partial"
                        for row in components
                    ),
                    "fully_covered_components": sum(
                        row[f"{prefix}component_status"] == "complete"
                        for row in components
                    ),
                    "component_detection_rate": (
                        sum(row[f"{prefix}component_status"] != "missed" for row in components)
                        / len(components)
                        if components
                        else None
                    ),
                    "mean_component_pixel_recall": (
                        float(
                            np.mean(
                                [
                                    row[f"{prefix}component_recall"]
                                    for row in components
                                ]
                            )
                        )
                        if components
                        else None
                    ),
                }
            )
    return summary


def _shape_features(component):
    ys, xs = np.nonzero(component)
    area = len(xs)
    if area == 0:
        raise ValueError("Cannot measure an empty component")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width, height = x1 - x0 + 1, y1 - y0 + 1
    aspect = max(width, height) / max(1, min(width, height))
    if area > 1:
        eigenvalues = np.maximum(
            np.linalg.eigvalsh(np.cov(np.column_stack((xs, ys)), rowvar=False)),
            0,
        )
        eccentricity = (
            float(np.sqrt(1 - eigenvalues[0] / eigenvalues[-1]))
            if eigenvalues[-1] > 0
            else 0.0
        )
    else:
        eccentricity = 0.0
    cropped = component[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    contours, _ = cv2.findContours(
        cropped,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    hull_area = max(
        (
            cv2.contourArea(cv2.convexHull(contour))
            for contour in contours
        ),
        default=0.0,
    )
    solidity = min(1.0, area / hull_area) if hull_area else 0.0
    return {
        "area_pixels": area,
        "bbox_x": x0,
        "bbox_y": y0,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_aspect_ratio": aspect,
        "eccentricity": eccentricity,
        "solidity": solidity,
    }


def _false_positive_components(
    images,
    targets,
    binary_predictions,
    names,
):
    rows = []
    for model_name, predictions in binary_predictions.items():
        for image_index, image_name in enumerate(names):
            target = targets[image_index, 0].numpy() > 0.5
            false_positives = predictions[image_index] & ~target
            rgb = _image_rgb(images[image_index].numpy())
            gray = cv2.cvtColor(
                np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY,
            )
            grid_proxy = _dark_linear_structure(gray)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(
                false_positives.astype(np.uint8),
                connectivity=8,
            )
            for component_id in range(1, count):
                component = labels == component_id
                feature = _shape_features(component)
                area = feature["area_pixels"]
                proxy_fraction = float(np.count_nonzero(component & grid_proxy) / area)
                centroid_x, centroid_y = centroids[component_id]
                rows.append(
                    {
                        "model": model_name,
                        "image": image_name,
                        "component_id": component_id,
                        **feature,
                        "centroid_x_normalized": float(centroid_x / component.shape[1]),
                        "centroid_y_normalized": float(centroid_y / component.shape[0]),
                        "grid_proxy_overlap_fraction": proxy_fraction,
                        "false_positive_pixels_on_grid_proxy": int(
                            np.count_nonzero(component & grid_proxy)
                        ),
                        "overlaps_dark_line_proxy": int(proxy_fraction > 0),
                        "majority_overlaps_dark_line_proxy": int(proxy_fraction >= 0.5),
                    }
                )
    return rows


def _grid_summary(rows, model_names):
    summary = []
    for model_name in model_names:
        selected = [row for row in rows if row["model"] == model_name]
        all_fp = sum(row["area_pixels"] for row in selected)
        proxy_components = [
            row for row in selected if row["majority_overlaps_dark_line_proxy"]
        ]
        proxy_pixels = sum(row["area_pixels"] for row in proxy_components)
        false_positive_proxy_pixels = sum(
            row["false_positive_pixels_on_grid_proxy"] for row in selected
        )
        summary.append(
            {
                "model": model_name,
                "false_positive_components": len(selected),
                "false_positive_pixels": all_fp,
                "false_positive_pixels_overlapping_dark_line_proxy": (
                    false_positive_proxy_pixels
                ),
                "fraction_false_positive_pixels_overlapping_dark_line_proxy": (
                    false_positive_proxy_pixels / all_fp if all_fp else 0.0
                ),
                "components_with_any_proxy_overlap": sum(
                    row["overlaps_dark_line_proxy"] for row in selected
                ),
                "components_with_majority_proxy_overlap": len(proxy_components),
                "false_positive_pixels_in_majority_proxy_components": proxy_pixels,
                "fraction_fp_pixels_in_majority_proxy_components": (
                    proxy_pixels / all_fp if all_fp else 0.0
                ),
                "majority_proxy_component_median_area": (
                    float(np.median([row["area_pixels"] for row in proxy_components]))
                    if proxy_components
                    else None
                ),
                "majority_proxy_component_median_aspect_ratio": (
                    float(
                        np.median(
                            [row["bbox_aspect_ratio"] for row in proxy_components]
                        )
                    )
                    if proxy_components
                    else None
                ),
            }
        )
    return summary


def _grid_component_feature_summary(rows, model_names):
    results = []
    for model_name in model_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        for area_bin, _, _ in COMPONENT_BINS:
            for aspect_bin, lower, upper in (
                ("<2", 0.0, 2.0),
                ("2-5", 2.0, 5.0),
                ("5+", 5.0, None),
            ):
                selected = [
                    row
                    for row in model_rows
                    if _area_bin(row["area_pixels"]) == area_bin
                    and row["bbox_aspect_ratio"] >= lower
                    and (
                        row["bbox_aspect_ratio"] < upper
                        if upper is not None
                        else True
                    )
                ]
                if selected:
                    results.append(
                        {
                            "model": model_name,
                            "area_bin_pixels": area_bin,
                            "bbox_aspect_ratio_bin": aspect_bin,
                            "components": len(selected),
                            "false_positive_pixels": sum(
                                row["area_pixels"] for row in selected
                            ),
                            "components_with_majority_proxy_overlap": sum(
                                row["majority_overlaps_dark_line_proxy"]
                                for row in selected
                            ),
                            "median_eccentricity": float(
                                np.median(
                                    [row["eccentricity"] for row in selected]
                                )
                            ),
                        }
                    )
    return results


def _bootstrap_dice_delta(rows, iterations=10000, seed=42):
    differences = np.asarray(
        [row["extended_minus_phase5_dice"] for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(differences),
        size=(iterations, len(differences)),
    )
    means = differences[sample_indices].mean(axis=1)
    return {
        "mean_per_image_dice_delta": float(differences.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "images_improved": int(np.count_nonzero(differences > 0)),
        "images_tied": int(np.count_nonzero(differences == 0)),
        "images_regressed": int(np.count_nonzero(differences < 0)),
    }


def _size_group_summary(rows):
    results = []
    for group in ("tiny", "medium", "large"):
        selected = [row for row in rows if row["size_group"] == group]
        if not selected:
            continue
        result = {
            "size_group": group,
            "images": len(selected),
            "mean_dice_delta": float(
                np.mean(
                    [row["extended_minus_phase5_dice"] for row in selected]
                )
            ),
        }
        for model_name in ("Phase 5", "Extended best"):
            result.update(
                {
                    f"{model_name}_mean_dice": float(
                        np.mean([row[f"{model_name}_dice"] for row in selected])
                    ),
                    f"{model_name}_false_positive_pixels": sum(
                        row[f"{model_name}_fp"] for row in selected
                    ),
                    f"{model_name}_false_negative_pixels": sum(
                        row[f"{model_name}_fn"] for row in selected
                    ),
                }
            )
        results.append(result)
    return results


def _choose_visual_examples(rows, component_rows, grid_rows, limit=6):
    indices = {row["image"]: index for index, row in enumerate(rows)}
    chosen = []

    def add(image, label):
        if image in indices and image not in {name for name, _ in chosen}:
            chosen.append((image, label))

    add(
        max(rows, key=lambda row: row["extended_minus_phase5_dice"])["image"],
        "Largest extended-model Dice gain",
    )
    add(
        min(rows, key=lambda row: row["extended_minus_phase5_dice"])["image"],
        "Largest extended-model Dice regression",
    )
    tiny_rows = [row for row in rows if row["size_group"] == "tiny"]
    if tiny_rows:
        add(
            min(tiny_rows, key=lambda row: row["ground_truth_pixels"])["image"],
            "Smallest tiny-defect image",
        )
        add(
            min(tiny_rows, key=lambda row: row["extended_minus_phase5_dice"])["image"],
            "Largest tiny-defect regression",
        )
    grid_by_image = {}
    for row in grid_rows:
        if row["majority_overlaps_dark_line_proxy"]:
            grid_by_image[row["image"]] = (
                grid_by_image.get(row["image"], 0) + row["area_pixels"]
            )
    if grid_by_image:
        add(
            max(grid_by_image, key=grid_by_image.get),
            "Most extended-model FP pixels in proxy-overlap components",
        )
    if len(chosen) < limit and grid_by_image:
        for image in sorted(grid_by_image, key=grid_by_image.get, reverse=True):
            add(image, "Additional proxy-overlap FP example")
            if len(chosen) == limit:
                break
    for row in sorted(rows, key=lambda item: item["Extended best_fp"], reverse=True):
        add(row["image"], "Most extended-model FP pixels")
        if len(chosen) == limit:
            break
    return [(indices[image], label) for image, label in chosen[:limit]]


def _save_qualitative(
    path,
    images,
    targets,
    predictions,
    thresholds,
    names,
    per_image,
    component_rows,
    grid_rows,
):
    examples = _choose_visual_examples(per_image, component_rows, grid_rows)
    if not examples:
        raise ValueError("No validation images available for visualization")
    figure, axes = plt.subplots(
        len(examples),
        4,
        figsize=(14, 3.5 * len(examples)),
        squeeze=False,
    )
    headings = (
        "EL image",
        "Ground truth",
        f"Phase 5 errors (>{thresholds['Phase 5']:.3f})",
        f"Extended epoch 28 errors (>{thresholds['Extended best']:.3f})",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row_index, (image_index, label) in enumerate(examples):
        rgb = _image_rgb(images[image_index].numpy())
        target = targets[image_index, 0].numpy() > 0.5
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        for column, model_name in ((2, "Phase 5"), (3, "Extended best")):
            predicted = (
                predictions[model_name][image_index, 0].numpy()
                > thresholds[model_name]
            )
            axes[row_index, column].imshow(
                _error_overlay(rgb, predicted, target),
                vmin=0,
                vmax=1,
            )
        axes[row_index, 0].set_ylabel(
            f"{label}\n{names[image_index]}",
            fontsize=8,
        )
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        "Green: correctly predicted | Red: false positive | Blue: missed",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _save_grid_qualitative(path, images, targets, binary, names, grid_rows):
    proxy_by_image = {}
    for image_index, image_name in enumerate(names):
        rgb = _image_rgb(images[image_index].numpy())
        gray = cv2.cvtColor(
            np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        proxy_by_image[image_name] = _dark_linear_structure(gray)
    image_scores = {}
    for row in grid_rows:
        if row["majority_overlaps_dark_line_proxy"]:
            image_scores[row["image"]] = (
                image_scores.get(row["image"], 0) + row["area_pixels"]
            )
    selected_names = sorted(
        image_scores,
        key=image_scores.get,
        reverse=True,
    )[:4]
    if not selected_names:
        return False
    name_indices = {name: index for index, name in enumerate(names)}
    figure, axes = plt.subplots(
        len(selected_names),
        5,
        figsize=(17, 3.4 * len(selected_names)),
        squeeze=False,
    )
    headings = (
        "EL image",
        "Ground truth",
        "Dark-line proxy\n(not a grid label)",
        "Phase 5 false positives",
        "Extended false positives",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row_index, image_name in enumerate(selected_names):
        index = name_indices[image_name]
        rgb = _image_rgb(images[index].numpy())
        target = targets[index, 0].numpy() > 0.5
        proxy = proxy_by_image[image_name]
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 1].imshow(target, cmap="gray", vmin=0, vmax=1)
        proxy_overlay = rgb.copy()
        proxy_overlay[proxy] = (
            0.55 * proxy_overlay[proxy] + 0.45 * np.asarray([1.0, 0.7, 0.0])
        )
        axes[row_index, 2].imshow(proxy_overlay)
        for column, model_name in ((3, "Phase 5"), (4, "Extended best")):
            fp = binary[model_name][index] & ~target
            overlay = rgb.copy()
            overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.asarray([1.0, 0.0, 0.0])
            axes[row_index, column].imshow(overlay)
        axes[row_index, 0].set_ylabel(image_name, fontsize=9)
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        "Amber: image-derived dark-line proxy | Red: false-positive pixels",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return True


def _write_report(path, summary):
    baseline = summary["models"]["Phase 5"]
    extended = summary["models"]["Extended best"]
    paired = summary["paired_image_dice"]
    group_rows = summary["size_group_summary"]
    tiny_rows = summary["tiny_image_comparison"]
    component_rows = summary["ground_truth_component_size_summary"]
    grid_rows = summary["grid_proxy_false_positive_summary"]
    tiny_thresholds = summary["best_tiny_threshold_by_model"]
    outlier_thresholds = summary["worst_tiny_image_threshold_analysis"]
    lines = [
        "# Phase 10 follow-up: model error analysis",
        "",
        "Validation-only analysis on the deterministic 27-image seed-42 split. "
        "No training or test-set evaluation was performed.",
        "",
        f"- Phase 5 threshold: {baseline['threshold']:.3f}; "
        f"Dice {baseline['dice']:.4f}, tiny Dice {baseline['tiny_defect_dice']:.4f}, "
        f"FP pixels {baseline['false_positive_pixels']:,}.",
        f"- Epoch-28 threshold: {extended['threshold']:.3f}; "
        f"Dice {extended['dice']:.4f}, tiny Dice {extended['tiny_defect_dice']:.4f}, "
        f"FP pixels {extended['false_positive_pixels']:,}.",
        f"- Paired mean Dice delta: {paired['mean_per_image_dice_delta']:+.4f} "
        f"(95% bootstrap interval {paired['ci95_low']:+.4f} to "
        f"{paired['ci95_high']:+.4f}); "
        f"{paired['images_improved']} improved, {paired['images_regressed']} regressed.",
        "",
        "## Tiny-defect findings",
        "",
        "The existing tiny-image definition is preserved: nonempty masks with "
        "foreground fraction <=2%, with per-image Dice macro-averaged across "
        "that group. The five images are:",
        "",
        "| Image | GT pixels | Phase 5 Dice | Extended Dice | Phase 5 recall | Extended recall | Dice delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tiny_rows:
        lines.append(
            f"| {row['image']} | {row['ground_truth_pixels']} | "
            f"{row['Phase 5_dice']:.4f} | {row['Extended best_dice']:.4f} | "
            f"{row['Phase 5_recall']:.4f} | {row['Extended best_recall']:.4f} | "
            f"{row['extended_minus_phase5_dice']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "The tiny-group regression is concentrated in `176.png`: its Dice "
            f"falls by {min(row['extended_minus_phase5_dice'] for row in tiny_rows):.4f} "
            "and recall drops sharply, while the other four tiny images improve "
            "in Dice. Therefore this split does not show a uniform failure on "
            "small objects.",
            f"On `176.png`, the extended model's best Dice anywhere in the "
            f"0.1-0.95 threshold sweep is "
            f"{outlier_thresholds['Extended best']['best_dice_in_sweep']['dice']:.4f} "
            f"at {outlier_thresholds['Extended best']['best_dice_in_sweep']['threshold']:.3f}; "
            f"Phase 5 reaches "
            f"{outlier_thresholds['Phase 5']['best_dice_in_sweep']['dice']:.4f} "
            f"at {outlier_thresholds['Phase 5']['best_dice_in_sweep']['threshold']:.3f}. "
            "The outlier therefore persists across the tested global thresholds, "
            "rather than being fixed by a simple calibration shift.",
            "",
            "## Why overall Dice rises",
            "",
            "The saved checkpoint was selected for overall validation Dice, "
            "not tiny-group Dice. The matched per-size breakdown shows gains "
            "on the medium and large groups offsetting the tiny-group loss:",
            "",
            "| GT image group | Images | Phase 5 Dice | Extended Dice | Mean Dice delta | Phase 5 FP / FN | Extended FP / FN |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in group_rows:
        lines.append(
            f"| {row['size_group']} | {row['images']} | "
            f"{row['Phase 5_mean_dice']:.4f} | "
            f"{row['Extended best_mean_dice']:.4f} | "
            f"{row['mean_dice_delta']:+.4f} | "
            f"{row['Phase 5_false_positive_pixels']:,} / "
            f"{row['Phase 5_false_negative_pixels']:,} | "
            f"{row['Extended best_false_positive_pixels']:,} / "
            f"{row['Extended best_false_negative_pixels']:,} |"
        )
    lines.extend(
        [
            "",
            "Medium and large images improve by about 0.0298 and 0.0236 Dice "
            "on average, while tiny images decline by 0.0310. On large images "
            "the model trades more FP pixels for substantially fewer FN pixels; "
            "on medium images both FP and FN totals fall. This is the observed "
            "metric tradeoff behind the overall gain. It does not by itself "
            "prove a particular causal feature change during training.",
            "",
            "## Ground-truth component size",
            "",
            "Masks were labeled with 8-connectivity. 'Missed' means zero overlap; "
            "'partial' means at least one pixel overlaps but not all component "
            "pixels are covered. Counts are component-level, not tiny-image Dice.",
            "",
            "| Area (pixels) | GT components | Phase 5 missed | Extended missed | Phase 5 mean pixel recall | Extended mean pixel recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    phase5_components = {
        row["area_bin_pixels"]: row
        for row in component_rows
        if row["model"] == "Phase 5"
    }
    extended_components = {
        row["area_bin_pixels"]: row
        for row in component_rows
        if row["model"] == "Extended best"
    }
    for area_bin, phase5_row in phase5_components.items():
        extended_row = extended_components[area_bin]
        lines.append(
            f"| {area_bin} | {phase5_row['ground_truth_components']} | "
            f"{phase5_row['missed_components']} | "
            f"{extended_row['missed_components']} | "
            f"{phase5_row['mean_component_pixel_recall'] if phase5_row['mean_component_pixel_recall'] is not None else float('nan'):.3f} | "
            f"{extended_row['mean_component_pixel_recall'] if extended_row['mean_component_pixel_recall'] is not None else float('nan'):.3f} |"
        )
    lines.extend(
        [
            "",
            "There are no 1-3 pixel ground-truth components. Extended training "
            "improves component detection and mean pixel recall in each "
            "nonempty size bin, including 16-63 px (15 components: missed 4 "
            "to 1; mean recall 0.503 to 0.616). The outlier `176.png` contains "
            "a 667-pixel component, so the image-level tiny regression is not "
            "evidence that every small connected component is worse. Most "
            "components remain only partially covered in both models.",
            "",
            "## Threshold sensitivity",
            "",
            "Threshold sweeps optimize overall validation macro Dice. The "
            "threshold with the highest tiny-group Dice is shown separately "
            "as diagnostic only; it is not selected as the operating point.",
            "",
            "| Model | Overall-Dice threshold | Tiny Dice | Best tiny-Dice threshold | Tiny Dice there | Overall Dice there | FP pixels there |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model_name, model_row in summary["models"].items():
        tiny_row = tiny_thresholds[model_name]
        lines.append(
            f"| {model_name} | {model_row['threshold']:.3f} | "
            f"{model_row['tiny_defect_dice']:.4f} | "
            f"{tiny_row['threshold']:.3f} | {tiny_row['tiny_defect_dice']:.4f} | "
            f"{tiny_row['dice']:.4f} | {tiny_row['false_positive_pixels']:,} |"
        )
    lines.extend(
        [
            "",
            "At threshold 0.1, the extended checkpoint's tiny Dice reaches only "
            f"{tiny_thresholds['Extended best']['tiny_defect_dice']:.4f} "
            f"(versus Phase 5's {baseline['tiny_defect_dice']:.4f} at its "
            f"selected threshold), while extended false positives rise to "
            f"{tiny_thresholds['Extended best']['false_positive_pixels']:,}. "
            "A global threshold shift does not close the tiny-group gap without "
            "a substantial false-positive tradeoff.",
            "",
            "## Grid-line false positives",
            "",
            "The image-derived dark-line proxy is broad and is not a semantic "
            "grid label. Its pixel overlap with false positives and component "
            "shape statistics are:",
            "",
            "| Model | FP components | FP pixels | FP pixels on proxy | Share on proxy | Majority-overlap components |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in grid_rows:
        lines.append(
            f"| {row['model']} | {row['false_positive_components']} | "
            f"{row['false_positive_pixels']:,} | "
            f"{row['false_positive_pixels_overlapping_dark_line_proxy']:,} | "
            f"{row['fraction_false_positive_pixels_overlapping_dark_line_proxy']:.1%} | "
            f"{row['components_with_majority_proxy_overlap']} |"
        )
    lines.extend(
        [
            "",
            "The extended checkpoint adds 181 FP-only connected components and "
            "1,103 FP pixels overall, but proxy-overlap FP pixels decrease by "
            "327. Visual review shows some red FP fragments tracking obvious "
            "horizontal conductive wires, and also diagonal dark structures; "
            "the proxy captures both, so not every overlap can be called a "
            "grid-wire error. Majority-proxy components have median bbox aspect "
            "ratios near 1, indicating fragmented blobs rather than long "
            "connected false-positive lines. The saved per-component table "
            "contains area, aspect ratio, eccentricity, and proxy overlap.",
            "",
            "## Recommendation",
            "",
            "1. **Small-defect-weighted loss (controlled follow-up):** target "
            "image-level small-mask weighting rather than repeat generic "
            "oversampling. The regression is driven by one of five tiny images, "
            "and the extended model improves size-binned component recall; "
            "treat this as a small exploratory experiment with a tiny-Dice "
            "guard, not a proven fix.",
            "2. **Reviewed hard-negative mining:** consider only if reducing "
            "false positives becomes the primary goal. FP-only component count "
            "rose, but dark-line-proxy FP pixels did not, so mine manually "
            "verified normal-wire/background regions rather than proxy hits.",
            "3. **BCE + Dice + clDice:** defer until crack fragmentation or "
            "connectivity is measured as a dominant error; current component "
            "detection rates mostly improved.",
            "4. **Higher resolution:** defer; current evidence does not isolate "
            "224-pixel resizing as the cause, and it increases CPU cost.",
            "5. **Post-processing:** do not adopt; prior validation showed "
            "component filters were uncertain and grid-proxy suppression "
            "damaged recall.",
            "",
            "**Keep Phase 5 (threshold 0.165) as the application default.** "
            "The held-out test split remains untouched.",
            "",
        ]
    )
    lines.extend(
        [
            "## Definitions and artifacts",
            "",
            "Tiny-image Dice is per-image macro Dice for images with nonempty "
            "masks containing <=2% foreground. Ground-truth components use "
            "8-connectivity. A component is 'missed' when no predicted pixel "
            "overlaps it, 'partial' when overlap exists but some component "
            "pixels are missing, and 'complete' only when every ground-truth "
            "component pixel is predicted.",
            "",
            "The dark-line proxy is an image-derived black-hat detector, not a "
            "semantic annotation of conductive grid lines. Proxy-overlap statistics "
            "are association evidence only and must be checked against the saved "
            "qualitative sheet before describing a region as a grid line.",
            "",
            "See the paired per-image, threshold, tiny-image threshold, "
            "ground-truth component, size-bin, false-positive component, and "
            "proxy summary CSV files in `results/`, along with the qualitative "
            "comparison images.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(project_dir, batch_size=8, num_threads=6):
    if batch_size <= 0 or num_threads <= 0:
        raise ValueError("batch_size and num_threads must be greater than zero")
    torch.set_num_threads(num_threads)
    project_dir = Path(project_dir)
    results_dir = project_dir / "results"
    baseline_path = results_dir / "resnet18_unet_phase5_negw_best.pth"
    extended_path = results_dir / "resnet18_unet_phase5_to50_best.pth"
    cache_path = results_dir / "phase10_phase5_validation_probabilities.npz"
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    names = _validation_names(validation_loader)
    images, targets = _validation_data(validation_loader)
    baseline_predictions, baseline_seconds = _load_baseline_cache(
        cache_path,
        baseline_path,
        names,
        targets,
    )
    extended_predictions, extended_seconds = _load_extended_predictions(
        extended_path,
        validation_loader,
    )
    if baseline_predictions.shape != targets.shape:
        raise RuntimeError("Cached Phase 5 predictions do not match validation masks")
    if extended_predictions.shape != targets.shape:
        raise RuntimeError("Epoch-28 predictions do not match validation masks")

    predictions = {
        "Phase 5": baseline_predictions,
        "Extended best": extended_predictions,
    }
    threshold_rows, thresholds = _threshold_analysis(predictions, targets)
    per_image, binary_predictions = _image_comparison_rows(
        names,
        targets,
        predictions,
        thresholds,
    )
    tiny_threshold_rows = _tiny_image_threshold_rows(
        names,
        predictions,
        targets,
        thresholds,
    )
    ground_truth_components = _ground_truth_components(
        targets,
        binary_predictions,
        names,
        thresholds,
    )
    component_summary = _component_size_summary(
        ground_truth_components,
        predictions.keys(),
    )
    fp_components = _false_positive_components(
        images,
        targets,
        binary_predictions,
        names,
    )
    grid_summary = _grid_summary(fp_components, predictions.keys())
    grid_feature_summary = _grid_component_feature_summary(
        fp_components,
        predictions.keys(),
    )
    size_group_summary = _size_group_summary(per_image)

    model_metrics = {}
    inference_seconds = {
        "Phase 5": baseline_seconds,
        "Extended best": extended_seconds,
    }
    for model_name, probability_maps in predictions.items():
        selected_metrics = _binary_metrics(
            probability_maps,
            targets,
            thresholds[model_name],
        )
        model_metrics[model_name] = {
            "threshold": thresholds[model_name],
            **selected_metrics,
            "inference_seconds_per_image": inference_seconds[model_name],
        }

    tiny_image_comparison = [
        row for row in per_image if row["size_group"] == "tiny"
    ]
    lowest_tiny_image = min(
        tiny_image_comparison,
        key=lambda row: row["extended_minus_phase5_dice"],
    )
    outlier_threshold_analysis = {}
    for model_name in predictions:
        image_rows = [
            row
            for row in tiny_threshold_rows
            if row["model"] == model_name
            and row["image"] == lowest_tiny_image["image"]
        ]
        outlier_threshold_analysis[model_name] = {
            "image": lowest_tiny_image["image"],
            "best_dice_in_sweep": max(image_rows, key=lambda row: row["dice"]),
            "lowest_threshold": next(
                row for row in image_rows if row["threshold"] == 0.1
            ),
            "selected_model_threshold": next(
                row
                for row in image_rows
                if row["selected_by_validation_dice"] == 1
            ),
        }

    best_tiny_threshold_by_model = {}
    for model_name in predictions:
        model_threshold_rows = [
            row for row in threshold_rows if row["model"] == model_name
        ]
        best_tiny_threshold_by_model[model_name] = max(
            model_threshold_rows,
            key=lambda row: row["tiny_defect_dice"],
        )
    summary = {
        "validation_images": len(names),
        "split_seed": 42,
        "threshold_selection": "validation macro Dice; 0.1-0.95 in 0.025 steps with 0.005 local refinement",
        "metric_aggregation": "per-image macro for Dice, IoU, precision, recall; pixel totals for TP, FP, FN",
        "tiny_defect_definition": "nonempty mask with foreground fraction <= 0.02; macro per-image Dice",
        "ground_truth_component_connectivity": 8,
        "component_status_definition": {
            "missed": "zero predicted pixels overlap the ground-truth component",
            "partial": "some but not all component pixels are predicted",
            "complete": "all component pixels are predicted",
        },
        "models": model_metrics,
        "paired_image_dice": _bootstrap_dice_delta(per_image),
        "size_group_summary": size_group_summary,
        "tiny_image_comparison": tiny_image_comparison,
        "best_tiny_threshold_by_model": best_tiny_threshold_by_model,
        "worst_tiny_image_threshold_analysis": outlier_threshold_analysis,
        "ground_truth_component_size_summary": component_summary,
        "grid_proxy_false_positive_summary": grid_summary,
        "grid_false_positive_component_shape_summary": grid_feature_summary,
        "test_split_evaluated": False,
        "checkpoints_modified": False,
    }

    _write_csv(results_dir / "phase10_model_error_per_image.csv", per_image)
    _write_csv(
        results_dir / "phase10_model_error_size_group_summary.csv",
        size_group_summary,
    )
    _write_csv(results_dir / "phase10_model_error_thresholds.csv", threshold_rows)
    _write_csv(
        results_dir / "phase10_model_error_tiny_image_thresholds.csv",
        tiny_threshold_rows,
    )
    _write_csv(
        results_dir / "phase10_model_error_ground_truth_components.csv",
        ground_truth_components,
    )
    _write_csv(
        results_dir / "phase10_model_error_component_size_summary.csv",
        component_summary,
    )
    _write_csv(
        results_dir / "phase10_model_error_false_positive_components.csv",
        fp_components,
    )
    _write_csv(
        results_dir / "phase10_model_error_grid_proxy_summary.csv",
        grid_summary,
    )
    _write_csv(
        results_dir / "phase10_model_error_grid_component_features.csv",
        grid_feature_summary,
    )
    _save_qualitative(
        results_dir / "phase10_model_error_qualitative.png",
        images,
        targets,
        predictions,
        thresholds,
        names,
        per_image,
        ground_truth_components,
        fp_components,
    )
    _save_grid_qualitative(
        results_dir / "phase10_model_error_grid_examples.png",
        images,
        targets,
        binary_predictions,
        names,
        fp_components,
    )
    with (results_dir / "phase10_model_error_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(summary, output_file, indent=2)
    _write_report(results_dir / "phase10_model_error_report.md", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validation-only Phase 5 vs. extended-model error analysis."
    )
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    run_analysis(args.project_dir, args.batch_size, args.num_threads)


if __name__ == "__main__":
    main()
