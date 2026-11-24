import argparse
import csv
import json
import time
from pathlib import Path

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
from src.train import compute_detailed_metrics


THRESHOLD = 0.165
MIN_COMPONENT_AREAS = (2, 4, 8, 16, 32)
BOOTSTRAP_ITERATIONS = 10000


def _validation_names(loader):
    subset = loader.dataset
    if not isinstance(subset, Subset):
        raise TypeError("Expected a validation loader backed by a Subset")
    return [subset.dataset.samples[index][0].name for index in subset.indices]


def _infer_validation(model, loader):
    predictions = []
    targets = []
    images = []
    model.eval()
    if len(loader) == 0:
        raise ValueError("The validation data loader is empty")
    warmup_images = next(iter(loader))[0]
    with torch.inference_mode():
        model(warmup_images)
        elapsed = 0.0
        image_count = 0
        for batch_images, batch_masks, _ in loader:
            started = time.perf_counter()
            probabilities = model(batch_images)
            elapsed += time.perf_counter() - started
            image_count += len(batch_images)
            predictions.append(probabilities.cpu())
            targets.append(batch_masks)
            images.append(batch_images.cpu())
    return (
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(images),
        elapsed / image_count,
    )


def _validation_tensors(loader):
    images = []
    targets = []
    for batch_images, batch_masks, _ in loader:
        images.append(batch_images)
        targets.append(batch_masks)
    if not images:
        raise ValueError("The validation data loader is empty")
    return torch.cat(images), torch.cat(targets)


def _gray_image(normalized_image):
    rgb = normalized_image.transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN
    rgb = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _prediction_components(binary_mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )
    return count, labels, stats, centroids


def _component_shape(labels, component_id, x, y, width, height, area):
    component = (labels[y : y + height, x : x + width] == component_id).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(
        component,
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
    aspect_ratio = max(width, height) / max(1, min(width, height))
    coordinates_y, coordinates_x = np.nonzero(component)
    if len(coordinates_x) > 1:
        covariance = np.cov(
            np.column_stack((coordinates_x, coordinates_y)),
            rowvar=False,
        )
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
        eccentricity = (
            float(np.sqrt(1 - eigenvalues[0] / eigenvalues[-1]))
            if eigenvalues[-1] > 0
            else 0.0
        )
    else:
        eccentricity = 0.0
    return aspect_ratio, eccentricity, solidity, area / max(1, width * height)


def _component_report(probabilities, targets, images, names, threshold):
    component_rows = []
    feature_rows = []
    for index, image_name in enumerate(names):
        probability = probabilities[index, 0].numpy()
        target = targets[index, 0].numpy() > 0.5
        predicted = probability > threshold
        gray = _gray_image(images[index].numpy())
        grid_proxy = _dark_linear_structure(gray)
        count, labels, stats, centroids = _prediction_components(predicted)
        per_component = []
        for component_id in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[component_id]
            )
            component_mask = labels == component_id
            target_pixels = int((component_mask & target).sum())
            false_positive_pixels = int((component_mask & ~target).sum())
            proxy_fraction = float(
                (component_mask & grid_proxy).sum() / area
            )
            centroid_x, centroid_y = centroids[component_id]
            aspect_ratio, eccentricity, solidity, extent = _component_shape(
                labels,
                component_id,
                x,
                y,
                width,
                height,
                area,
            )
            normalized_x = float(centroid_x / predicted.shape[1])
            normalized_y = float(centroid_y / predicted.shape[0])
            location = (
                "edge"
                if min(normalized_x, normalized_y, 1 - normalized_x, 1 - normalized_y)
                < 0.15
                else "center"
                if 0.35 <= normalized_x <= 0.65
                and 0.35 <= normalized_y <= 0.65
                else "interior"
            )
            row = {
                "image": image_name,
                "component_id": component_id,
                "area_pixels": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_width": width,
                "bbox_height": height,
                "centroid_x_normalized": normalized_x,
                "centroid_y_normalized": normalized_y,
                "location": location,
                "aspect_ratio": aspect_ratio,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "extent": extent,
                "ground_truth_overlap_pixels": target_pixels,
                "false_positive_pixels": false_positive_pixels,
                "false_positive_only_component": int(target_pixels == 0),
                "grid_proxy_overlap_fraction": proxy_fraction,
                "mean_probability": float(probability[component_mask].mean()),
            }
            component_rows.append(row)
            per_component.append(row)

        for feature_name, feature_value in (
            ("area", lambda row: _area_bin(row["area_pixels"])),
            ("aspect_ratio", lambda row: _aspect_bin(row["aspect_ratio"])),
            ("eccentricity", lambda row: _eccentricity_bin(row["eccentricity"])),
            ("location", lambda row: row["location"]),
            (
                "grid_proxy_overlap",
                lambda row: _proxy_bin(row["grid_proxy_overlap_fraction"]),
            ),
        ):
            groups = sorted({feature_value(row) for row in per_component})
            for group in groups:
                selected = [
                    row for row in per_component if feature_value(row) == group
                ]
                feature_rows.append(
                    {
                        "image": image_name,
                        "feature": feature_name,
                        "group": group,
                        "components": len(selected),
                        "false_positive_only_components": sum(
                            row["false_positive_only_component"]
                            for row in selected
                        ),
                        "component_false_positive_pixels": sum(
                            row["false_positive_pixels"] for row in selected
                        ),
                        "components_with_ground_truth_overlap": sum(
                            row["ground_truth_overlap_pixels"] > 0
                            for row in selected
                        ),
                    }
                )
    return component_rows, feature_rows


def _area_bin(area):
    if area <= 3:
        return "1-3"
    if area <= 15:
        return "4-15"
    if area <= 63:
        return "16-63"
    if area <= 255:
        return "64-255"
    return "256+"


def _aspect_bin(value):
    if value < 2:
        return "<2"
    if value < 5:
        return "2-5"
    return "5+"


def _eccentricity_bin(value):
    if value < 0.75:
        return "<0.75"
    if value < 0.95:
        return "0.75-0.95"
    return "0.95+"


def _proxy_bin(value):
    if value == 0:
        return "0"
    if value < 0.25:
        return "(0,0.25)"
    if value < 0.75:
        return "[0.25,0.75)"
    return "[0.75,1]"


def _apply_variant(probability, gray, threshold, strategy):
    predicted = probability > threshold
    if strategy == "none":
        return predicted
    count, labels, stats, _ = _prediction_components(predicted)
    if strategy.startswith("min_area_"):
        minimum_area = int(strategy.removeprefix("min_area_"))
        areas = stats[:, cv2.CC_STAT_AREA]
        keep = areas >= minimum_area
        keep[0] = False
        return keep[labels]
    if strategy == "compact_area32_aspect2":
        keep = np.ones(count, dtype=bool)
        keep[0] = False
        for component_id in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[component_id]
            )
            ratio = max(width, height) / max(1, min(width, height))
            if area <= 32 and ratio <= 2:
                keep[component_id] = False
        return keep[labels]
    if strategy == "suppress_grid_proxy_pixels":
        grid_proxy = _dark_linear_structure(gray)
        return predicted & ~grid_proxy
    if strategy == "suppress_grid_aligned_components":
        grid_proxy = _dark_linear_structure(gray)
        keep = np.ones(count, dtype=bool)
        keep[0] = False
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            component_mask = labels == component_id
            proxy_fraction = float(
                (component_mask & grid_proxy).sum() / area
            )
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            aspect_ratio = max(width, height) / max(1, min(width, height))
            if aspect_ratio >= 5 and proxy_fraction >= 0.8:
                keep[component_id] = False
        return keep[labels]
    raise ValueError(f"Unknown post-processing strategy: {strategy}")


def _candidate_strategies():
    return (
        ("none", "Threshold only"),
        *(
            (f"min_area_{area}", f"Remove components <{area} px")
            for area in MIN_COMPONENT_AREAS
        ),
        ("compact_area32_aspect2", "Remove compact components <=32 px"),
        ("suppress_grid_proxy_pixels", "Suppress dark-line proxy pixels"),
        (
            "suppress_grid_aligned_components",
            "Suppress highly grid-aligned elongated components",
        ),
    )


def _mask_set(probabilities, grays, threshold, strategy):
    return np.stack(
        [
            _apply_variant(probability, gray, threshold, strategy)
            for probability, gray in zip(probabilities, grays)
        ]
    )


def _metric_values(binary_masks, targets):
    binary_tensor = torch.from_numpy(binary_masks[:, None].astype(np.float32))
    return compute_detailed_metrics(
        binary_tensor,
        targets,
        threshold=0.5,
    )


def _threshold_metrics(probabilities, targets, grays, threshold, strategy, groups):
    binary_masks = _mask_set(probabilities, grays, threshold, strategy)
    metrics = _metric_values(binary_masks, targets)
    per_image = _per_image_scores(binary_masks, targets)
    tiny_indices = np.flatnonzero(groups == "tiny")
    tiny_dice = (
        float(np.mean([per_image[index]["dice"] for index in tiny_indices]))
        if len(tiny_indices)
        else None
    )
    binary_tensor = torch.from_numpy(binary_masks[:, None].astype(np.float32))
    boundary_f1 = _boundary_f1(
        binary_tensor,
        targets,
        threshold=0.5,
        tolerance=1,
    )
    return binary_masks, {
        **metrics,
        "tiny_defect_dice": tiny_dice,
        "boundary_f1_tolerance_1px": boundary_f1,
        "true_positive_pixels": sum(
            row["true_positive_pixels"] for row in per_image
        ),
        "false_positive_pixels": sum(
            row["false_positive_pixels"] for row in per_image
        ),
        "false_negative_pixels": sum(
            row["false_negative_pixels"] for row in per_image
        ),
    }


def _threshold_candidates(probabilities, targets, grays, strategy, groups):
    coarse = [round(0.1 + step * 0.025, 3) for step in range(35)]
    scored = {}
    for threshold in coarse:
        _, scored[threshold] = _threshold_metrics(
            probabilities,
            targets,
            grays,
            threshold,
            strategy,
            groups,
        )
    center = max(coarse, key=lambda value: scored[value]["dice"])
    fine = {
        round(center - 0.025 + step * 0.005, 3)
        for step in range(11)
    }
    for threshold in sorted(fine):
        if threshold not in scored:
            _, scored[threshold] = _threshold_metrics(
                probabilities,
                targets,
                grays,
                threshold,
                strategy,
                groups,
            )
    selected = max(scored, key=lambda value: scored[value]["dice"])
    return selected, scored


def _size_groups(targets):
    target_array = targets[:, 0].numpy() > 0.5
    fractions = target_array.reshape(len(target_array), -1).mean(axis=1)
    return np.where(
        fractions == 0,
        "empty",
        np.where(fractions <= 0.02, "tiny", np.where(fractions <= 0.05, "medium", "large")),
    )


def _per_image_scores(binary_masks, targets):
    truth = targets[:, 0].numpy() > 0.5
    rows = []
    for index in range(len(binary_masks)):
        predicted = binary_masks[index]
        target = truth[index]
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        rows.append(
            {
                "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "iou": tp / max(1, tp + fp + fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
            }
        )
    return rows


def _bootstrap_delta(candidate_scores, baseline_scores, seed=42):
    candidate = np.asarray([row["dice"] for row in candidate_scores])
    baseline = np.asarray([row["dice"] for row in baseline_scores])
    return _bootstrap_values(candidate - baseline, seed)


def _bootstrap_values(differences, seed=42):
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(differences),
        size=(BOOTSTRAP_ITERATIONS, len(differences)),
    )
    means = differences[sample_indices].mean(axis=1)
    return {
        "paired_mean_dice_delta": float(differences.mean()),
        "dice_delta_ci95_low": float(np.quantile(means, 0.025)),
        "dice_delta_ci95_high": float(np.quantile(means, 0.975)),
        "paired_images_improved": int((differences > 0).sum()),
        "paired_images_unchanged": int((differences == 0).sum()),
        "paired_images_regressed": int((differences < 0).sum()),
    }


def _candidate_row(
    name,
    strategy,
    threshold,
    binary_masks,
    targets,
    groups,
    baseline_scores,
    grays,
    inference_seconds,
):
    metrics = _metric_values(binary_masks, targets)
    per_image = _per_image_scores(binary_masks, targets)
    tiny_indices = np.flatnonzero(groups == "tiny")
    tiny_dice = (
        float(np.mean([per_image[index]["dice"] for index in tiny_indices]))
        if len(tiny_indices)
        else None
    )
    tiny_recall = (
        float(np.mean([per_image[index]["recall"] for index in tiny_indices]))
        if len(tiny_indices)
        else None
    )
    binary_tensor = torch.from_numpy(binary_masks[:, None].astype(np.float32))
    boundary_f1 = _boundary_f1(
        binary_tensor,
        targets,
        threshold=0.5,
        tolerance=1,
    )
    grid_fp = 0
    grid_tp = 0
    for prediction, target, gray in zip(binary_masks, targets[:, 0].numpy() > 0.5, grays):
        grid_proxy = _dark_linear_structure(gray)
        grid_fp += int((prediction & ~target & grid_proxy).sum())
        grid_tp += int((prediction & target & grid_proxy).sum())
    bootstrap = _bootstrap_delta(per_image, baseline_scores)
    tiny_candidate = [per_image[index] for index in tiny_indices]
    tiny_baseline = [baseline_scores[index] for index in tiny_indices]
    tiny_bootstrap = _bootstrap_delta(
        tiny_candidate,
        tiny_baseline,
        seed=43,
    ) if len(tiny_indices) else {
        "paired_mean_dice_delta": None,
        "dice_delta_ci95_low": None,
        "dice_delta_ci95_high": None,
    }
    false_positive = sum(row["false_positive_pixels"] for row in per_image)
    false_negative = sum(row["false_negative_pixels"] for row in per_image)
    total_truth = int((targets > 0.5).sum())
    predicted_area_ratio = (
        (total_truth - false_negative + false_positive) / total_truth
        if total_truth
        else 0.0
    )
    return {
        "candidate": name,
        "strategy": strategy,
        "threshold": threshold,
        **metrics,
        "tiny_defect_dice": tiny_dice,
        "tiny_defect_recall": tiny_recall,
        "boundary_f1_tolerance_1px": boundary_f1,
        "true_positive_pixels": total_truth - false_negative,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "predicted_to_ground_truth_area_ratio": predicted_area_ratio,
        "false_positive_pixels_on_dark_linear_proxy": grid_fp,
        "true_positive_pixels_on_dark_linear_proxy": grid_tp,
        "false_positive_proxy_fraction": (
            grid_fp / false_positive if false_positive else 0.0
        ),
        "model_inference_seconds_per_image": inference_seconds,
        **bootstrap,
        "tiny_paired_mean_dice_delta": tiny_bootstrap["paired_mean_dice_delta"],
        "tiny_dice_delta_ci95_low": tiny_bootstrap["dice_delta_ci95_low"],
        "tiny_dice_delta_ci95_high": tiny_bootstrap["dice_delta_ci95_high"],
    }, per_image


def _postprocess_timing(probabilities, grays, threshold, strategy, repeats=7):
    elapsed_per_image = []
    for _ in range(repeats):
        started = time.perf_counter()
        for probability, gray in zip(probabilities, grays):
            _apply_variant(probability, gray, threshold, strategy)
        elapsed_per_image.append(
            (time.perf_counter() - started) / len(probabilities)
        )
    return float(np.median(elapsed_per_image))


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _component_summary(component_rows, feature_rows):
    rows = []
    feature_names = sorted({row["feature"] for row in feature_rows})
    for feature in feature_names:
        groups = sorted(
            {row["group"] for row in feature_rows if row["feature"] == feature}
        )
        for group in groups:
            selected = [
                row
                for row in feature_rows
                if row["feature"] == feature and row["group"] == group
            ]
            rows.append(
                {
                    "feature": feature,
                    "group": group,
                    "components": sum(row["components"] for row in selected),
                    "false_positive_only_components": sum(
                        row["false_positive_only_components"]
                        for row in selected
                    ),
                    "component_false_positive_pixels": sum(
                        row["component_false_positive_pixels"]
                        for row in selected
                    ),
                    "components_with_ground_truth_overlap": sum(
                        row["components_with_ground_truth_overlap"]
                        for row in selected
                    ),
                }
            )
    return rows


def _grid_proxy_summary(binary_masks, targets, grays):
    total_proxy_pixels = 0
    proxy_truth_pixels = 0
    proxy_true_positive_pixels = 0
    proxy_false_positive_pixels = 0
    for prediction, target, gray in zip(
        binary_masks,
        targets[:, 0].numpy() > 0.5,
        grays,
    ):
        proxy = _dark_linear_structure(gray)
        total_proxy_pixels += int(proxy.sum())
        proxy_truth_pixels += int((proxy & target).sum())
        proxy_true_positive_pixels += int((proxy & target & prediction).sum())
        proxy_false_positive_pixels += int((proxy & ~target & prediction).sum())
    return {
        "proxy_pixels": total_proxy_pixels,
        "proxy_fraction_of_all_pixels": (
            total_proxy_pixels / targets[:, 0].numel()
        ),
        "ground_truth_pixels_inside_proxy": proxy_truth_pixels,
        "true_positive_pixels_inside_proxy": proxy_true_positive_pixels,
        "false_positive_pixels_inside_proxy": proxy_false_positive_pixels,
        "predicted_pixel_precision_inside_proxy": (
            proxy_true_positive_pixels
            / max(
                1,
                proxy_true_positive_pixels + proxy_false_positive_pixels,
            )
        ),
    }


def _error_overlay(image, predicted, target):
    colors = np.zeros_like(image)
    colors[predicted & target] = (0.0, 1.0, 0.0)
    colors[predicted & ~target] = (1.0, 0.0, 0.0)
    colors[~predicted & target] = (0.0, 0.35, 1.0)
    selected = predicted | target
    overlay = image.copy()
    overlay[selected] = 0.45 * overlay[selected] + 0.55 * colors[selected]
    return overlay


def _visual_examples(per_image_by_candidate, names, baseline_components):
    baseline = per_image_by_candidate["Phase 5 baseline"]
    proxy_counts = {}
    for row in baseline_components:
        proxy_counts[row["image"]] = proxy_counts.get(row["image"], 0) + (
            row["false_positive_pixels"]
            if row["grid_proxy_overlap_fraction"] >= 0.75
            else 0
        )
    selected = []
    if proxy_counts:
        selected.append(
            (max(proxy_counts, key=proxy_counts.get), "Highest proxy-overlap FP")
        )
    fp_counts = {
        names[index]: row["false_positive_pixels"]
        for index, row in enumerate(baseline)
    }
    selected.append((max(fp_counts, key=fp_counts.get), "Most baseline FP pixels"))
    target_counts = {
        names[index]: row["true_positive_pixels"] + row["false_negative_pixels"]
        for index, row in enumerate(baseline)
    }
    nonempty = [name for name, count in target_counts.items() if count > 0]
    if nonempty:
        selected.append(
            (
                min(nonempty, key=lambda name: target_counts[name]),
                "Smallest annotated defect",
            )
        )
    unique = []
    for name, label in selected:
        if name not in [item[0] for item in unique]:
            unique.append((name, label))
    return unique


def _save_qualitative(
    output_path,
    images,
    targets,
    probabilities,
    names,
    baseline_masks,
    selected_masks,
    selected_name,
    baseline_components,
):
    per_image = {
        "Phase 5 baseline": _per_image_scores(baseline_masks, targets)
    }
    examples = _visual_examples(per_image, names, baseline_components)
    figure, axes = plt.subplots(
        len(examples),
        7,
        figsize=(23, 3.7 * len(examples)),
        squeeze=False,
    )
    headings = (
        "EL input",
        "Ground truth",
        "Baseline mask",
        "Selected post-process",
        "Baseline errors",
        "Selected errors",
        "Grid-line proxy",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row_index, (image_name, reason) in enumerate(examples):
        index = names.index(image_name)
        image = np.clip(
            images[index].numpy().transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN,
            0,
            1,
        )
        target = targets[index, 0].numpy() > 0.5
        probability = probabilities[index]
        baseline = baseline_masks[index]
        selected = selected_masks[index]
        gray = _gray_image(images[index].numpy())
        proxy = _dark_linear_structure(gray)
        panels = (
            image,
            target,
            baseline,
            selected,
            _error_overlay(image, baseline, target),
            _error_overlay(image, selected, target),
            proxy,
        )
        cmaps = (None, "gray", "gray", "gray", None, None, "gray")
        for column, (panel, cmap) in enumerate(zip(panels, cmaps)):
            axes[row_index, column].imshow(panel, cmap=cmap, vmin=0, vmax=1)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(f"{image_name}: {reason}", fontsize=8)
    figure.suptitle(
        f"Phase 5 baseline vs {selected_name} | "
        "Green: TP, red: FP, blue: FN",
        y=1.005,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_analysis(project_dir, checkpoint_path, batch_size=8, num_threads=6):
    if batch_size <= 0 or num_threads <= 0:
        raise ValueError("batch_size and num_threads must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    torch.set_num_threads(num_threads)
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    state, metadata = _load_state(checkpoint_path)
    if (
        metadata.get("negative_bce_weight") != 1.1
        or float(metadata.get("boundary_weight") or 0.0) != 0.0
        or (metadata.get("sampling") or {}).get("strategy", "shuffle") != "shuffle"
    ):
        raise ValueError("Checkpoint metadata does not match the Phase 5 model")
    names = _validation_names(validation_loader)
    cache_path = project_dir / "results" / "phase10_phase5_validation_probabilities.npz"
    checkpoint_stat = checkpoint_path.stat()
    cached = None
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            required_cache_keys = {
                "probabilities",
                "targets",
                "image_names",
                "inference_seconds_per_image",
                "checkpoint_size_bytes",
                "checkpoint_mtime_ns",
            }
            if required_cache_keys.issubset(cache.files):
                matches = (
                    np.array_equal(cache["image_names"], np.asarray(names))
                    and int(cache["checkpoint_size_bytes"])
                    == checkpoint_stat.st_size
                    and int(cache["checkpoint_mtime_ns"])
                    == checkpoint_stat.st_mtime_ns
                )
                if matches:
                    cached = {
                        key: cache[key].copy()
                        for key in required_cache_keys
                    }
    if cached is not None:
        images, targets = _validation_tensors(validation_loader)
        if not np.array_equal(
            cached["targets"],
            targets[:, 0].numpy().astype(np.uint8),
        ):
            raise RuntimeError("Cached validation masks do not match this split")
        probabilities = cached["probabilities"]
        predictions = torch.from_numpy(probabilities[:, None])
        inference_seconds = float(cached["inference_seconds_per_image"])
    else:
        model = ResNet18UNet(pretrained=False).cpu()
        model.load_state_dict(state)
        model.eval()
        predictions, targets, images, inference_seconds = _infer_validation(
            model,
            validation_loader,
        )
        probabilities = predictions[:, 0].numpy()
        np.savez_compressed(
            cache_path,
            probabilities=probabilities.astype(np.float32),
            targets=targets[:, 0].numpy().astype(np.uint8),
            image_names=np.asarray(names),
            threshold=np.float32(THRESHOLD),
            inference_seconds_per_image=np.float64(inference_seconds),
            checkpoint_size_bytes=np.int64(checkpoint_stat.st_size),
            checkpoint_mtime_ns=np.int64(checkpoint_stat.st_mtime_ns),
        )
    if len(names) != len(predictions):
        raise RuntimeError("Validation image order does not match predictions")
    if predictions.shape != targets.shape:
        raise RuntimeError("Validation predictions are not aligned to targets")
    probability_maps = probabilities
    normalized_images = images.numpy()
    grays = [_gray_image(image) for image in normalized_images]
    groups = _size_groups(targets)

    output_dir = project_dir / "results"
    component_rows, feature_rows = _component_report(
        predictions,
        targets,
        images,
        names,
        THRESHOLD,
    )
    _write_csv(
        output_dir / "phase10_phase5_prediction_components.csv",
        component_rows,
    )
    _write_csv(
        output_dir / "phase10_phase5_component_feature_summary.csv",
        _component_summary(component_rows, feature_rows),
    )
    _write_csv(
        output_dir / "phase10_phase5_component_feature_by_image.csv",
        feature_rows,
    )

    baseline_masks = _mask_set(
        probability_maps,
        grays,
        THRESHOLD,
        "none",
    )
    baseline_scores = _per_image_scores(baseline_masks, targets)
    baseline_metrics = _metric_values(baseline_masks, targets)
    baseline_components = component_rows

    candidate_rows = []
    per_image_rows = []
    threshold_rows = []
    candidate_masks = {}
    for strategy, display_name in _candidate_strategies():
        if strategy == "none":
            threshold, scores = _threshold_candidates(
                probability_maps,
                targets,
                grays,
                strategy,
                groups,
            )
            threshold_rows.extend(
                {
                    "candidate": display_name,
                    "strategy": strategy,
                    "threshold": value,
                    **metrics,
                    "selected": int(value == threshold),
                }
                for value, metrics in sorted(scores.items())
            )
            if display_name == "Threshold only":
                fixed_masks = baseline_masks
                fixed_row, fixed_per_image = _candidate_row(
                    "Phase 5 baseline",
                    strategy,
                    THRESHOLD,
                    fixed_masks,
                    targets,
                    groups,
                    baseline_scores,
                    grays,
                    inference_seconds,
                )
                fixed_row["postprocess_seconds_per_image"] = 0.0
                fixed_row["total_seconds_per_image"] = inference_seconds
                fixed_row.update(
                    {
                        "paired_mean_dice_delta": 0.0,
                        "dice_delta_ci95_low": 0.0,
                        "dice_delta_ci95_high": 0.0,
                        "paired_images_improved": 0,
                        "paired_images_unchanged": len(names),
                        "paired_images_regressed": 0,
                    }
                )
                candidate_rows.append(fixed_row)
                per_image_rows.extend(
                    {
                        "candidate": "Phase 5 baseline",
                        "strategy": strategy,
                        "threshold": THRESHOLD,
                        "image": image_name,
                        **row,
                        "paired_dice_delta": 0.0,
                    }
                    for image_name, row in zip(names, fixed_per_image)
                )
            candidate_masks[display_name] = _mask_set(
                probability_maps,
                grays,
                threshold,
                strategy,
            )
        else:
            threshold, scores = _threshold_candidates(
                probability_maps,
                targets,
                grays,
                strategy,
                groups,
            )
            threshold_rows.extend(
                {
                    "candidate": display_name,
                    "strategy": strategy,
                    "threshold": value,
                    **metrics,
                    "selected": int(value == threshold),
                }
                for value, metrics in sorted(scores.items())
            )
        selected_masks = _mask_set(
            probability_maps,
            grays,
            threshold,
            strategy,
        )
        candidate_masks[display_name] = selected_masks
        row, image_rows = _candidate_row(
            display_name,
            strategy,
            threshold,
            selected_masks,
            targets,
            groups,
            baseline_scores,
            grays,
            inference_seconds,
        )
        if display_name != "Threshold only":
            postprocess_seconds = _postprocess_timing(
                probability_maps,
                grays,
                threshold,
                strategy,
            )
            row["postprocess_seconds_per_image"] = postprocess_seconds
            row["total_seconds_per_image"] = inference_seconds + postprocess_seconds
            candidate_rows.append(row)
            per_image_rows.extend(
                {
                    "candidate": display_name,
                    "strategy": strategy,
                    "threshold": threshold,
                    "image": image_name,
                    **image_row,
                    "paired_dice_delta": (
                        image_row["dice"] - baseline_scores[index]["dice"]
                    ),
                }
                for index, (image_name, image_row) in enumerate(
                    zip(names, image_rows)
                )
            )

    threshold_only_name = "Threshold only"
    threshold_only_threshold = max(
        (
            row
            for row in threshold_rows
            if row["candidate"] == threshold_only_name
        ),
        key=lambda row: row["dice"],
    )["threshold"]
    threshold_only_masks = _mask_set(
        probability_maps,
        grays,
        threshold_only_threshold,
        "none",
    )
    threshold_only_row, threshold_only_per_image = _candidate_row(
        threshold_only_name,
        "none",
        threshold_only_threshold,
        threshold_only_masks,
        targets,
        groups,
        baseline_scores,
        grays,
        inference_seconds,
    )
    threshold_only_row["postprocess_seconds_per_image"] = 0.0
    threshold_only_row["total_seconds_per_image"] = inference_seconds
    candidate_rows.append(threshold_only_row)
    per_image_rows.extend(
        {
            "candidate": threshold_only_name,
            "strategy": "none",
            "threshold": threshold_only_threshold,
            "image": image_name,
            **row,
            "paired_dice_delta": row["dice"] - baseline_scores[index]["dice"],
        }
        for index, (image_name, row) in enumerate(
            zip(names, threshold_only_per_image)
        )
    )
    candidate_masks[threshold_only_name] = threshold_only_masks

    _write_csv(
        output_dir / "phase10_postprocessing_comparison.csv",
        candidate_rows,
    )
    _write_csv(
        output_dir / "phase10_postprocessing_thresholds.csv",
        threshold_rows,
    )
    _write_csv(
        output_dir / "phase10_postprocessing_per_image.csv",
        per_image_rows,
    )

    baseline_row = next(
        row for row in candidate_rows if row["candidate"] == "Phase 5 baseline"
    )
    eligible = [
        row
        for row in candidate_rows
        if row["candidate"] != "Phase 5 baseline"
        and row["dice"] >= baseline_row["dice"]
        and row["tiny_defect_dice"] >= baseline_row["tiny_defect_dice"] - 0.01
        and row["tiny_defect_recall"]
        >= baseline_row["tiny_defect_recall"] - 0.015
        and row["boundary_f1_tolerance_1px"]
        >= baseline_row["boundary_f1_tolerance_1px"] - 0.005
        and row["false_positive_pixels"] < baseline_row["false_positive_pixels"]
        and row["false_positive_pixels_on_dark_linear_proxy"]
        < baseline_row["false_positive_pixels_on_dark_linear_proxy"]
    ]
    robust_candidates = [
        row for row in eligible if row["dice_delta_ci95_low"] > 0
    ]
    best_candidate_row = (
        max(eligible, key=lambda row: row["dice"])
        if eligible
        else baseline_row
    )
    selected_row = (
        max(robust_candidates, key=lambda row: row["dice"])
        if robust_candidates
        else baseline_row
    )
    selected_name = selected_row["candidate"]
    visualization_name = best_candidate_row["candidate"]
    _save_qualitative(
        output_dir / "phase10_postprocessing_qualitative.png",
        images,
        targets,
        probability_maps,
        names,
        baseline_masks,
        candidate_masks[visualization_name],
        visualization_name,
        baseline_components,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "validation_images": len(names),
        "test_split_evaluated": False,
        "baseline_threshold": THRESHOLD,
        "baseline_metrics": baseline_metrics,
        "dark_linear_proxy": _grid_proxy_summary(
            baseline_masks,
            targets,
            grays,
        ),
        "candidate_selection_guard": (
            "No Dice regression; tiny Dice decline <=0.01; boundary F1 decline "
            "<=0.005; tiny recall decline <=0.015; both total FP and "
            "dark-linear-proxy FP must decrease; "
            "adopt only if the paired 95% Dice bootstrap interval is entirely "
            "above zero."
        ),
        "eligible_candidates": [row["candidate"] for row in eligible],
        "candidates_with_positive_paired_dice_ci": [
            row["candidate"] for row in robust_candidates
        ],
        "recommended_candidate": selected_name,
        "best_guarded_experimental_candidate": visualization_name,
        "component_count": len(component_rows),
        "false_positive_only_components": sum(
            row["false_positive_only_component"] for row in component_rows
        ),
        "false_positive_pixels_in_components": sum(
            row["false_positive_pixels"] for row in component_rows
        ),
        "inference_seconds_per_image": inference_seconds,
        "probability_cache": str(
            output_dir / "phase10_phase5_validation_probabilities.npz"
        ),
    }
    with (output_dir / "phase10_postprocessing_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(summary, output_file, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validation-only Phase 5 connected-component and grid analysis."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    run_analysis(
        project_dir,
        args.checkpoint,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
