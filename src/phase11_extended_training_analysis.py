import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
import numpy as np
import torch
from torch.utils.data import Subset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.dataset import get_dataloaders
from src.models import ResNet18UNet
from src.phase7_analysis import _boundary_f1
from src.phase8_tta import _select_threshold
from src.train import compute_detailed_metrics


def _validation_names(loader):
    subset = loader.dataset
    if not isinstance(subset, Subset):
        raise TypeError("Expected validation loader to use a Subset")
    return [subset.dataset.samples[index][0].name for index in subset.indices]


def _validation_data(loader):
    images = []
    targets = []
    for batch_images, batch_masks, _ in loader:
        images.append(batch_images)
        targets.append(batch_masks)
    if not images:
        raise ValueError("The validation data loader is empty")
    return torch.cat(images), torch.cat(targets)


def _infer(model, loader):
    model.eval()
    warmup_images = next(iter(loader))[0]
    with torch.inference_mode():
        model(warmup_images)
        predictions = []
        elapsed = 0.0
        image_count = 0
        for images, masks, _ in loader:
            started = time.perf_counter()
            predictions.append(model(images).cpu())
            elapsed += time.perf_counter() - started
            image_count += len(images)
    if image_count == 0:
        raise ValueError("The validation data loader is empty")
    return torch.cat(predictions), elapsed / image_count


def _size_groups(targets):
    masks = targets[:, 0].numpy() > 0.5
    fractions = masks.reshape(len(masks), -1).mean(axis=1)
    return np.where(
        fractions == 0,
        "empty",
        np.where(fractions <= 0.02, "tiny", np.where(fractions <= 0.05, "medium", "large")),
    )


def _per_image_rows(name, predictions, targets, threshold, image_names):
    groups = _size_groups(targets)
    rows = []
    for index, image_name in enumerate(image_names):
        target = targets[index, 0].numpy() > 0.5
        predicted = predictions[index, 0].numpy() > threshold
        tp = int((predicted & target).sum())
        fp = int((predicted & ~target).sum())
        fn = int((~predicted & target).sum())
        rows.append(
            {
                "model": name,
                "image": image_name,
                "size_group": groups[index],
                "ground_truth_pixels": int(target.sum()),
                "true_positive_pixels": tp,
                "false_positive_pixels": fp,
                "false_negative_pixels": fn,
                "dice": 2 * tp / max(1, 2 * tp + fp + fn),
                "iou": tp / max(1, tp + fp + fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
            }
        )
    return rows


def _summary_row(
    name,
    predictions,
    targets,
    threshold,
    inference_seconds,
    training_seconds,
    continued_training_seconds,
    per_image,
):
    metrics = compute_detailed_metrics(predictions, targets, threshold)
    predicted = predictions > threshold
    truth = targets > 0.5
    groups = _size_groups(targets)
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
    return {
        "model": name,
        "threshold": threshold,
        **metrics,
        "tiny_defect_images": int(len(tiny_indices)),
        "tiny_defect_dice": tiny_dice,
        "tiny_defect_recall": tiny_recall,
        "boundary_f1_tolerance_1px": _boundary_f1(
            predictions,
            targets,
            threshold,
            tolerance=1,
        ),
        "true_positive_pixels": int((predicted & truth).sum()),
        "false_positive_pixels": int((predicted & ~truth).sum()),
        "false_negative_pixels": int((~predicted & truth).sum()),
        "inference_seconds_per_image": inference_seconds,
        "training_seconds_including_phase5": training_seconds,
        "continued_training_seconds": continued_training_seconds,
    }


def _paired_bootstrap(candidate_rows, baseline_rows, seed=42):
    candidate = np.asarray([row["dice"] for row in candidate_rows])
    baseline = np.asarray([row["dice"] for row in baseline_rows])
    differences = candidate - baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(10000, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "mean_dice_delta": float(differences.mean()),
        "dice_delta_ci95_low": float(np.quantile(means, 0.025)),
        "dice_delta_ci95_high": float(np.quantile(means, 0.975)),
        "images_improved": int((differences > 0).sum()),
        "images_unchanged": int((differences == 0).sum()),
        "images_regressed": int((differences < 0).sum()),
    }


def _threshold_rows(name, scores, selected):
    rows = []
    for threshold, metrics in sorted(scores.items()):
        rows.append(
            {
                "model": name,
                "threshold": threshold,
                **metrics,
                "selected": int(threshold == selected),
            }
        )
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _training_rows(history):
    rows = []
    for record in history:
        validation = record.get("validation", {})
        rates = record.get("learning_rates_used", [])
        rows.append(
            {
                "epoch": record["epoch"],
                "phase": record["phase"],
                "train_loss": record["train_loss"],
                "validation_loss": record.get("validation_loss", ""),
                "validation_dice": validation.get("dice", ""),
                "validation_iou": validation.get("iou", ""),
                "validation_precision": validation.get("precision", ""),
                "validation_recall": validation.get("recall", ""),
                "tiny_defect_dice": validation.get("tiny_defect_dice", ""),
                "false_positive_pixels": validation.get(
                    "false_positive_pixels",
                    "",
                ),
                "threshold": record.get("threshold", ""),
                "learning_rate_used": rates[0] if rates else "",
                "epoch_seconds": record.get("epoch_seconds", ""),
            }
        )
    return rows


def _save_training_history(history, output_dir):
    rows = _training_rows(history)
    _write_csv(output_dir / "phase11_extended_training_history.csv", rows)
    epochs = [row["epoch"] for row in rows]
    train_losses = np.asarray([row["train_loss"] for row in rows], dtype=float)
    validation_losses = np.asarray(
        [
            float(row["validation_loss"]) if row["validation_loss"] != "" else np.nan
            for row in rows
        ]
    )
    dice = np.asarray(
        [
            float(row["validation_dice"]) if row["validation_dice"] != "" else np.nan
            for row in rows
        ]
    )
    tiny_dice = np.asarray(
        [
            float(row["tiny_defect_dice"])
            if row["tiny_defect_dice"] != ""
            else np.nan
            for row in rows
        ]
    )
    fp_pixels = np.asarray(
        [
            float(row["false_positive_pixels"])
            if row["false_positive_pixels"] != ""
            else np.nan
            for row in rows
        ]
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, train_losses, label="train")
    axes[0, 0].plot(epochs, validation_losses, label="validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, dice, label="overall Dice")
    axes[0, 1].plot(epochs, tiny_dice, label="tiny-defect Dice")
    axes[0, 1].set_title("Validation Dice")
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, fp_pixels)
    axes[1, 0].set_title("Validation false-positive pixels")
    axes[1, 1].plot(
        epochs,
        [
            float(row["learning_rate_used"])
            if row["learning_rate_used"] != ""
            else np.nan
            for row in rows
        ],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Decoder learning rate")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    figure.tight_layout()
    figure.savefig(output_dir / "phase11_extended_training_curves.png", dpi=160)
    plt.close(figure)
    return rows


def _error_overlay(image, prediction, target):
    overlay = image.copy()
    colors = np.zeros_like(overlay)
    colors[prediction & target] = (0.0, 1.0, 0.0)
    colors[prediction & ~target] = (1.0, 0.0, 0.0)
    colors[~prediction & target] = (0.0, 0.35, 1.0)
    selected = prediction | target
    overlay[selected] = 0.45 * overlay[selected] + 0.55 * colors[selected]
    return overlay


def _visual_indices(names, baseline_rows, extended_rows):
    candidates = []
    defect_indices = [
        index
        for index, row in enumerate(baseline_rows)
        if row["ground_truth_pixels"] > 0
    ]
    tiny_indices = [
        index
        for index, row in enumerate(baseline_rows)
        if row["size_group"] == "tiny"
    ]
    if tiny_indices:
        candidates.append(
            (
                min(tiny_indices, key=lambda index: baseline_rows[index]["ground_truth_pixels"]),
                "Smallest tiny defect",
            )
        )
    if defect_indices:
        candidates.append(
            (
                max(
                    defect_indices,
                    key=lambda index: (
                        extended_rows[index]["dice"] - baseline_rows[index]["dice"]
                    ),
                ),
                "Largest Dice gain",
            )
        )
        candidates.append(
            (
                min(
                    defect_indices,
                    key=lambda index: (
                        extended_rows[index]["dice"] - baseline_rows[index]["dice"]
                    ),
                ),
                "Largest Dice regression",
            )
        )
    candidates.append(
        (
            max(
                range(len(names)),
                key=lambda index: extended_rows[index]["false_positive_pixels"],
            ),
            "Most extended-model FP pixels",
        )
    )
    selected = []
    for index, label in candidates:
        if index not in [item[0] for item in selected]:
            selected.append((index, label))
    return selected


def _save_qualitative(
    path,
    images,
    targets,
    baseline,
    extended,
    thresholds,
    names,
    baseline_rows,
    extended_rows,
):
    examples = _visual_indices(names, baseline_rows, extended_rows)
    figure, axes = plt.subplots(
        len(examples),
        6,
        figsize=(21, 3.6 * len(examples)),
        squeeze=False,
    )
    headings = (
        "EL image",
        "Ground truth",
        "Phase 5 mask",
        "Extended best mask",
        "Phase 5 errors",
        "Extended best errors",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row_index, (index, label) in enumerate(examples):
        image = np.clip(
            images[index].numpy().transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN,
            0,
            1,
        )
        target = targets[index, 0].numpy() > 0.5
        baseline_mask = baseline[index, 0].numpy() > thresholds["Phase 5"]
        extended_mask = extended[index, 0].numpy() > thresholds["Extended best"]
        panels = (
            image,
            target,
            baseline_mask,
            extended_mask,
            _error_overlay(image, baseline_mask, target),
            _error_overlay(image, extended_mask, target),
        )
        cmaps = (None, "gray", "gray", "gray", None, None)
        for column, (panel, cmap) in enumerate(zip(panels, cmaps)):
            axes[row_index, column].imshow(panel, cmap=cmap, vmin=0, vmax=1)
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(label, fontsize=8)
    figure.suptitle("Green: TP | Red: FP | Blue: FN")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_analysis(project_dir, baseline_path, extended_path, batch_size=8, num_threads=6):
    torch.set_num_threads(num_threads)
    _, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=False,
    )
    names = _validation_names(validation_loader)
    images, targets = _validation_data(validation_loader)
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Phase 5 checkpoint not found: {baseline_path}")
    baseline_state, baseline_metadata = _load_state(baseline_path)
    baseline_stat = baseline_path.stat()
    cache_path = project_dir / "results" / "phase10_phase5_validation_probabilities.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        if not np.array_equal(cache["image_names"], np.asarray(names)):
            raise RuntimeError("Cached Phase 5 predictions use a different split")
        if (
            int(cache["checkpoint_size_bytes"]) != baseline_stat.st_size
            or int(cache["checkpoint_mtime_ns"]) != baseline_stat.st_mtime_ns
        ):
            raise RuntimeError("Cached predictions were produced by another checkpoint")
        if not np.array_equal(
            cache["targets"],
            targets[:, 0].numpy().astype(np.uint8),
        ):
            raise RuntimeError("Cached Phase 5 masks do not match validation")
        baseline_predictions = torch.from_numpy(
            cache["probabilities"][:, None].copy()
        )
        baseline_inference_seconds = float(cache["inference_seconds_per_image"])

    state, extended_metadata = _load_state(extended_path)
    model = ResNet18UNet(pretrained=False).cpu()
    model.load_state_dict(state)
    extended_predictions, extended_inference_seconds = _infer(
        model,
        validation_loader,
    )
    if extended_predictions.shape != targets.shape:
        raise RuntimeError("Extended-model predictions do not match validation masks")
    if len(baseline_state) != len(state):
        raise RuntimeError("Baseline and extended model architectures differ")

    threshold_results = {}
    predictions = {
        "Phase 5": baseline_predictions,
        "Extended best": extended_predictions,
    }
    training_seconds = {
        "Phase 5": float(baseline_metadata.get("training_seconds", "")),
        "Extended best": float(extended_metadata.get("training_seconds", "")),
    }
    continued_seconds = {
        "Phase 5": 0.0,
        "Extended best": float(
            extended_metadata.get("continued_training_seconds", 0.0)
        ),
    }
    inference_seconds = {
        "Phase 5": baseline_inference_seconds,
        "Extended best": extended_inference_seconds,
    }
    summary_rows = []
    per_model_rows = {}
    threshold_rows = []
    selected_thresholds = {}
    for name, probabilities in predictions.items():
        threshold, scores = _select_threshold(probabilities, targets)
        selected_thresholds[name] = threshold
        threshold_results[name] = scores
        image_rows = _per_image_rows(
            name,
            probabilities,
            targets,
            threshold,
            names,
        )
        per_model_rows[name] = image_rows
        summary_rows.append(
            _summary_row(
                name,
                probabilities,
                targets,
                threshold,
                inference_seconds[name],
                training_seconds[name],
                continued_seconds[name],
                image_rows,
            )
        )
        threshold_rows.extend(_threshold_rows(name, scores, threshold))

    baseline_rows = per_model_rows["Phase 5"]
    extended_rows = per_model_rows["Extended best"]
    paired = _paired_bootstrap(extended_rows, baseline_rows)
    tiny_indices = [
        index for index, row in enumerate(baseline_rows) if row["size_group"] == "tiny"
    ]
    tiny_paired = _paired_bootstrap(
        [extended_rows[index] for index in tiny_indices],
        [baseline_rows[index] for index in tiny_indices],
        seed=43,
    )
    comparison = {
        "baseline": summary_rows[0],
        "extended_best": summary_rows[1],
        "paired_dice": paired,
        "tiny_paired_dice": tiny_paired,
        "best_epoch": max(
            extended_metadata["history"],
            key=lambda record: record["validation"]["dice"],
        )["epoch"],
        "last_epoch": int(
            torch.load(
                extended_path.with_name(
                    extended_path.name.replace("_best.pth", "_last.pth")
                ),
                map_location="cpu",
                weights_only=False,
            )["phase_epoch"]
        ),
        "test_split_evaluated": False,
    }
    output_dir = project_dir / "results"
    _write_csv(
        output_dir / "phase11_extended_training_comparison.csv",
        summary_rows,
    )
    per_image_rows = [
        {
            **baseline_rows[index],
            "extended_dice": extended_rows[index]["dice"],
            "extended_iou": extended_rows[index]["iou"],
            "extended_precision": extended_rows[index]["precision"],
            "extended_recall": extended_rows[index]["recall"],
            "extended_true_positive_pixels": extended_rows[index]["true_positive_pixels"],
            "extended_false_positive_pixels": extended_rows[index]["false_positive_pixels"],
            "extended_false_negative_pixels": extended_rows[index]["false_negative_pixels"],
            "paired_dice_delta": (
                extended_rows[index]["dice"] - baseline_rows[index]["dice"]
            ),
        }
        for index in range(len(names))
    ]
    _write_csv(
        output_dir / "phase11_extended_training_per_image.csv",
        per_image_rows,
    )
    _write_csv(
        output_dir / "phase11_extended_training_thresholds.csv",
        threshold_rows,
    )
    _save_training_history(extended_metadata["history"], output_dir)
    _save_qualitative(
        output_dir / "phase11_extended_training_qualitative.png",
        images,
        targets,
        baseline_predictions,
        extended_predictions,
        selected_thresholds,
        names,
        baseline_rows,
        extended_rows,
    )
    with (output_dir / "phase11_extended_training_summary.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(comparison, output_file, indent=2)
    print(json.dumps(comparison, indent=2))


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validation-only comparison of extended Phase 5 training."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_negw_best.pth",
    )
    parser.add_argument(
        "--extended",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_phase5_to50_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    args = parser.parse_args()
    run_analysis(
        project_dir,
        args.baseline,
        args.extended,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
