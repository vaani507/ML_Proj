import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _read_phase5_comparison(path):
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    return rows


def _read_per_image_results(path):
    with path.open("r", newline="", encoding="utf-8-sig") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        for field in (
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
        ):
            row[field] = float(row[field])
    return rows


def _save_model_architecture(path):
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.set_xlim(0, 13)
    axis.set_ylim(0, 5)
    axis.axis("off")

    boxes = [
        (0.25, 1.55, 1.9, 1.9, "EL image\n224 x 224 x 3"),
        (2.65, 1.55, 2.0, 1.9, "ImageNet\npre-trained\nResNet-18"),
        (5.15, 1.55, 2.0, 1.9, "Skip feature\nencoder levels"),
        (7.65, 1.55, 2.0, 1.9, "U-Net\ndecoder"),
        (10.15, 1.55, 2.3, 1.9, "Sigmoid\nprobability mask"),
    ]
    for x, y, width, height, label in boxes:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.8,
            facecolor="#eef4ff",
            edgecolor="#3567a8",
        )
        axis.add_patch(box)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    for start_x in (2.15, 4.65, 7.15, 9.65):
        axis.add_patch(
            FancyArrowPatch(
                (start_x, 2.5),
                (start_x + 0.42, 2.5),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.8,
                color="#23395d",
            )
        )
    axis.text(
        6.5,
        4.35,
        "Selected Phase 5 ResNet-18 U-Net",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )
    axis.text(
        6.5,
        0.55,
        "224 x 224 RGB input  |  sigmoid output  |  fixed inference threshold 0.165",
        ha="center",
        fontsize=11,
        color="#444",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_model_comparison(path, comparison_rows):
    candidates = []
    for row in comparison_rows:
        if "Custom U-Net" in row["model"]:
            label = "Custom U-Net"
        elif "Phase 5" in row["model"]:
            label = "Phase 5 ResNet-18"
        elif "20-epoch" in row["model"]:
            label = "ResNet-18 20 epochs"
        elif "5 epochs" in row["model"]:
            label = "ResNet-18 5 epochs"
        else:
            continue
        candidates.append(
            (
                label,
                float(row["test_dice"]),
                float(row["test_iou"]),
                float(row["test_precision"]),
                float(row["test_recall"]),
            )
        )
    if not candidates:
        raise ValueError("No model comparison rows found")

    labels = [row[0] for row in candidates]
    metric_names = ("Dice", "IoU", "Precision", "Recall")
    data = [
        [row[index] for row in candidates]
        for index in range(1, 5)
    ]
    x = range(len(labels))
    width = 0.19
    figure, axis = plt.subplots(figsize=(12, 7))
    for offset, (metric_name, values) in enumerate(zip(metric_names, data)):
        positions = [position + (offset - 1.5) * width for position in x]
        bars = axis.bar(
            positions,
            values,
            width=width,
            label=metric_name,
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    axis.set_xticks(list(x), labels, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test metric")
    axis.set_title("Solar-cell segmentation model comparison")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend(ncols=4, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_test_metrics(path, metrics):
    metric_names = ("dice", "iou", "precision", "recall")
    labels = ("Dice", "IoU", "Precision", "Recall")
    values = [float(metrics[name]) for name in metric_names]
    colors = ["#3268a8", "#5b9bd5", "#70ad47", "#ed7d31"]
    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.bar(labels, values, color=colors)
    axis.bar_label(bars, fmt="%.4f", fontsize=11, padding=5)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Phase 5 held-out test-set metrics")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_confusion_matrix(path, metrics):
    true_positive = int(metrics["true_positive_pixels"])
    false_positive = int(metrics["false_positive_pixels"])
    false_negative = int(metrics["false_negative_pixels"])
    true_negative = int(metrics["true_negative_pixels"])
    counts = [
        [true_negative, false_positive],
        [false_negative, true_positive],
    ]
    normalized = [
        [
            true_negative / max(1, true_negative + false_positive),
            false_positive / max(1, true_negative + false_positive),
        ],
        [
            false_negative / max(1, false_negative + true_positive),
            true_positive / max(1, false_negative + true_positive),
        ],
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, matrix, title, fmt in (
        (
            axes[0],
            counts,
            "Pixel counts",
            lambda value: f"{value:,}",
        ),
        (
            axes[1],
            normalized,
            "Row-normalized proportions",
            lambda value: f"{value:.3f}",
        ),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(map(max, matrix)))
        axis.set_xticks([0, 1], ["Predicted background", "Predicted defect"])
        axis.set_yticks([0, 1], ["Ground-truth background", "Ground-truth defect"])
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        labels = (("TN", "FP"), ("FN", "TP"))
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    f"{labels[row_index][column_index]}\n{fmt(matrix[row_index][column_index])}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white" if matrix[row_index][column_index] > max(map(max, matrix)) * 0.45 else "black",
                )
        axis.set_xticks([0, 1], ["Background", "Defect"])
        axis.set_yticks([0, 1], ["Background", "Defect"])
    figure.suptitle("Phase 5 held-out test pixel confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_per_image_distribution(path, rows):
    dice = [row["dice"] for row in rows]
    median = statistics.median(dice)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.hist(
        dice,
        bins=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        color="#4c83b6",
        edgecolor="white",
        linewidth=1.2,
    )
    axis.axvline(
        sum(dice) / len(dice),
        color="#e67e22",
        linestyle="--",
        linewidth=2,
        label=f"Mean {sum(dice) / len(dice):.3f}",
    )
    axis.axvline(
        median,
        color="#2c6e49",
        linestyle=":",
        linewidth=2,
        label=f"Median {median:.3f}",
    )
    axis.set_xlabel("Per-image Dice")
    axis.set_ylabel("Number of test images")
    axis.set_title("How segmentation Dice varies across 56 test images")
    axis.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_performance_vs_defect_size(path, rows):
    colors = {
        "Tiny (<=2%)": "#d95f02",
        "Medium (2-5%)": "#1b9e77",
        "Large (>5%)": "#386cb0",
    }
    figure, axis = plt.subplots(figsize=(10, 6))
    for label, lower, upper in (
        ("Tiny (<=2%)", 0.0, 0.02),
        ("Medium (2-5%)", 0.02, 0.05),
        ("Large (>5%)", 0.05, 1.0),
    ):
        selected = [
            row
            for row in rows
            if lower
            < row["ground_truth_pixels"] / (224 * 224)
            <= upper
        ]
        if label == "Tiny (<=2%)":
            selected = [
                row
                for row in rows
                if 0 < row["ground_truth_pixels"] / (224 * 224) <= 0.02
            ]
        axis.scatter(
            [
                100 * row["ground_truth_pixels"] / (224 * 224)
                for row in selected
            ],
            [row["dice"] for row in selected],
            label=f"{label}, n={len(selected)}",
            color=colors[label],
            alpha=0.8,
            edgecolors="white",
            linewidths=0.5,
            s=55,
        )
    difficult = sorted(rows, key=lambda row: row["dice"])[:4]
    for row in difficult:
        axis.annotate(
            row["image"],
            (
                100 * row["ground_truth_pixels"] / (224 * 224),
                row["dice"],
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Ground-truth defect pixels (% of image)")
    axis.set_ylabel("Per-image Dice")
    axis.set_ylim(0, 1.03)
    axis.set_title("Per-image Dice versus annotated defect size")
    axis.grid(linestyle="--", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_largest_error_images(path, rows, limit=12):
    selected = sorted(
        rows,
        key=lambda row: row["false_positive_pixels"] + row["false_negative_pixels"],
        reverse=True,
    )[:limit]
    labels = [row["image"] for row in selected]
    fp = [row["false_positive_pixels"] for row in selected]
    fn = [row["false_negative_pixels"] for row in selected]
    positions = list(range(len(selected)))
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        positions,
        fp,
        color="#d95f5f",
        label="False-positive pixels",
    )
    axis.barh(
        positions,
        fn,
        left=fp,
        color="#4c83b6",
        label="False-negative pixels",
    )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Error pixels (FP + FN)")
    axis.set_title(f"Images contributing most pixel errors (top {len(selected)})")
    axis.grid(axis="x", linestyle="--", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(project_dir):
    project_dir = Path(project_dir)
    results_dir = project_dir / "results"
    final_result_path = results_dir / "phase12_final_test_metrics.json"
    comparison_path = results_dir / "phase5_model_comparison.csv"
    per_image_path = results_dir / "phase12_final_test_per_image.csv"
    if not final_result_path.is_file():
        raise FileNotFoundError(final_result_path)
    if not comparison_path.is_file():
        raise FileNotFoundError(comparison_path)
    if not per_image_path.is_file():
        raise FileNotFoundError(per_image_path)
    final_result = json.loads(final_result_path.read_text(encoding="utf-8"))
    metrics = final_result["metrics"]
    per_image_rows = _read_per_image_results(per_image_path)

    _save_model_architecture(results_dir / "phase12_model_architecture.png")
    _save_test_metrics(results_dir / "phase12_test_metrics.png", metrics)
    _save_confusion_matrix(results_dir / "phase12_test_confusion_matrix.png", metrics)
    _save_model_comparison(
        results_dir / "phase12_model_comparison_test.png",
        _read_phase5_comparison(comparison_path),
    )
    _save_per_image_distribution(
        results_dir / "phase12_test_per_image_dice_distribution.png",
        per_image_rows,
    )
    _save_performance_vs_defect_size(
        results_dir / "phase12_test_dice_vs_defect_size.png",
        per_image_rows,
    )
    _save_largest_error_images(
        results_dir / "phase12_test_largest_error_images.png",
        per_image_rows,
    )
    print(
        json.dumps(
            {
                "model_architecture": "results/phase12_model_architecture.png",
                "test_metrics": "results/phase12_test_metrics.png",
                "test_confusion_matrix": "results/phase12_test_confusion_matrix.png",
                "model_comparison": "results/phase12_model_comparison_test.png",
                "per_image_dice_distribution": "results/phase12_test_per_image_dice_distribution.png",
                "dice_vs_defect_size": "results/phase12_test_dice_vs_defect_size.png",
                "largest_error_images": "results/phase12_test_largest_error_images.png",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate presentation graphs.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()
    run(arguments.project_dir)
