import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import get_dataloaders
from src.models import ResNet18UNet, UNet
from src.train import compute_detailed_metrics


THRESHOLD_CANDIDATES = (0.3, 0.4, 0.5, 0.6, 0.7)
IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def _load_state(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint
    return checkpoint, {}


def _predict(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    images_for_visualization = []
    forward_seconds = 0.0
    sample_count = 0
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device)
            started = time.perf_counter()
            outputs = model(images)
            forward_seconds += time.perf_counter() - started
            predictions.append(outputs.cpu())
            targets.append(masks)
            images_for_visualization.append(images.cpu())
            sample_count += len(images)
    if sample_count == 0:
        raise ValueError("The evaluation data loader is empty")
    return (
        torch.cat(predictions),
        torch.cat(targets),
        torch.cat(images_for_visualization),
        forward_seconds / sample_count,
    )


def _validation_threshold(model, loader, device):
    predictions, targets, _, _ = _predict(model, loader, device)
    scores = {
        threshold: compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        for threshold in THRESHOLD_CANDIDATES
    }
    return max(THRESHOLD_CANDIDATES, key=lambda value: scores[value]["dice"]), scores


def _example_indices(images, masks):
    image_array = images.numpy()
    mask_array = masks.numpy()[:, 0]
    areas = mask_array.reshape(len(mask_array), -1).sum(axis=1)
    selected = []
    labels = []

    def add(index, label):
        if int(index) not in selected:
            selected.append(int(index))
            labels.append(label)

    add(np.argmin(areas), "Smallest annotated defect")

    grid_scores = []
    thin_scores = []
    for index, (image, mask) in enumerate(zip(image_array, mask_array)):
        rgb = np.clip(
            image.transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN,
            0,
            1,
        )
        gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        grid_scores.append(
            max(gray.mean(axis=0).std(), gray.mean(axis=1).std())
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        component_scores = []
        for x, y, width, height, area in stats[1:count]:
            if area >= 12:
                component_scores.append(
                    max(width, height) / max(1, min(width, height))
                    * np.sqrt(area)
                )
        thin_scores.append(max(component_scores, default=0))

    for index in np.argsort(thin_scores)[::-1]:
        if thin_scores[index] > 0:
            add(index, "Elongated / thin defect")
            break
    for index in np.argsort(grid_scores)[::-1]:
        add(index, "Strong visible grid-line pattern")
        if len(selected) == 3:
            break
    for index in np.argsort(areas):
        if len(selected) >= min(3, len(areas)):
            break
        add(index, "Additional test example")
    return selected, labels


def _save_comparison_figure(path, images, targets, custom, resnet):
    indices, labels = _example_indices(images, targets)
    figure, axes = plt.subplots(
        len(indices),
        4,
        figsize=(14, 3.5 * len(indices)),
        squeeze=False,
    )
    headings = (
        "Original EL image",
        "Ground truth",
        "Custom U-Net (overlay)",
        "ResNet-18 U-Net (overlay)",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading)
    for row, (index, label) in enumerate(zip(indices, labels)):
        rgb = np.clip(
            images[index].numpy().transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN,
            0,
            1,
        )
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(targets[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        target = targets[index, 0].numpy().astype(bool)
        for column, prediction in (
            (2, custom[index, 0].numpy().astype(bool)),
            (3, resnet[index, 0].numpy().astype(bool)),
        ):
            true_positive = target & prediction
            false_positive = ~target & prediction
            false_negative = target & ~prediction
            overlay = np.zeros((*target.shape, 4), dtype=np.float32)
            overlay[true_positive] = (0.1, 1.0, 0.1, 0.75)
            overlay[false_positive] = (1.0, 0.1, 0.1, 0.85)
            overlay[false_negative] = (0.1, 0.75, 1.0, 0.85)
            axes[row, column].imshow(rgb)
            axes[row, column].imshow(overlay)
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].set_ylabel(label, fontsize=9)
    figure.suptitle(
        "Green: correct defect | Red: false positive | Blue: missed defect",
        fontsize=11,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def compare_models(
    custom_checkpoint,
    resnet_checkpoint,
    processed_data_dir,
    output_dir,
    batch_size=4,
    num_threads=None,
    output_suffix="",
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_threads is not None:
        if num_threads <= 0:
            raise ValueError("num_threads must be greater than zero")
        torch.set_num_threads(num_threads)
    device = torch.device("cpu")
    _, validation_loader, test_loader = get_dataloaders(
        processed_data_dir,
        batch_size=batch_size,
        validation_fraction=0.2,
        seed=42,
    )

    custom = UNet().to(device)
    custom_state, _ = _load_state(custom_checkpoint)
    custom.load_state_dict(custom_state)
    resnet = ResNet18UNet(pretrained=False).to(device)
    resnet_state, resnet_metadata = _load_state(resnet_checkpoint)
    resnet.load_state_dict(resnet_state)

    custom_threshold, custom_validation_scores = _validation_threshold(
        custom,
        validation_loader,
        device,
    )
    resnet_threshold, resnet_validation_scores = _validation_threshold(
        resnet,
        validation_loader,
        device,
    )
    custom_predictions, targets, images, custom_inference_time = _predict(
        custom,
        test_loader,
        device,
    )
    resnet_predictions, resnet_targets, _, resnet_inference_time = _predict(
        resnet,
        test_loader,
        device,
    )
    if not torch.equal(targets, resnet_targets):
        raise RuntimeError("The models were evaluated against different test masks")

    custom_metrics = compute_detailed_metrics(
        custom_predictions,
        targets,
        threshold=custom_threshold,
    )
    resnet_metrics = compute_detailed_metrics(
        resnet_predictions,
        targets,
        threshold=resnet_threshold,
    )
    custom_metrics["threshold"] = custom_threshold
    resnet_metrics["threshold"] = resnet_threshold
    custom_metrics["training_seconds"] = "Not recorded"
    resnet_metrics["training_seconds"] = resnet_metadata.get(
        "training_seconds",
        "Not recorded",
    )
    custom_metrics["inference_seconds_per_image"] = custom_inference_time
    resnet_metrics["inference_seconds_per_image"] = resnet_inference_time

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"phase4_model_comparison{output_suffix}.csv"
    rows = (
        ("Dice Score", custom_metrics["dice"], resnet_metrics["dice"]),
        ("IoU", custom_metrics["iou"], resnet_metrics["iou"]),
        ("Precision", custom_metrics["precision"], resnet_metrics["precision"]),
        ("Recall", custom_metrics["recall"], resnet_metrics["recall"]),
        ("F1 Score", custom_metrics["f1"], resnet_metrics["f1"]),
        ("Validation threshold", custom_threshold, resnet_threshold),
        (
            "Training time (seconds)",
            custom_metrics["training_seconds"],
            resnet_metrics["training_seconds"],
        ),
        (
            "Inference time (seconds/image)",
            custom_inference_time,
            resnet_inference_time,
        ),
    )
    with table_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("Metric", "Custom U-Net", "ResNet-18 U-Net"))
        writer.writerows(rows)

    figure_path = output_dir / f"phase4_model_comparison{output_suffix}.png"
    _save_comparison_figure(
        figure_path,
        images,
        targets,
        custom_predictions > custom_threshold,
        resnet_predictions > resnet_threshold,
    )
    print(f"Custom U-Net validation threshold: {custom_threshold:.1f}")
    print(
        "Custom U-Net validation Dice by threshold: "
        + str({key: value["dice"] for key, value in custom_validation_scores.items()})
    )
    print(f"ResNet-18 U-Net validation threshold: {resnet_threshold:.1f}")
    print(
        "ResNet-18 validation Dice by threshold: "
        + str({key: value["dice"] for key, value in resnet_validation_scores.items()})
    )
    print(f"Test comparison table: {table_path}")
    print(f"Prediction comparison figure: {figure_path}")


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate the trained ResNet-18 U-Net against the custom U-Net."
    )
    parser.add_argument(
        "--custom-checkpoint",
        type=Path,
        default=project_dir / "results" / "unet_solar_defect.pth",
    )
    parser.add_argument(
        "--resnet-checkpoint",
        type=Path,
        default=project_dir / "results" / "resnet18_unet_best.pth",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if args.num_threads is not None and args.num_threads <= 0:
        parser.error("--num-threads must be greater than zero")
    compare_models(
        custom_checkpoint=args.custom_checkpoint,
        resnet_checkpoint=args.resnet_checkpoint,
        processed_data_dir=project_dir / "data" / "processed" / "SolarCells",
        output_dir=project_dir / "results",
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
