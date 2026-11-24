from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.train import compute_metrics


def post_process_mask(mask_bin, kernel_size=3):
    """Apply morphological opening to remove isolated foreground speckles."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned

def evaluate_and_visualize(
    model,
    test_loader,
    device: torch.device | str = "cpu",
    num_samples=4,
    threshold=0.5,
    output_path: str | Path | None = None,
):
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")

    device = torch.device(device)
    model.eval()
    sample_images = []
    sample_masks = []
    sample_predictions = []
    dice_total = 0.0
    iou_total = 0.0
    sample_count = 0

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            dice, iou = compute_metrics(preds, masks, threshold=threshold)
            batch_size = len(images)
            dice_total += dice * batch_size
            iou_total += iou * batch_size
            sample_count += batch_size

            samples_needed = num_samples - len(sample_images)
            if samples_needed > 0:
                sample_images.extend(images[:samples_needed].cpu())
                sample_masks.extend(masks[:samples_needed].cpu())
                sample_predictions.extend(preds[:samples_needed].cpu())

    if sample_count == 0:
        raise ValueError("The test data loader is empty")

    num_samples = len(sample_images)
    fig, axes = plt.subplots(
        num_samples,
        3,
        figsize=(10, 3 * num_samples),
        squeeze=False,
    )
    print(
        "\n=== Phase 3: Test Evaluation ==="
        f"\nTest Dice Score: {dice_total / sample_count:.4f}"
        f"\nTest IoU: {iou_total / sample_count:.4f}"
    )
    
    for i in range(num_samples):
        img = sample_images[i].numpy().transpose(1, 2, 0)
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

        raw_pred = (sample_predictions[i][0] > threshold).numpy()
        cleaned_pred = post_process_mask(raw_pred)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input EL Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(sample_masks[i][0].numpy(), cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cleaned_pred, cmap='gray')
        axes[i, 2].set_title(f"U-Net Prediction (Threshold={threshold})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "phase3_evaluation_results.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Evaluation figure saved to {output_path}")