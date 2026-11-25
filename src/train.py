from pathlib import Path

import torch
import torch.nn as nn


THRESHOLD_CANDIDATES = tuple(value / 10 for value in range(1, 10))


class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, negative_weight=1.0):
        super().__init__()
        if negative_weight <= 0:
            raise ValueError("negative_weight must be greater than zero")
        self.smooth = smooth
        self.negative_weight = negative_weight

    def forward(self, pred, target):
        pixel_weights = torch.where(
            target > 0.5,
            torch.ones_like(target),
            torch.full_like(target, self.negative_weight),
        )
        bce_per_pixel = nn.functional.binary_cross_entropy(
            pred,
            target,
            reduction="none",
        )
        bce_loss = (bce_per_pixel * pixel_weights).sum() / pixel_weights.sum()
        
        intersection = (pred * target).sum(dim=(2, 3))
        total = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2. * intersection + self.smooth) / (total + self.smooth)).mean()
        
        return bce_loss + dice_loss


class BoundaryAwareBCEDiceLoss(nn.Module):
    def __init__(
        self,
        smooth=1e-6,
        negative_weight=1.0,
        boundary_weight=0.1,
        boundary_kernel_size=3,
    ):
        super().__init__()
        if boundary_weight < 0:
            raise ValueError("boundary_weight must not be negative")
        if boundary_kernel_size <= 0 or boundary_kernel_size % 2 == 0:
            raise ValueError("boundary_kernel_size must be a positive odd integer")
        self.primary_loss = BCEDiceLoss(
            smooth=smooth,
            negative_weight=negative_weight,
        )
        self.boundary_weight = boundary_weight
        self.boundary_kernel_size = boundary_kernel_size

    def forward(self, pred, target):
        if self.boundary_weight == 0:
            return self.primary_loss(pred, target)
        if pred.shape != target.shape or pred.ndim != 4:
            raise ValueError(
                "pred and target must have the same NCHW shape"
            )
        primary_loss = self.primary_loss(pred, target)

        padding = self.boundary_kernel_size // 2
        predicted_dilation = nn.functional.max_pool2d(
            pred,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=padding,
        )
        predicted_erosion = -nn.functional.max_pool2d(
            -pred,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=padding,
        )
        predicted_boundary = (predicted_dilation - predicted_erosion).clamp(0, 1)

        with torch.no_grad():
            target_dilation = nn.functional.max_pool2d(
                target,
                kernel_size=self.boundary_kernel_size,
                stride=1,
                padding=padding,
            )
            target_erosion = -nn.functional.max_pool2d(
                -target,
                kernel_size=self.boundary_kernel_size,
                stride=1,
                padding=padding,
            )
            target_boundary = (target_dilation - target_erosion).clamp(0, 1)

        boundary_weights = torch.where(
            target_boundary > 0.5,
            torch.ones_like(target_boundary),
            torch.full_like(target_boundary, 0.1),
        )
        boundary_error = nn.functional.smooth_l1_loss(
            predicted_boundary,
            target_boundary,
            reduction="none",
        )
        boundary_loss = (
            (boundary_error * boundary_weights).sum()
            / boundary_weights.sum().clamp_min(1)
        )
        return primary_loss + self.boundary_weight * boundary_loss


def compute_metrics(pred, target, threshold=0.5, smooth=1e-6):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    total = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (total + smooth)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.mean().item(), iou.mean().item()


def compute_detailed_metrics(pred, target, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    pred_bin = pred > threshold
    target_bin = target > 0.5
    dimensions = (1, 2, 3)
    true_positive = (pred_bin & target_bin).sum(dim=dimensions).float()
    false_positive = (pred_bin & ~target_bin).sum(dim=dimensions).float()
    false_negative = (~pred_bin & target_bin).sum(dim=dimensions).float()

    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = (
        2 * true_positive
        / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    )
    iou = (
        true_positive
        / (true_positive + false_positive + false_negative).clamp_min(1)
    )
    return {
        "dice": f1.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
    }


def train_unet(
    model,
    train_loader,
    val_loader,
    epochs=15,
    lr=1e-3,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
    threshold: float | None = None,
) -> float:
    device = torch.device(device)
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if lr <= 0:
        raise ValueError("lr must be greater than zero")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be between zero and one")
    if len(train_loader) == 0:
        raise ValueError("The training data loader is empty")
    if len(val_loader) == 0:
        raise ValueError("The validation data loader is empty")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\n=== Phase 2: Training U-Net Segmentation Model (Combined BCE+Dice Loss) ===")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        training_samples = 0
        
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            batch_size = len(images)
            train_loss += loss.item() * batch_size
            training_samples += batch_size

        model.eval()
        val_predictions = []
        val_targets = []
        validation_samples = 0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_predictions.append(preds.cpu())
                val_targets.append(masks.cpu())
                validation_samples += len(images)

        validation_predictions = torch.cat(val_predictions)
        validation_targets = torch.cat(val_targets)
        thresholds = (
            (threshold,)
            if threshold is not None
            else THRESHOLD_CANDIDATES
        )
        validation_scores = {
            candidate: compute_metrics(
                validation_predictions,
                validation_targets,
                threshold=candidate,
            )
            for candidate in thresholds
        }
        selected_threshold = max(
            thresholds,
            key=lambda candidate: validation_scores[candidate][0],
        )
        val_dice, val_iou = validation_scores[selected_threshold]
        avg_loss = train_loss / training_samples
        
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | "
            f"Val Dice Score: {val_dice:.4f} | Val IoU: {val_iou:.4f} | "
            f"Threshold: {selected_threshold:.1f}"
        )

    if checkpoint_path is None:
        checkpoint_path = (
            Path(__file__).resolve().parents[1]
            / "results"
            / "unet_solar_defect_retrained.pth"
        )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
    return selected_threshold
