import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def post_process_mask(mask_bin, kernel_size=3):
    """Applies morphological opening to remove isolated background speckles."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned

def evaluate_and_visualize(model, val_loader, device='cpu', num_samples=4, threshold=0.6):
    model.eval()
    images, masks, _ = next(iter(val_loader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        preds = model(images)
        # Apply optimal confidence thresholding
        preds_binary = (preds > threshold).float()

    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    print("\n=== Phase 3: Generating Visualizations ===")
    
    for i in range(num_samples):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

        raw_pred = preds_binary[i, 0].cpu().numpy()
        cleaned_pred = post_process_mask(raw_pred)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input EL Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masks[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(cleaned_pred, cmap='gray')
        axes[i, 2].set_title(f"U-Net Prediction (Thresh={threshold})")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("phase3_evaluation_results.png", dpi=300)
    print("Cleaned evaluation figure saved to phase3_evaluation_results.png")