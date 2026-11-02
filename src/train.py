import torch
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        intersection = (pred * target).sum(dim=(2, 3))
        total = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2. * intersection + self.smooth) / (total + self.smooth)).mean()
        
        return bce_loss + dice_loss

def compute_metrics(pred, target, smooth=1e-6):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(2, 3))
    total = pred_bin.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (total + smooth)
    union = total - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return dice.mean().item(), iou.mean().item()

def train_unet(model, train_loader, val_loader, epochs=15, lr=1e-3, device='cpu'):
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\n=== Phase 2: Training U-Net Segmentation Model (Combined BCE+Dice Loss) ===")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_dice, val_iou = 0.0, 0.0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                d, i = compute_metrics(preds, masks)
                val_dice += d
                val_iou += i

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f} | Val Dice Score: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

    torch.save(model.state_dict(), "unet_solar_defect.pth")
    print("Model checkpoint saved to unet_solar_defect.pth")