import os
from pathlib import Path
import torch
from src.dataset import get_dataloaders
from src.models import UNet
from src.train import train_unet
from src.evaluate import evaluate_and_visualize

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running execution pipeline on device: {device}")

    processed_data_dir = Path(__file__).resolve().parent / "data" / "processed" / "SolarCells"

    train_loader, test_loader = get_dataloaders(
        root_processed_dir=processed_data_dir,
        batch_size=8
    )

    model = UNet().to(device)
    train_unet(model, train_loader, test_loader, epochs=15, device=device)
    evaluate_and_visualize(model, test_loader, device=device)