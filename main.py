from pathlib import Path
import torch
from src.dataset import get_dataloaders
from src.models import UNet
from src.train import train_unet
from src.evaluate import evaluate_and_visualize

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running execution pipeline on device: {device}")

    project_dir = Path(__file__).resolve().parent
    processed_data_dir = project_dir / "data" / "processed" / "SolarCells"
    output_dir = project_dir / "results"

    train_loader, val_loader, test_loader = get_dataloaders(
        root_processed_dir=processed_data_dir,
        batch_size=8,
        validation_fraction=0.2,
        seed=42,
    )

    model = UNet().to(device)
    threshold = train_unet(
        model,
        train_loader,
        val_loader,
        epochs=15,
        device=device,
        checkpoint_path=output_dir / "unet_solar_defect.pth",
    )
    evaluate_and_visualize(
        model,
        test_loader,
        device=device,
        threshold=threshold,
        output_path=output_dir / "phase3_evaluation_results.png",
    )