import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.dataset import get_dataloaders
from src.models import ResNet18UNet
from src.train import BoundaryAwareBCEDiceLoss, compute_detailed_metrics


THRESHOLD_CANDIDATES = tuple(
    round(0.1 + step * 0.025, 3)
    for step in range(35)
)
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_CONFIG = {
    "mode": "max",
    "factor": 0.5,
    "patience": 2,
    "threshold": 0.001,
    "threshold_mode": "abs",
    "min_lr": 1e-5,
}


def _optimizer(model, fine_tune):
    decoder_parameters = list(model.decoder.parameters())
    if fine_tune:
        return torch.optim.AdamW(
            [
                {"params": decoder_parameters, "lr": 5e-5},
                {"params": model.encoder.layer4.parameters(), "lr": 1e-5},
            ]
        )
    return torch.optim.AdamW(decoder_parameters, lr=1e-3)


def _scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **SCHEDULER_CONFIG,
    )


def _threshold_grid(coarse_scores):
    coarse = list(THRESHOLD_CANDIDATES)
    center = max(coarse, key=lambda threshold: coarse_scores[threshold]["dice"])
    fine = {
        round(center - 0.025 + step * 0.005, 3)
        for step in range(11)
    }
    return tuple(sorted(set(coarse) | fine))


def _validation_scores(model, loader, device, criterion):
    predictions = []
    targets = []
    total_loss = 0.0
    sample_count = 0
    model.eval()
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, masks.to(device)).item() * len(images)
            sample_count += len(images)
            predictions.append(outputs.cpu())
            targets.append(masks)

    if not predictions:
        raise ValueError("The validation data loader is empty")

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    defect_fractions = targets.sum(dim=(1, 2, 3)) / targets[0, 0].numel()
    tiny_indices = (defect_fractions > 0) & (defect_fractions <= 0.02)

    def score_threshold(threshold):
        prediction = predictions > threshold
        target = targets > 0.5
        metrics = compute_detailed_metrics(
            predictions,
            targets,
            threshold=threshold,
        )
        metrics["tiny_defect_dice"] = (
            compute_detailed_metrics(
                predictions[tiny_indices],
                targets[tiny_indices],
                threshold=threshold,
            )["dice"]
            if tiny_indices.any()
            else None
        )
        metrics["false_positive_pixels"] = int((prediction & ~target).sum())
        metrics["false_negative_pixels"] = int((~prediction & target).sum())
        return metrics

    scores = {
        threshold: score_threshold(threshold)
        for threshold in THRESHOLD_CANDIDATES
    }
    for threshold in _threshold_grid(scores):
        if threshold not in scores:
            scores[threshold] = score_threshold(threshold)
    return scores, total_loss / sample_count


def _save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _small_defect_sampler(training_subset, cutoff=0.02, multiplier=1.5, seed=42):
    if not isinstance(training_subset, Subset):
        raise TypeError("Small-defect sampling requires a training Subset")
    if not 0 < cutoff < 1 or multiplier <= 1:
        raise ValueError("cutoff must be in (0, 1) and multiplier must exceed one")
    dataset = training_subset.dataset
    if not hasattr(dataset, "samples"):
        raise TypeError("The training dataset does not expose paired samples")

    weights = []
    small_count = 0
    for sample_index in training_subset.indices:
        _, mask_path = dataset.samples[sample_index]
        with Image.open(mask_path) as source_mask:
            mask = np.asarray(source_mask.convert("L"))
        defect_fraction = float(np.count_nonzero(mask)) / mask.size
        is_small_defect = 0 < defect_fraction <= cutoff
        small_count += int(is_small_defect)
        weights.append(multiplier if is_small_defect else 1.0)

    if not weights:
        raise ValueError("The training subset is empty")
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return sampler, {
        "strategy": "small_defect_weighted_sampling",
        "small_defect_fraction_cutoff": cutoff,
        "small_defect_sampling_weight": multiplier,
        "small_defect_training_images": small_count,
        "training_images": len(weights),
        "expected_small_defect_sample_fraction": (
            small_count * multiplier
            / (len(weights) - small_count + small_count * multiplier)
        ),
    }


def train_resnet_unet(
    model,
    train_loader,
    validation_loader,
    device,
    epochs,
    checkpoint_prefix,
    fine_tune=False,
    resume_path=None,
    initialize_from=None,
    initial_best_checkpoint=None,
    negative_bce_weight=1.0,
    boundary_weight=0.0,
    sampling_metadata=None,
):
    maximum_epochs = 50
    if epochs <= 0 or epochs > maximum_epochs:
        raise ValueError(f"epochs must be between 1 and {maximum_epochs}")
    if len(train_loader) == 0 or len(validation_loader) == 0:
        raise ValueError("Training and validation data loaders must not be empty")
    if negative_bce_weight <= 0:
        raise ValueError("negative_bce_weight must be greater than zero")
    if boundary_weight < 0:
        raise ValueError("boundary_weight must not be negative")
    if resume_path is not None and initialize_from is not None:
        raise ValueError("resume_path and initialize_from cannot both be provided")
    if initial_best_checkpoint is not None and resume_path is None:
        raise ValueError("initial_best_checkpoint requires a resume checkpoint")

    device = torch.device(device)
    model.to(device)
    phase = "fine_tune" if fine_tune else "decoder"
    suffix = "_finetuned" if fine_tune else ""
    best_path = checkpoint_prefix.with_name(
        checkpoint_prefix.name + suffix + "_best.pth"
    )
    last_path = checkpoint_prefix.with_name(
        checkpoint_prefix.name + suffix + "_last.pth"
    )
    start_epoch = 0
    best_dice = -1.0 if fine_tune else 0.0
    best_threshold = THRESHOLD_CANDIDATES[0]
    elapsed_training_seconds = 0.0
    continued_training_seconds = 0.0
    epochs_without_improvement = 0
    history = []
    baseline_checkpoint_path = None
    resume_payload = None
    resume_model_state = None
    initialization_metadata = None

    if initialize_from is not None:
        initialization_metadata = torch.load(
            initialize_from,
            map_location=device,
            weights_only=False,
        )
        if "model_state" not in initialization_metadata:
            raise ValueError("Initialization checkpoint has no model_state")
        if initialization_metadata.get("phase") != "decoder":
            raise ValueError("Initialization checkpoint must be a Stage 1 decoder checkpoint")
        model.load_state_dict(initialization_metadata["model_state"])

    if resume_path is not None:
        resume_payload = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        resume_model_state = resume_payload["model_state"]
        model.load_state_dict(resume_payload["model_state"])
        if resume_payload.get("phase") == phase:
            saved_negative_weight = resume_payload.get("negative_bce_weight", 1.0)
            if saved_negative_weight != negative_bce_weight:
                raise ValueError(
                    "The checkpoint negative BCE weight does not match the "
                    "requested value"
                )
            if resume_payload.get("boundary_weight", 0.0) != boundary_weight:
                raise ValueError(
                    "The checkpoint boundary loss weight does not match the "
                    "requested value"
                )
            if "optimizer_state" not in resume_payload:
                raise ValueError("The selected checkpoint cannot resume its optimizer")
            start_epoch = resume_payload["phase_epoch"]
            best_dice = resume_payload["best_validation_dice"]
            best_threshold = resume_payload["best_threshold"]
            elapsed_training_seconds = resume_payload["training_seconds"]
            continued_training_seconds = resume_payload.get(
                "continued_training_seconds",
                0.0,
            )
            epochs_without_improvement = resume_payload["epochs_without_improvement"]
            history = resume_payload["history"]
        elif not (fine_tune and resume_payload.get("phase") == "decoder"):
            raise ValueError(
                f"Checkpoint phase {resume_payload.get('phase')!r} does not match "
                f"requested phase {phase!r}"
            )
        else:
            baseline_checkpoint_path = Path(resume_path)
        if fine_tune and resume_payload.get("phase") == "fine_tune":
            baseline_checkpoint_path = Path(
                resume_payload["baseline_checkpoint_path"]
            )

    if fine_tune:
        model.unfreeze_final_block()
    else:
        model.freeze_encoder()

    optimizer = _optimizer(model, fine_tune)
    scheduler = _scheduler(optimizer)
    if resume_payload is not None and resume_payload.get("phase") == phase:
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if "torch_rng_state" in resume_payload:
            torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
            random.setstate(resume_payload["python_rng_state"])
            np.random.set_state(resume_payload["numpy_rng_state"])

    criterion = BoundaryAwareBCEDiceLoss(
        negative_weight=negative_bce_weight,
        boundary_weight=boundary_weight,
    )
    if initial_best_checkpoint is not None and not best_path.exists():
        initial_best_path = Path(initial_best_checkpoint)
        if not initial_best_path.is_file():
            raise FileNotFoundError(
                f"Initial best checkpoint not found: {initial_best_path}"
            )
        if resume_payload is None or resume_payload.get("phase") != phase:
            raise ValueError(
                "The initial best checkpoint requires a same-phase resume"
            )
        best_payload = torch.load(
            initial_best_path,
            map_location=device,
            weights_only=False,
        )
        if best_payload.get("phase") != phase:
            raise ValueError(
                f"Initial best checkpoint phase {best_payload.get('phase')!r} "
                f"does not match requested phase {phase!r}"
            )
        if best_payload.get("negative_bce_weight", 1.0) != negative_bce_weight:
            raise ValueError(
                "The initial best checkpoint negative BCE weight does not match"
            )
        if float(best_payload.get("boundary_weight") or 0.0) != boundary_weight:
            raise ValueError(
                "The initial best checkpoint boundary weight does not match"
            )
        model.load_state_dict(best_payload["model_state"])
        initial_scores, _ = _validation_scores(
            model,
            validation_loader,
            device,
            criterion,
        )
        best_threshold = max(
            initial_scores,
            key=lambda candidate: initial_scores[candidate]["dice"],
        )
        best_metrics = initial_scores[best_threshold]
        best_dice = best_metrics["dice"]
        baseline_checkpoint_path = initial_best_path
        epochs_without_improvement = 0
        scheduler.step(best_dice)
        seeded_best = dict(best_payload)
        seeded_best.update(
            {
                "model_state": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                "best_validation_dice": best_dice,
                "best_threshold": best_threshold,
                "validation_metrics": best_metrics,
                "training_seconds": elapsed_training_seconds,
                "continued_training_seconds": 0.0,
                "history": history.copy(),
                "baseline_checkpoint_path": str(initial_best_path),
                "scheduler_config": dict(SCHEDULER_CONFIG),
                "batch_size": train_loader.batch_size,
                "cpu_threads": torch.get_num_threads(),
                "selected_from": str(initial_best_path),
            }
        )
        _save_checkpoint(best_path, seeded_best)
        if resume_model_state is None:
            raise RuntimeError("Resume model state was not preserved")
        model.load_state_dict(resume_model_state)

    patience = EARLY_STOPPING_PATIENCE
    if start_epoch >= epochs:
        print(f"Checkpoint already reached the requested {epochs} epochs.")
        if not best_path.exists():
            raise RuntimeError("No best checkpoint exists at the requested epoch")
        return best_path
    if epochs_without_improvement >= patience:
        print(
            f"Early stopping after {patience} epoch(s) without "
            "validation Dice improvement."
        )
        if not best_path.exists():
            raise RuntimeError("No best checkpoint exists after early stopping")
        return best_path

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        sample_count = 0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            batch_size = len(images)
            total_loss += loss.item() * batch_size
            sample_count += batch_size

        if sample_count == 0:
            raise ValueError("The training data loader produced no samples")

        threshold_results, validation_loss = _validation_scores(
            model,
            validation_loader,
            device,
            criterion,
        )
        selected_threshold = max(
            threshold_results,
            key=lambda candidate: threshold_results[candidate]["dice"],
        )
        selected_metrics = threshold_results[selected_threshold]
        learning_rates_used = [
            group["lr"] for group in optimizer.param_groups
        ]
        scheduler.step(selected_metrics["dice"])
        learning_rates_next = [
            group["lr"] for group in optimizer.param_groups
        ]
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed_training_seconds += epoch_seconds
        continued_training_seconds += epoch_seconds
        record = {
            "phase": phase,
            "epoch": epoch + 1,
            "train_loss": total_loss / sample_count,
            "validation_loss": validation_loss,
            "validation": selected_metrics,
            "threshold": selected_threshold,
            "epoch_seconds": epoch_seconds,
            "learning_rates_used": learning_rates_used,
            "learning_rates_next": learning_rates_next,
        }
        history.append(record)
        improved = selected_metrics["dice"] > best_dice
        if improved:
            best_dice = selected_metrics["dice"]
            best_threshold = selected_threshold
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path,
                {
                    "model_state": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "phase": phase,
                    "best_validation_dice": best_dice,
                    "best_threshold": best_threshold,
                    "validation_metrics": selected_metrics,
                    "training_seconds": elapsed_training_seconds,
                    "continued_training_seconds": continued_training_seconds,
                    "history": history.copy(),
                    "negative_bce_weight": negative_bce_weight,
                    "boundary_weight": boundary_weight,
                    "baseline_checkpoint_path": (
                        str(baseline_checkpoint_path)
                        if baseline_checkpoint_path is not None
                        else None
                    ),
                    "optimizer_learning_rates": (
                        {"decoder": 5e-5, "encoder_layer4": 1e-5}
                        if fine_tune
                        else {"decoder": 1e-3}
                    ),
                    "batch_size": train_loader.batch_size,
                    "cpu_threads": torch.get_num_threads(),
                    "layer4_batch_norm_running_stats": (
                        "frozen" if fine_tune else None
                    ),
                    "sampling": sampling_metadata or {"strategy": "shuffle"},
                    "scheduler_config": dict(SCHEDULER_CONFIG),
                    "initialized_from": (
                        str(initialize_from)
                        if initialize_from is not None
                        else None
                    ),
                },
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            last_path,
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "phase": phase,
                "phase_epoch": epoch + 1,
                "best_validation_dice": best_dice,
                "best_threshold": best_threshold,
                "training_seconds": elapsed_training_seconds,
                "continued_training_seconds": continued_training_seconds,
                "epochs_without_improvement": epochs_without_improvement,
                "baseline_checkpoint_path": (
                    str(baseline_checkpoint_path)
                    if baseline_checkpoint_path is not None
                    else None
                ),
                "negative_bce_weight": negative_bce_weight,
                "boundary_weight": boundary_weight,
                "scheduler_config": dict(SCHEDULER_CONFIG),
                "optimizer_learning_rates": (
                    {"decoder": 5e-5, "encoder_layer4": 1e-5}
                    if fine_tune
                    else {"decoder": 1e-3}
                ),
                "batch_size": train_loader.batch_size,
                "cpu_threads": torch.get_num_threads(),
                "layer4_batch_norm_running_stats": (
                    "frozen" if fine_tune else None
                ),
                "sampling": sampling_metadata or {"strategy": "shuffle"},
                "initialized_from": (
                    str(initialize_from)
                    if initialize_from is not None
                    else None
                ),
                "history": history,
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            },
        )
        print(
            f"{phase} epoch {epoch + 1}/{epochs} | "
            f"loss={record['train_loss']:.4f} | "
            f"validation_loss={validation_loss:.4f} | "
            f"val_dice={selected_metrics['dice']:.4f} | "
            f"val_iou={selected_metrics['iou']:.4f} | "
            f"precision={selected_metrics['precision']:.4f} | "
            f"recall={selected_metrics['recall']:.4f} | "
            f"tiny_dice={selected_metrics['tiny_defect_dice']} | "
            f"false_positives={selected_metrics['false_positive_pixels']} | "
            f"threshold={selected_threshold:.3f} | "
            f"lr={learning_rates_used} -> {learning_rates_next} | "
            f"seconds={epoch_seconds:.1f}"
        )
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after {patience} epoch(s) without "
                "validation Dice improvement."
            )
            break

    if not best_path.exists():
        raise RuntimeError("Training completed without saving a best checkpoint")

    best_checkpoint = torch.load(
        best_path,
        map_location="cpu",
        weights_only=False,
    )
    best_checkpoint["training_seconds"] = elapsed_training_seconds
    best_checkpoint["continued_training_seconds"] = continued_training_seconds
    best_checkpoint["history"] = history
    _save_checkpoint(best_path, best_checkpoint)
    print(f"Best checkpoint: {best_path}")
    print(f"Resumable checkpoint: {last_path}")
    print(f"Best validation Dice: {best_dice:.4f} at threshold {best_threshold:.3f}")
    print(f"Cumulative training time: {elapsed_training_seconds:.1f} seconds")
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Train the CPU-friendly ResNet-18 U-Net segmentation model."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--initial-best",
        type=Path,
        help="Seed the output best checkpoint from the selected model while resuming a later checkpoint.",
    )
    parser.add_argument("--negative-bce-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument(
        "--small-defect-sampling",
        action="store_true",
        help="Sample training images with <=2%% defect pixels 1.5x more often.",
    )
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "resnet18_unet",
    )
    args = parser.parse_args()

    if args.num_threads is not None and args.num_threads <= 0:
        parser.error("--num-threads must be greater than zero")
    if args.fine_tune and args.resume is None:
        parser.error("--fine-tune requires --resume with a decoder checkpoint")
    if args.initial_best is not None and args.resume is None:
        parser.error("--initial-best requires --resume")
    if args.initial_best is not None and args.fine_tune:
        parser.error("--initial-best is supported for decoder training only")
    if args.resume is not None and args.initialize_from is not None:
        parser.error("--resume and --initialize-from cannot be used together")
    if args.negative_bce_weight <= 0:
        parser.error("--negative-bce-weight must be greater than zero")
    if args.boundary_weight < 0:
        parser.error("--boundary-weight must not be negative")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    project_dir = Path(__file__).resolve().parents[1]
    train_loader, validation_loader, _ = get_dataloaders(
        project_dir / "data" / "processed" / "SolarCells",
        batch_size=args.batch_size,
        validation_fraction=0.2,
        seed=42,
        augment_training=True,
    )
    sampling_metadata = {"strategy": "shuffle"}
    if args.small_defect_sampling:
        sampler, sampling_metadata = _small_defect_sampler(
            train_loader.dataset,
        )
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            sampler=sampler,
        )
        print(
            f"Small-defect sampling: "
            f"{sampling_metadata['small_defect_training_images']}/"
            f"{sampling_metadata['training_images']} training images at "
            f"<=2% foreground; expected sampled fraction "
            f"{sampling_metadata['expected_small_defect_sample_fraction']:.1%}."
        )
    model = ResNet18UNet(
        pretrained=(
            args.resume is None
            and args.initialize_from is None
            and not args.no_pretrained
        ),
        freeze_encoder=not args.fine_tune,
    )
    if args.fine_tune:
        epochs = args.fine_tune_epochs
    else:
        epochs = args.epochs
    train_resnet_unet(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device="cpu",
        epochs=epochs,
        checkpoint_prefix=args.output_prefix,
        fine_tune=args.fine_tune,
        resume_path=args.resume,
        initialize_from=args.initialize_from,
        initial_best_checkpoint=args.initial_best,
        negative_bce_weight=args.negative_bce_weight,
        boundary_weight=args.boundary_weight,
        sampling_metadata=sampling_metadata,
    )


if __name__ == "__main__":
    main()
