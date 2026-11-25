from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
import torch.nn.functional as F

from src.compare_models import IMAGE_MEAN, IMAGE_STD, _load_state
from src.models import ResNet18UNet, UNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_INPUT_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.165
PHASE5_CHECKPOINT_SHA256 = (
    "518a4de1485a8828f6bbeea89da19bafe5db86ae65054e5dbc6cadbbab4a8cd8"
)
CUSTOM_UNET_THRESHOLD = 0.525
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    architecture: str
    checkpoint: Path
    threshold: float


MODEL_CONFIGS = {
    "phase5_resnet18": ModelConfig(
        key="phase5_resnet18",
        label="Phase 5 ResNet-18 U-Net",
        architecture="resnet18_unet",
        checkpoint=PROJECT_ROOT / "results" / "resnet18_unet_phase5_negw_best.pth",
        threshold=DEFAULT_THRESHOLD,
    ),
    "custom_unet": ModelConfig(
        key="custom_unet",
        label="Original Custom U-Net",
        architecture="custom_unet",
        checkpoint=PROJECT_ROOT / "results" / "unet_solar_defect.pth",
        threshold=CUSTOM_UNET_THRESHOLD,
    ),
}


@dataclass
class PredictionResult:
    file_name: str
    source_format: str
    model_key: str
    model_label: str
    original_image: Image.Image
    model_probability_map: np.ndarray
    probability_map: np.ndarray
    predicted_mask: np.ndarray
    threshold: float
    inference_seconds: float


def model_checkpoint_path(model_key, project_dir=PROJECT_ROOT):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
    return Path(project_dir) / MODEL_CONFIGS[model_key].checkpoint.relative_to(PROJECT_ROOT)


def load_model(model_key, project_dir=PROJECT_ROOT):
    """Load a trusted project checkpoint and return an eval-mode CPU model."""
    checkpoint_path = model_checkpoint_path(model_key, project_dir)
    config = MODEL_CONFIGS[model_key]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with checkpoint_path.open("rb") as checkpoint:
        if model_key == "phase5_resnet18":
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
            if digest != PHASE5_CHECKPOINT_SHA256:
                raise ValueError(
                    f"Checkpoint SHA-256 does not match the evaluated Phase 5 model: "
                    f"{checkpoint_path}"
                )
            checkpoint.seek(0)
        state, _ = _load_state(checkpoint)

    if config.architecture == "resnet18_unet":
        model = ResNet18UNet(pretrained=False)
    else:
        model = UNet()

    model.load_state_dict(state)
    model.to(torch.device("cpu"))
    model.eval()
    return model


def decode_uploaded_image(image_bytes, file_name):
    """Validate an uploaded PNG/JPEG and return its EXIF-corrected RGB image."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    if Path(str(file_name).replace("\\", "/")).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Upload a PNG, JPG, or JPEG image.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise ValueError("The image content must be PNG or JPEG.")
            source_format = source.format
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ValueError("The image has invalid dimensions.")
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "The image is too large to process safely "
                    "(maximum 20 million pixels)."
                )
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "The image could not be decoded. Please choose a valid PNG or JPEG."
        ) from error

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("The image has invalid dimensions.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "The image is too large to process safely "
            "(maximum 20 million pixels)."
        )
    return image.copy(), source_format


def preprocess_image(image):
    """Resize and normalize an RGB PIL image exactly like the training input."""
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")
    resized = image.convert("RGB").resize(
        MODEL_INPUT_SIZE,
        resample=Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    normalized = (pixels - IMAGE_MEAN) / IMAGE_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1).copy())
    return tensor.unsqueeze(0)


def threshold_probability_map(probability_map, threshold=DEFAULT_THRESHOLD):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    probability_map = np.asarray(probability_map)
    if probability_map.ndim != 2:
        raise ValueError("Expected a two-dimensional probability map.")
    if not np.isfinite(probability_map).all():
        raise ValueError("Probability map contains non-finite values.")
    return (probability_map > threshold).astype(np.uint8)


def resize_binary_mask(mask, image_size):
    """Restore a thresholded 224-pixel mask without inventing edge pixels."""
    width, height = image_size
    return np.asarray(
        Image.fromarray((np.asarray(mask) > 0).astype(np.uint8)).resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )


def mask_at_threshold(prediction, threshold):
    model_mask = threshold_probability_map(
        prediction.model_probability_map,
        threshold,
    )
    return resize_binary_mask(
        model_mask,
        prediction.original_image.size,
    )


def summarize_mask(mask):
    """Describe predicted regions in pixels, not calibrated physical units."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask.")
    binary = (mask > 0).astype(np.uint8)
    predicted_pixels = int(binary.sum())
    if predicted_pixels:
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
        region_areas = stats[1:component_count, cv2.CC_STAT_AREA]
        region_count = component_count - 1
        largest_region_pixels = int(region_areas.max())
    else:
        region_count = 0
        largest_region_pixels = 0

    total_pixels = int(binary.size)
    return {
        "defect_pixels_detected": predicted_pixels > 0,
        "connected_regions": region_count,
        "predicted_defect_pixels": predicted_pixels,
        "predicted_image_area_percent": (
            100.0 * predicted_pixels / total_pixels if total_pixels else 0.0
        ),
        "largest_region_pixels": largest_region_pixels,
    }


def run_inference(
    model,
    image_bytes,
    file_name,
    model_key="phase5_resnet18",
    threshold=None,
):
    """Predict an uploaded image on CPU and restore probabilities to its size."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}")
    config = MODEL_CONFIGS[model_key]
    if threshold is None:
        threshold = config.threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    original_image, source_format = decode_uploaded_image(
        image_bytes,
        file_name,
    )
    input_tensor = preprocess_image(original_image)
    model.to(torch.device("cpu"))
    model.eval()

    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_tensor)
    inference_seconds = time.perf_counter() - started

    expected_shape = (1, 1, *MODEL_INPUT_SIZE)
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Expected model output shape {expected_shape}, got "
            f"{getattr(output, 'shape', type(output).__name__)}."
        )
    if not torch.isfinite(output).all():
        raise RuntimeError("The model returned non-finite predictions.")
    if output.min().item() < 0.0 or output.max().item() > 1.0:
        raise RuntimeError(
            "The selected segmentation model must return probabilities in [0, 1]."
        )

    model_probability_map = output[0, 0].cpu().numpy()
    probability_map = F.interpolate(
        output,
        size=(original_image.height, original_image.width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    model_mask = threshold_probability_map(model_probability_map, threshold)
    predicted_mask = resize_binary_mask(model_mask, original_image.size)
    return PredictionResult(
        file_name=Path(str(file_name).replace("\\", "/")).name,
        source_format=source_format,
        model_key=model_key,
        model_label=config.label,
        original_image=original_image,
        model_probability_map=model_probability_map,
        probability_map=probability_map,
        predicted_mask=predicted_mask,
        threshold=threshold,
        inference_seconds=inference_seconds,
    )


def create_mask_image(mask):
    return Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)


def create_overlay_image(original_image, mask):
    original = np.asarray(original_image.convert("RGB"), dtype=np.uint8)
    binary = np.asarray(mask) > 0
    overlay = original.copy()
    highlight = np.array((255, 54, 72), dtype=np.float32)
    overlay[binary] = np.round(
        0.48 * original[binary].astype(np.float32) + 0.52 * highlight
    ).astype(np.uint8)
    return Image.fromarray(overlay)


def create_heatmap_image(probability_map):
    grayscale = np.round(
        np.clip(probability_map, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)
    return Image.fromarray(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))


def encode_png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
