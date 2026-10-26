
from pathlib import Path
import re
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "SolarCells"
OUT_DIR = ROOT / "data" / "processed" / "SolarCells"
REPORT_DIR = ROOT / "results" / "preprocessing"

SPLITS = ["train", "test"]
IMAGE_SIZE = (224, 224)
MASK_THRESHOLD = 127

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = ("_mask", "_label", "_groundtruth", "_gt")
AUGMENTATION_SUFFIX = re.compile(
    r"(?:[_-](?:aug(?:mented)?|flip|rot(?:ate|ation)?|crop|noise|brightness|contrast|zoom|shear|shift|scale)"
    r"(?:[_-]?(?:h|v|horizontal|vertical|90|180|270|\d+))?)+$",
    re.IGNORECASE,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def canonical_stem(path, is_label=False):
    key = path.stem.casefold()

    if is_label:
        for suffix in MASK_SUFFIXES:
            if key.endswith(suffix):
                return key[:-len(suffix)]

    return key


def find_images(folder, is_label=False):
    if not folder.exists():
        print(f"Warning: folder not found: {folder}")
        return {}

    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    result = {}
    duplicates = []

    for path in files:
        key = canonical_stem(path, is_label=is_label)

        if key in result:
            duplicates.append(key)
        else:
            result[key] = path

    if duplicates:
        raise ValueError(
            f"Duplicate filename stems in {folder}: "
            f"{sorted(set(duplicates))[:10]}"
        )

    return result


def original_image_id(path):
    key = canonical_stem(path)
    previous_key = None

    while key != previous_key:
        previous_key = key
        key = AUGMENTATION_SUFFIX.sub("", key)

    return key


def find_split_overlaps():
    files_by_original = {}

    for split in SPLITS:
        image_dir = RAW_DIR / split / "defect"
        for path in image_dir.rglob("*") if image_dir.exists() else ():
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                original_id = original_image_id(path)
                files_by_original.setdefault(original_id, []).append((split, path))

    overlaps = []
    for original_id, entries in sorted(files_by_original.items()):
        splits = sorted({split for split, _ in entries})
        if len(splits) > 1:
            overlaps.append({
                "original_id": original_id,
                "splits": ",".join(splits),
                "files": ";".join(str(path.relative_to(RAW_DIR)) for _, path in entries),
            })

    return overlaps


def preprocess_pair(image_path, mask_path, split):
    image_out = OUT_DIR / split / "images" / f"{image_path.stem}.png"
    mask_out = OUT_DIR / split / "masks" / f"{image_path.stem}.png"

    image_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    # Process the EL image.
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_image_size = image.size
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR
        )
        image.save(image_out)

    # Nearest-neighbor preserves mask labels while resizing.
    with Image.open(mask_path) as mask:
        original_mask_size = mask.size
        mask = mask.convert("L")
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )
        mask_array = np.asarray(mask)
        binary_mask = (mask_array > MASK_THRESHOLD).astype(np.uint8)
        Image.fromarray(binary_mask).save(mask_out)

        mask_values = np.unique(binary_mask)

    return {
        "split": split,
        "image": str(image_path.relative_to(RAW_DIR)),
        "mask": str(mask_path.relative_to(RAW_DIR)),
        "original_image_size": str(original_image_size),
        "original_mask_size": str(original_mask_size),
        "processed_size": str(IMAGE_SIZE),
        "mask_values": str(mask_values[:30].tolist()),
        "foreground_pixels": int(np.count_nonzero(binary_mask)),
        "empty_mask": not bool(np.any(binary_mask)),
        "status": "ok",
    }


def process_split(split):
    image_dir = RAW_DIR / split / "defect"
    mask_dir = RAW_DIR / split / "label"

    images = find_images(image_dir)
    masks = find_images(mask_dir, is_label=True)

    print(f"\n{split.upper()}")
    print(f"Images found: {len(images)}")
    print(f"Labels found: {len(masks)}")

    image_keys = set(images)
    mask_keys = set(masks)

    missing_masks = sorted(image_keys - mask_keys)
    missing_images = sorted(mask_keys - image_keys)

    print(f"Images without matching labels: {len(missing_masks)}")
    print(f"Labels without matching images: {len(missing_images)}")

    records = []

    for key in sorted(image_keys & mask_keys):
        try:
            records.append(
                preprocess_pair(images[key], masks[key], split)
            )
        except Exception as error:
            records.append({
                "split": split,
                "image": str(images[key]),
                "mask": str(masks[key]),
                "status": f"error: {error}",
            })

    pairing_issues = [
        {"split": split, "issue": "image_without_label", "key": key, "path": str(images[key])}
        for key in missing_masks
    ]
    pairing_issues.extend(
        {"split": split, "issue": "label_without_image", "key": key, "path": str(masks[key])}
        for key in missing_images
    )
    pd.DataFrame(
        pairing_issues,
        columns=["split", "issue", "key", "path"],
    ).to_csv(REPORT_DIR / f"{split}_pairing_issues.csv", index=False)

    return records, pairing_issues


def main():
    all_records = []
    all_pairing_issues = []

    for split in SPLITS:
        records, pairing_issues = process_split(split)
        all_records.extend(records)
        all_pairing_issues.extend(pairing_issues)

    empty_masks = [
        {
            "split": record["split"],
            "image": record["image"],
            "mask": record["mask"],
        }
        for record in all_records
        if record.get("status") == "ok" and record.get("empty_mask")
    ]
    pd.DataFrame(
        empty_masks,
        columns=["split", "image", "mask"],
    ).to_csv(REPORT_DIR / "empty_masks.csv", index=False)

    split_overlaps = find_split_overlaps()
    pd.DataFrame(
        split_overlaps,
        columns=["original_id", "splits", "files"],
    ).to_csv(REPORT_DIR / "split_overlaps.csv", index=False)

    report_path = REPORT_DIR / "preprocessing_report.csv"
    pd.DataFrame(all_records).to_csv(report_path, index=False)

    print("\nPreprocessing finished.")
    print(f"Unmatched image/label files: {len(all_pairing_issues)}")
    print(f"Empty binary masks: {len(empty_masks)}")
    print(f"Potential train/test original-image overlaps: {len(split_overlaps)}")
    if split_overlaps:
        print("Warning: related images occur across splits; see split_overlaps.csv")
    print(f"Report saved to: {report_path}")
    print(f"Processed dataset saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
