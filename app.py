import hashlib
import json
import os
from pathlib import Path

import streamlit as st
import torch

from src.inference import (
    DEFAULT_THRESHOLD,
    PROJECT_ROOT,
    create_heatmap_image,
    create_mask_image,
    create_overlay_image,
    decode_uploaded_image,
    encode_png,
    load_model,
    mask_at_threshold,
    run_inference,
    summarize_mask,
)


DEMO_IMAGES = {
    "Smallest validation defect": (
        "185.png",
        "Smallest annotated defect in the validation examples.",
    ),
    "Low-scoring small defect": (
        "112.png",
        "A small defect that was difficult for the Phase 5 model.",
    ),
    "Difficult thin/low-contrast defect": (
        "165.png",
        "A thin, low-contrast example with a known missed-defect limitation.",
    ),
}


st.set_page_config(
    page_title="Solar Cell Defect Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading the selected CPU model...")
def get_cached_model(model_key):
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    return load_model(model_key)


def _demo_directory():
    return PROJECT_ROOT / "data" / "processed" / "SolarCells" / "train" / "images"


def _source_image(uploaded_file, demo_choice):
    if uploaded_file is not None:
        return uploaded_file.getvalue(), uploaded_file.name, None
    if demo_choice != "Choose a demo image":
        file_name, description = DEMO_IMAGES[demo_choice]
        path = _demo_directory() / file_name
        if not path.is_file():
            raise FileNotFoundError(
                f"Demo image is not available in the processed dataset: {path}"
            )
        return path.read_bytes(), file_name, description
    return None, None, None


def _show_prediction(prediction, threshold):
    mask = mask_at_threshold(prediction, threshold)
    mask_image = create_mask_image(mask)
    overlay_image = create_overlay_image(prediction.original_image, mask)
    heatmap_image = create_heatmap_image(prediction.probability_map)
    summary = summarize_mask(mask)

    image_columns = st.columns(2)
    with image_columns[0]:
        st.image(
            prediction.original_image,
            caption=f"Original image ({prediction.original_image.width} × "
            f"{prediction.original_image.height})",
            width="stretch",
        )
    with image_columns[1]:
        st.image(
            mask_image,
            caption=f"Binary defect mask at threshold {threshold:.3f}",
            width="stretch",
        )

    image_columns = st.columns(2)
    with image_columns[0]:
        st.image(
            overlay_image,
            caption="Defect overlay (red highlights predicted pixels)",
            width="stretch",
        )
    with image_columns[1]:
        st.image(
            heatmap_image,
            caption="Probability heatmap (inferno color scale)",
            width="stretch",
        )

    st.caption(
        "The predicted area is a percentage of raster pixels, not a calibrated "
        "physical crack measurement. The mask does not establish defect "
        "severity or expected power loss."
    )
    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Defect pixels detected",
        "Yes" if summary["defect_pixels_detected"] else "No",
    )
    summary_columns[1].metric(
        "Connected regions",
        f"{summary['connected_regions']:,}",
    )
    summary_columns[2].metric(
        "Predicted image area",
        f"{summary['predicted_image_area_percent']:.3f}%",
    )
    summary_columns[3].metric(
        "Largest region",
        f"{summary['largest_region_pixels']:,} px",
    )

    summary_data = {
        "file_name": prediction.file_name,
        "source_format": prediction.source_format,
        "model": prediction.model_label,
        "threshold": float(threshold),
        "input_width_pixels": prediction.original_image.width,
        "input_height_pixels": prediction.original_image.height,
        "predicted_defect_pixels": summary["predicted_defect_pixels"],
        "predicted_image_area_percent": summary[
            "predicted_image_area_percent"
        ],
        "connected_predicted_regions": summary["connected_regions"],
        "largest_region_pixels": summary["largest_region_pixels"],
        "defect_pixels_detected": summary["defect_pixels_detected"],
        "model_inference_seconds": round(prediction.inference_seconds, 6),
        "notes": [
            "Predicted area is a raster-pixel fraction, not a physical measurement.",
            "No defect severity, power loss, or per-image accuracy is inferred.",
            "Threshold changes are exploratory and are not validation results.",
        ],
    }

    st.subheader("Download results")
    download_columns = st.columns(4)
    stem = Path(prediction.file_name).stem or "solar_cell"
    outputs = (
        ("Binary mask PNG", f"{stem}_defect_mask.png", encode_png(mask_image)),
        ("Overlay PNG", f"{stem}_defect_overlay.png", encode_png(overlay_image)),
        (
            "Probability heatmap PNG",
            f"{stem}_probability_heatmap.png",
            encode_png(heatmap_image),
        ),
        (
            "Inference summary JSON",
            f"{stem}_inference_summary.json",
            json.dumps(summary_data, indent=2).encode("utf-8"),
        ),
    )
    for column, (label, file_name, contents) in zip(download_columns, outputs):
        column.download_button(
            label=label,
            data=contents,
            file_name=file_name,
            mime=(
                "application/json"
                if file_name.endswith(".json")
                else "image/png"
            ),
            width="stretch",
        )

    return mask, overlay_image


def main():
    st.markdown(
        """
        <style>
        .stApp { background: #f4f7fb; }
        [data-testid="stHeader"] { background: rgba(244,247,251,.9); }
        [data-testid="stSidebar"] { background: #eef3f8; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e1e8f0;
            padding: 14px;
            border-radius: 12px;
        }
        div.stButton > button[kind="primary"] {
            border-radius: 10px;
            min-height: 3rem;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Solar Cell Defect Detection & Localization")
    st.subheader("Deep Learning-Based Quality Inspection Using ResNet-18 U-Net")
    st.write(
        "Upload an electroluminescence (EL) image to generate a binary defect "
        "mask, an overlay, and a pixel-wise probability heatmap. Inference "
        "runs locally on CPU using the selected Phase 5 checkpoint."
    )
    st.info(
        "This application localizes patterns learned from labeled masks. "
        "It does not estimate defect severity, physical crack dimensions, "
        "power loss, or per-image accuracy."
    )

    with st.sidebar:
        st.header("Model options")
        st.write("**Default:** Phase 5 ResNet-18 U-Net")
        st.caption(
            "Validation-selected threshold: 0.165. Model weights stay on CPU."
        )
        compare_custom = st.checkbox(
            "Also compare with the original Custom U-Net",
            value=False,
            help=(
                "Loads the legacy Custom U-Net checkpoint and runs a second "
                "CPU inference. It uses its own validation-selected threshold "
                "of 0.525."
            ),
        )
        st.divider()
        st.caption(
            "The historical test-set metrics are documented in README.md; "
            "they are aggregate evaluation results, not scores for this image."
        )

    st.subheader("1. Choose an EL image")
    uploaded_file = st.file_uploader(
        "Upload a solar cell image",
        type=["png", "jpg", "jpeg"],
        help=(
            "PNG, JPG, and JPEG are supported. Images are resized to "
            "224 × 224; the input limit is 20 million pixels."
        ),
    )
    available_demos = [
        label
        for label, (file_name, _) in DEMO_IMAGES.items()
        if (_demo_directory() / file_name).is_file()
    ]
    demo_choice = st.selectbox(
        "Or load a validation-set demonstration image",
        ["Choose a demo image", *available_demos],
        help=(
            "These examples include small and difficult defects; they are "
            "demonstrations, not evaluation metrics."
        ),
    )

    try:
        image_bytes, file_name, demo_description = _source_image(
            uploaded_file,
            demo_choice,
        )
    except OSError as error:
        st.error(str(error))
        image_bytes, file_name, demo_description = None, None, None

    original_image = None
    if image_bytes:
        try:
            original_image, _ = decode_uploaded_image(image_bytes, file_name)
            st.image(
                original_image,
                caption=f"Selected image: {file_name}",
                width="stretch",
            )
            if uploaded_file is not None and demo_choice != "Choose a demo image":
                st.caption("The uploaded image takes precedence over the demo selection.")
            if demo_description:
                st.caption(f"Demonstration only — {demo_description}")
                st.caption(
                    "This example is from the project validation split; it is "
                    "not a held-out test evaluation."
                )
        except ValueError as error:
            st.error(str(error))
            image_bytes, file_name = None, None

    digest = (
        hashlib.sha256(
            image_bytes + b"\0" + file_name.encode("utf-8")
        ).hexdigest()
        if image_bytes and file_name
        else None
    )
    analyze = st.button(
        "Analyze Solar Cell",
        type="primary",
        disabled=digest is None,
        width="stretch",
    )

    if analyze and digest:
        try:
            primary_model = get_cached_model("phase5_resnet18")
            primary_result = run_inference(
                primary_model,
                image_bytes,
                file_name,
                model_key="phase5_resnet18",
                threshold=DEFAULT_THRESHOLD,
            )
            custom_result = None
            if compare_custom:
                custom_model = get_cached_model("custom_unet")
                custom_result = run_inference(
                    custom_model,
                    image_bytes,
                    file_name,
                    model_key="custom_unet",
                )
            st.session_state["phase9_analysis"] = {
                "digest": digest,
                "compare_custom": compare_custom,
                "primary": primary_result,
                "custom": custom_result,
            }
        except Exception as error:
            st.session_state.pop("phase9_analysis", None)
            st.error(
                "Analysis could not be completed. Check the checkpoint files "
                "and input image, then try again."
            )
            with st.expander("Technical details"):
                st.exception(error)

    analysis = st.session_state.get("phase9_analysis")
    if analysis and analysis["digest"] == digest:
        if analysis["compare_custom"] != compare_custom:
            st.info("Click Analyze Solar Cell again to run the selected model comparison.")
            return

        st.subheader("2. Segmentation results")
        threshold = st.slider(
            "Exploratory segmentation threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_THRESHOLD,
            step=0.005,
            format="%.3f",
            key=f"threshold-{digest[:12]}",
            help=(
                "0.165 is the validation-selected Phase 5 threshold. Changing "
                "this slider is exploratory and is not a validated improvement."
            ),
        )
        st.caption(
            "Threshold changes are exploratory only; the slider does not "
            "recalculate or claim evaluation metrics."
        )
        primary_mask, primary_overlay = _show_prediction(
            analysis["primary"],
            threshold,
        )

        if analysis["custom"] is not None:
            st.subheader("Optional model comparison")
            st.caption(
                "The Custom U-Net uses its own previously validation-selected "
                "threshold of 0.525. The slider above changes only the Phase 5 "
                "ResNet-18 result."
            )
            custom_result = analysis["custom"]
            custom_mask = custom_result.predicted_mask
            custom_overlay = create_overlay_image(
                custom_result.original_image,
                custom_mask,
            )
            compare_columns = st.columns(2)
            compare_columns[0].image(
                primary_mask * 255,
                caption=f"Phase 5 ResNet-18 mask ({threshold:.3f})",
                clamp=True,
                width="stretch",
            )
            compare_columns[1].image(
                custom_mask * 255,
                caption="Custom U-Net mask (0.525)",
                clamp=True,
                width="stretch",
            )
            compare_columns[0].image(
                primary_overlay,
                caption="Phase 5 ResNet-18 overlay",
                width="stretch",
            )
            compare_columns[1].image(
                custom_overlay,
                caption="Custom U-Net overlay",
                width="stretch",
            )
    elif analysis and digest is not None:
        st.info("The selected image changed. Click Analyze Solar Cell to update results.")
    elif digest is None:
        st.caption("Select a valid image, then analyze it to see segmentation results.")


if __name__ == "__main__":
    main()
