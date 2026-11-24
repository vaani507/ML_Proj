# Solar Cell Defect Detection & Localization

Deep learning-based quality inspection of solar-cell electroluminescence (EL)
images. The completed pipeline performs binary pixel-level defect
segmentation; it does not classify defect types or estimate defect severity.

## Interactive application

The local Streamlit app accepts PNG, JPG, and JPEG images and runs CPU inference
with the saved Phase 5 ResNet-18 U-Net. It displays the original image, binary
mask, red defect overlay, probability heatmap, and a summary of predicted
regions. The mask, overlay, heatmap, and a JSON summary can be downloaded.

The default threshold is **0.165**, selected using validation data. The
threshold slider is for exploration only; changing it does not produce a
validated performance result. An optional checkbox compares the result with
the original Custom U-Net using its separate validation-selected threshold of
0.525.

The app also offers three validation-split demonstration images when the
processed dataset is present: the smallest annotated defect, a difficult small
defect, and a difficult thin/low-contrast defect. These are demonstrations,
not independent test results.

### Run the app from a fresh clone

The instructions below use Windows PowerShell and Python 3.12 (recommended).
Clone the repository and create an isolated environment:

```powershell
git clone https://github.com/vaani507/ML_Proj.git
cd ML_Proj
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run `.\.venv\Scripts\python.exe`
instead of `python` in the commands below. If `.venv` already exists, activate
it and install the requirements without recreating the environment.

Start the app from the repository root:

```powershell
python -m streamlit run app.py
```

Streamlit prints the local URL (normally `http://localhost:8501`) after
startup. Press `Ctrl+C` in the terminal to stop it.

The trained checkpoints and dataset are intentionally excluded from Git. The
default ResNet checkpoint must be present at
`results/resnet18_unet_phase5_negw_best.pth`. For the optional comparison,
`results/unet_solar_defect.pth` must also be present. Upload-only inference
does not require the dataset; the demo selector requires the processed images
under `data/processed/SolarCells/train/images`.

Run the focused automated tests with:

```powershell
python -m unittest discover -s tests -v
```

## Dataset and preprocessing

The local dataset has paired EL images and binary masks. Its raw layout is
`data/SolarCells/{train,test}/{defect,label}`; processed pairs are stored under
`data/processed/SolarCells/{train,test}/{images,masks}`. The documented
snapshot contains 134 training pairs and 56 test pairs. The training split is
divided deterministically into 107 train and 27 validation images using seed
42. The held-out test split is kept separate.

Preprocessing corrects EXIF orientation, converts images to RGB, resizes
images to **224 × 224** with bilinear interpolation, and resizes masks with
nearest-neighbor interpolation before binarizing them. Model inputs use
ImageNet normalization: means `[0.485, 0.456, 0.406]` and standard deviations
`[0.229, 0.224, 0.225]`. The upload path follows the same RGB conversion,
224 × 224 resize, and normalization. Output probabilities are interpolated
back to the uploaded image dimensions for display and download.

To prepare data from the expected raw layout:

```powershell
python src\preprocessing.py
```

For preprocessing, provide paired images and masks in
`data/SolarCells/train/defect`, `data/SolarCells/train/label`,
`data/SolarCells/test/defect`, and `data/SolarCells/test/label`. Matching
image/mask filename stems are required; common mask suffixes such as `_mask`
and `_label` are recognized. The script writes the processed pairs to
`data/processed/SolarCells/{train,test}/{images,masks}`. The dataset itself is
not distributed in this repository.

## Models and training

`src/models.py` contains both the original Custom U-Net and the ResNet-18
encoder/U-Net decoder initialized from ImageNet-pretrained ResNet-18 weights.
The selected model is the Phase 5 ResNet-18 U-Net,
loaded from `results/resnet18_unet_phase5_negw_best.pth`. Its forward method
returns sigmoid probabilities with shape `[batch, 1, height, width]`; the
inference pipeline does not apply a second sigmoid.

The project’s training pipeline uses BCE + Dice loss, paired geometric
augmentation, and a deterministic validation split. The Phase 5 run used a
modest **1.1 negative/background BCE weight**, batch size 8, six CPU threads,
and early stopping; it trained for 8 epochs and took **195.2 seconds**. Later
Phase 6 sampling and Phase 7 boundary-loss experiments did not replace the
Phase 5 checkpoint. Phase 8 flip-based test-time augmentation did not show a
sufficiently robust validation gain, so the app uses ordinary inference.
Phase 9 final-block fine-tuning, Phase 10 post-processing, and Phase 11
continued training to a 50-epoch ceiling also did not replace Phase 5:
continued training improved overall validation Dice but regressed the tiny
image group. Those experimental checkpoints remain separate.

If you have the dataset but not the saved Phase 5 checkpoint, you can train a
compatible ResNet-18 U-Net checkpoint after preprocessing:

```powershell
python -m src.train_resnet_unet --epochs 8 --batch-size 8 --num-threads 6 --negative-bce-weight 1.1 --output-prefix results/resnet18_unet_phase5_negw
```

The command saves the best model as
`results/resnet18_unet_phase5_negw_best.pth`, the filename expected by the app.
It may download the pretrained ResNet-18 encoder weights on the first run, so
internet access is needed unless those weights are already cached. A newly
trained checkpoint is not the original evaluated checkpoint and will not
reproduce the test metrics reported below.

The selected checkpoint is frozen at
`results/resnet18_unet_phase5_negw_best.pth` (SHA-256
`518a4de1485a8828f6bbeea89da19bafe5db86ae65054e5dbc6cadbbab4a8cd8`).
On Windows it is marked read-only. The app's `load_model("phase5_resnet18")`
checks the Phase 5 training metadata, loads the weights on CPU, and enters
evaluation mode. The finalization smoke test used this app loader and an
in-memory PNG; inference returned a correctly aligned mask and probability
map at the original 320 × 240 image dimensions with threshold 0.165.

## Final evaluation results

The selected Phase 5 checkpoint was evaluated on the held-out 56-image test
set at the fixed validation-selected threshold **0.165**. No threshold tuning
or model selection was performed on the test data. The final fixed-threshold
pass reproduced the previously recorded Phase 5 results:

| Metric | Test-set result |
|---|---:|
| Dice / F1 | 0.7569 |
| IoU | 0.6261 |
| Precision | 0.7565 |
| Recall | 0.7869 |
| True-positive pixels | 148,295 |
| False-positive pixels | 33,417 |
| False-negative pixels | 34,878 |
| Micro Dice | 0.8128 |

These are dataset evaluation metrics, **not an accuracy score for an uploaded
image**. The app has no ground truth for user uploads and therefore displays
no per-image Dice, IoU, precision, recall, or accuracy. The results are from
the final aggregate evaluation and are not recomputed at app startup.

Six distinct examples (within the requested 5–10) were saved, including the
lowest-Dice case (`076.png`), highest-FP case (`026.png`),
smallest annotated defect (`129.png`), median case (`108.png`), best Dice
(`083.png`), and highest-FN case (`029.png`). The sheet shows original EL,
ground truth, binary prediction, and a TP/FP/FN overlay. Files are
`results/phase12_final_test_examples.png`,
`results/phase12_final_test_examples.csv`,
`results/phase12_final_test_per_image.csv`, and
`results/phase12_final_test_metrics.json`.

The one-time evaluation command is:

```powershell
python -m src.final_phase5_test_evaluation --batch-size 8 --num-threads 6
```

The script refuses to overwrite existing final outputs or repeat the full
test evaluation. The Phase 5 checkpoint remains read-only.

## Limitations and future work

- The model predicts the binary defect patterns represented in its training
  masks; it does not identify defect categories.
- Small and low-contrast cracks may be missed, and dark grid-like structures
  can produce false positives.
- Predicted pixel area and connected-component sizes are image-raster
  quantities, not calibrated physical crack measurements.
- The mask does not establish defect severity, cell power loss, or inspection
  acceptability.
- Threshold changes in the app are exploratory and do not alter or revalidate
  model metrics.
- The reported evaluation comes from one split and one selected run; further
  multi-seed and external-data validation would be needed before deployment.
- The default checkpoint and local dataset are not included in the repository;
  they must be supplied at the documented paths.

Potential future work includes multi-seed validation, broader external-data
testing, better grid-line discrimination, and physical calibration if
engineering measurements are required. No such work is claimed as completed.
