# Low-Light Eye-State Detection for Drowsiness Monitoring

This project studies how low-light conditions affect eye-state detection and how detector training can be adapted to improve robustness. The task is binary classification:

- `open` eye
- `closed` eye

The final goal is useful for driver-drowsiness or fatigue-monitoring systems, where correctly detecting closed eyes in poor lighting is safety-critical.

## Project Purpose

Normal eye-state detectors can work well on clean eye images but lose recall when the same images become dark or noisy. In this project, we tested whether low-light robustness is better improved by:

1. training only on clean images,
2. fine-tuning only on low-light images, or
3. fine-tuning on a mixed clean + low-light dataset.

The final result shows that mixed-domain training gives the best overall detector. It improves low-light performance while preserving clean-image performance.

## Final Outcome

The best final model is the **mixed detector**, a ResNet18-based eye-state classifier fine-tuned using both clean and synthetic low-light eye crops.

Across two balanced held-out 5k test subsets, the mixed detector was consistently the strongest model.

| Model | Clean F1 | Low-Light F1 | Average F1 | Main Observation |
|---|---:|---:|---:|---|
| Clean detector | 92.86 +/- 0.25 | 88.13 +/- 0.61 | 90.50 +/- 0.43 | Strong on clean images, loses closed-eye recall in low light |
| Low-light-only detector | 69.87 +/- 0.11 | 89.83 +/- 0.06 | 79.85 +/- 0.08 | Overfits to low-light and performs poorly on clean images |
| Mixed detector | 93.82 +/- 0.11 | 93.98 +/- 0.06 | 93.90 +/- 0.08 | Best clean/low-light tradeoff |

The important low-light closed-eye recall result is:

| Model | Low-Light Closed-Eye Recall |
|---|---:|
| Clean detector | 79.94 +/- 0.76 |
| Low-light-only detector | 99.18 +/- 0.25 |
| Mixed detector | 98.86 +/- 0.20 |

The low-light-only detector gets high closed-eye recall, but it collapses on clean images. The mixed detector keeps high low-light recall while also keeping clean performance strong.

## Main Conclusion

The project conclusion is:

- low light consistently hurts a clean-only detector,
- low-light-only fine-tuning overfits to the degraded domain,
- mixed clean + low-light training gives the best practical detector.

For this dataset, the image-enhancement path was not used as the final solution because the Zero-DCE-style enhancer changed grayscale eye crops too aggressively. The final approach is detector adaptation, not image enhancement.

## Dataset

The final experiments use the Kaggle dataset:

```bash
kaggle datasets download kutaykutlu/drowsiness-detection
```

The dataset already contains eye crops grouped by eye state. It was reorganized into:

```text
kaggle_v1/raw_clean/open
kaggle_v1/raw_clean/closed
```

The final pipeline uses subject-wise splitting, not random image-level splitting. This avoids leakage where images from the same subject appear in both training and testing.

Final clean split roots used in Colab:

```text
/content/drive/MyDrive/task_driven_video_pipeline/kaggle_v1/train_clean_subject_class_balanced_20k
/content/drive/MyDrive/task_driven_video_pipeline/kaggle_v1/val_clean_subject_class_balanced_20k
/content/drive/MyDrive/task_driven_video_pipeline/kaggle_v1/test_clean_subject_class_balanced_20k
```

Final low-light split roots:

```text
/content/drive/MyDrive/task_driven_video_pipeline/kaggle_v1/train_lowlight_subject_class_balanced_20k_eye_mid
/content/drive/MyDrive/task_driven_video_pipeline/kaggle_v1/val_lowlight_subject_class_balanced_20k_eye_mid
/content/drive/MyDrive/task_driven_video_pipeline/kaggle_v1/test_lowlight_subject_class_balanced_20k_eye_mid
```

The low-light benchmark uses the custom `eye_mid` degradation preset:

```text
gamma = 1.75
brightness_factor = 0.72
contrast_factor = 0.84
black_level_shift = 0.02
gaussian_sigma = 1.5
poisson_strength = 0.08
motion_blur_kernel = 0
desaturation_factor = 0.00
```

## Methodology

The project was implemented in phases.

| Phase | Purpose |
|---|---|
| Phase 0 | Download and canonicalize Kaggle dataset into `open` and `closed` folders |
| Phase 1 | Audit dataset subjects, class counts, and image sizes |
| Phase 2 | Build subject-wise train/validation/test split |
| Phase 3 | Run a small pilot clean-detector training job |
| Phase 4 | Train the full clean baseline detector |
| Phase 5 | Generate synthetic low-light datasets using `eye_mid` |
| Phase 6 | Measure clean detector degradation from clean to low light |
| Phase 7 | Fine-tune a low-light-only detector |
| Phase 8 | Compare clean detector vs low-light-only detector |
| Phase 9 | Train mixed clean + low-light detector |
| Phase 10 | Compare all three models on a 5k held-out subset |
| Phase 11 | Repeat comparison on another 5k held-out subset for stability |

The experiment notebooks are stored in:

```text
notebooks/kaggle_v1/
```

The original detailed development log is preserved in:

```text
docs/DEVELOPMENT_LOG.md
```

## Models

The final detector family is based on ResNet18 transfer learning. The main checkpoints used in the final comparison were:

```text
clean detector:
/content/drive/MyDrive/task_driven_checkpoints/kaggle_v1/full_clean_detector_subject_class_balanced_20k_resnet18/kaggle_v1_clean_full_subject_class_balanced_20k_best.pt

low-light-only detector:
/content/drive/MyDrive/task_driven_checkpoints/kaggle_v1/detector_lowlight_subject_class_balanced_20k_eye_mid_resnet18/detector_lowlight_best.pt

mixed detector:
/content/drive/MyDrive/task_driven_checkpoints/kaggle_v1/detector_mixed_subject_class_balanced_20k_eye_mid_resnet18/detector_mixed_best.pt
```

Large checkpoint files are not stored in this GitHub repository. They should be shared separately through Google Drive, GitHub Releases, or another file-sharing link.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── dataset.py
├── dataloader.py
├── train_transfer_detector.py
├── evaluate_transfer_detector.py
├── generate_lowlight_dataset.py
├── analyze_detector_recoveries.py
├── demo/
│   ├── README.md
│   ├── run_demo.py
│   └── sample_images/
├── docs/
│   └── DEVELOPMENT_LOG.md
├── notebooks/
│   ├── train_transfer_detector_colab.ipynb
│   └── kaggle_v1/
├── models/
├── losses/
├── utils/
├── configs/
└── tests/
```

## Important Scripts

| File | Purpose |
|---|---|
| `dataset.py` | Folder-based eye-state dataset loader |
| `train_transfer_detector.py` | Main ResNet18 detector training script |
| `generate_lowlight_dataset.py` | Generates low-light versions of clean splits |
| `evaluate_transfer_detector.py` | Evaluates detector on clean and low-light roots |
| `analyze_detector_recoveries.py` | Finds cases where mixed detector recovers clean-detector low-light mistakes |
| `demo/run_demo.py` | Simple examiner/demo prediction script |

## Quick Demo

The easiest demo is to show the clean detector and mixed detector on low-light samples.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Put demo images in a folder

You can use either a flat folder:

```text
demo/sample_images/image1.png
demo/sample_images/image2.png
```

or class folders:

```text
demo/sample_images/open/*.png
demo/sample_images/closed/*.png
```

Class folders are better because the demo can show whether each model is correct.

### 3. Run the demo

```bash
python demo/run_demo.py \
  --clean-checkpoint /path/to/clean_detector_best.pt \
  --mixed-checkpoint /path/to/detector_mixed_best.pt \
  --image-dir demo/sample_images \
  --output-dir demo_outputs
```

The demo writes:

```text
demo_outputs/predictions.csv
demo_outputs/demo_contact_sheet.png
demo_outputs/demo_summary.txt
```

The contact sheet is the simplest file to show during the college demo.

## Recovery Example Generation

To show examples where the clean detector failed but the mixed detector recovered the correct prediction, use:

```bash
python analyze_detector_recoveries.py \
  /path/to/clean_detector_best.pt \
  /path/to/detector_mixed_best.pt \
  /path/to/test_lowlight_subject_class_balanced_20k_eye_mid \
  --clean-root /path/to/test_clean_subject_class_balanced_20k \
  --subset-manifest /path/to/subset_manifest.csv \
  --focus-label closed \
  --top-k 12 \
  --output-dir recovered_examples
```

This creates:

```text
recovered_examples/all_recovered_cases.csv
recovered_examples/top_recovered_cases.csv
recovered_examples/top_recovered_contact_sheet.png
recovered_examples/top_case_figures/
```

These examples are useful for explaining the practical improvement visually.

## How to Reproduce the Final Results

Open the notebooks in this order:

```text
notebooks/kaggle_v1/phase0.ipynb
notebooks/kaggle_v1/phase1.ipynb
notebooks/kaggle_v1/phase2.ipynb
notebooks/kaggle_v1/phase3.ipynb
notebooks/kaggle_v1/phase4.ipynb
notebooks/kaggle_v1/phase5.ipynb
notebooks/kaggle_v1/phase6.ipynb
notebooks/kaggle_v1/phase7_updated.ipynb
notebooks/kaggle_v1/phase8.ipynb
notebooks/kaggle_v1/phase9.ipynb
notebooks/kaggle_v1/phase10.ipynb
notebooks/kaggle_v1/phase11.ipynb
```

Notes:

- `phase7.ipynb` contains the discarded enhancer experiment.
- `phase7_updated.ipynb` contains the detector-only low-light fine-tuning path.
- `phase10.ipynb` and `phase11.ipynb` contain the final two-subset comparison.

## What Must Be Provided Separately

The GitHub repository contains code, notebooks, and documentation. It should not contain the full dataset or large checkpoints.

For a complete college submission/demo, provide:

1. GitHub repository link.
2. Trained clean detector checkpoint.
3. Trained mixed detector checkpoint.
4. Optional low-light-only checkpoint for comparison.
5. A small folder of demo images, preferably with `open/` and `closed/` subfolders.
6. Optional Google Drive link to final result CSVs and contact sheets.

## Limitations

- The final published-style result is based on two balanced 5k held-out subsets, not the full 17,288-image test root.
- The low-light data is synthetically generated from clean eye crops.
- The final approach is optimized for eye-crop images, not full-face video frames.
- Large model checkpoints are not included in the repository.

## Future Work

Possible next steps:

- run the final comparison on the full held-out test split,
- test on real night-driving eye images,
- export the mixed detector to ONNX for deployment,
- build a small webcam demo around the trained detector,
- evaluate additional lighting conditions and camera noise profiles.

## Summary

This project demonstrates that low-light eye-state detection can be improved without relying on image enhancement. A clean-only detector loses closed-eye recall under low light. A low-light-only detector overfits. A mixed clean + low-light detector gives the strongest and most stable performance across clean and degraded images.
