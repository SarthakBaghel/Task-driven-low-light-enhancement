# Low-Light Eye-State Detection for Drowsiness Monitoring

This repository is the cleaned college-submission version of the project.

The project studies one focused problem:

> how to classify eye images as `open` or `closed` reliably when lighting becomes poor.

Instead of building a full drowsiness system from scratch, the work concentrates on the eye-state detector, because that is the most important visual component for drowsiness monitoring and the part that breaks most easily in low light.

## At a Glance

- Problem studied: eye-state classification under low-light conditions
- Final backbone used: `ResNet18`
- Final best model: mixed clean + low-light detector
- Benchmark style: subject-wise split with synthetic low-light evaluation
- Main result: mixed-domain training outperformed both clean-only and low-light-only training

## 1. Project Purpose

In good lighting, an eye-state detector can usually distinguish `open` and `closed` eyes with high accuracy. In low-light scenes, however, contrast drops, eye texture becomes weaker, and a detector trained only on normal images starts missing the safety-critical `closed-eye` cases.

The goal of this project was to answer three questions:

1. How much does low light reduce the performance of a clean detector?
2. Does training only on low-light images solve the problem?
3. Is a mixed clean + low-light training strategy a better final solution?

## 2. Final Outcome

The final best model is the **mixed detector**.

It was trained on both:

- clean eye images
- synthetic low-light eye images

This model gave the best balance across both domains:

- strong performance on clean images
- strong performance on low-light images
- much better low-light robustness than the clean-only detector
- much better clean-image retention than the low-light-only detector

## 3. Main Achievement

The key finding of the project is:

> **Mixed-domain detector training is a more practical solution for low-light eye-state detection than either clean-only training or low-light-only fine-tuning.**

In the final evaluation:

- the clean detector lost performance in low light
- the low-light-only detector overfit to degraded images and performed poorly on clean images
- the mixed detector performed best overall and stayed stable across two different held-out 5k test subsets

## Research Positioning and Novelty

This project does **not** claim a new deep-learning backbone or a new low-light enhancement network. The main contribution is **methodological**.

The project is positioned relative to:

- recent eye-state / drowsiness detection literature, especially deep learning approaches using open-eye and closed-eye classification
- low-light drowsiness literature that emphasizes preprocessing or enhancement
- low-light robustness literature suggesting that domain adaptation can be more practical than enhancement alone

The main recent base paper used for positioning this work is:

- Hassan et al., 2025, *Real-time driver drowsiness detection using transformer architectures: a novel deep learning approach*

Important low-light supporting work includes:

- Alzami et al., 2024, *Time Distributed MobileNetV2 with Auto-CLAHE for Eye Region Drowsiness Detection in Low Light Conditions*

The novelty of this project is:

- a leak-free **subject-wise** eye-state benchmark
- a paired **clean-to-low-light** evaluation protocol
- a controlled comparison of three training strategies under the **same detector architecture**
- the finding that **mixed-domain training** gives the best clean / low-light trade-off

## 4. Final Results

The final comparison was repeated on two balanced held-out `5000`-image subsets of the test set (`2500 open + 2500 closed` each) using different subset seeds. The table below reports the mean and standard deviation across those two runs.

| Model | Clean F1 | Low-Light F1 | Avg F1 | Low-Light Closed Recall | Interpretation |
|---|---:|---:|---:|---:|---|
| Clean detector | `92.86 +/- 0.25` | `88.13 +/- 0.61` | `90.50 +/- 0.43` | `79.94 +/- 0.76` | Good on clean data, but misses many closed-eye cases in low light |
| Low-light-only detector | `69.87 +/- 0.11` | `89.83 +/- 0.06` | `79.85 +/- 0.08` | `99.18 +/- 0.25` | Strong on low-light data, but overfits and collapses on clean images |
| Mixed detector | `93.82 +/- 0.11` | `93.98 +/- 0.06` | `93.90 +/- 0.08` | `98.86 +/- 0.20` | Best overall model and the final recommended detector |

### Important Interpretation

- The **clean detector** is a strong baseline, but low light causes a clear drop in `F1` and a major drop in `closed-eye recall`.
- The **low-light-only detector** recovers low-light performance, but it becomes biased toward the degraded domain and performs badly on clean data.
- The **mixed detector** improves low-light performance while preserving clean performance, which is exactly the intended outcome.

Compared with the clean detector, the mixed detector improved low-light `F1` by about `+5.85` points on average while also slightly improving clean `F1`.

Selected result CSVs used for this summary are included in:

```text
results/
```

## 5. Dataset Used

The source dataset was downloaded from Kaggle:

```bash
kaggle datasets download kutaykutlu/drowsiness-detection
```

The original dataset contains eye images grouped into:

- `open_eye`
- `closed_eye`

For this project, the folder names were normalized to:

```text
open
closed
```

so the rest of the codebase could use a consistent binary classification layout.

### Final Kaggle V1 Benchmark Used in This Project

Although the original download is larger, the final benchmark used in this submission was a curated, balanced subset named:

```text
subject_class_balanced_20k
```

It contains:

- `10,000` open-eye images
- `10,000` closed-eye images

This balanced benchmark was chosen because subject distribution in the original download is highly uneven, and a careful subject-wise split is more important than simply using every image.

## 6. Why Subject-Wise Splitting Was Necessary

This was one of the most important design choices in the project.

If the same subject appears in both training and testing, the detector can partially memorize that subject's appearance, which makes the reported accuracy artificially high. To prevent this, the project used a **subject-wise split**, not a random image-level split.

From the final audited benchmark:

- usable subject IDs: `16`
- train subjects: `7`
- validation subjects: `4`
- test subjects: `5`

The final subject allocation was:

- Train: `s0002, s0003, s0005, s0007, s0009, s0010, s0020`
- Validation: `s0011, s0013, s0016, s0017`
- Test: `s0001, s0012, s0014, s0015, s0019`

### Final Clean Split Counts

| Split | Open | Closed | Total |
|---|---:|---:|---:|
| Train | `383` | `383` | `766` |
| Validation | `973` | `973` | `1946` |
| Test | `8644` | `8644` | `17288` |

The split is not image-count balanced across train/val/test because the subjects themselves are highly imbalanced. The project deliberately prioritized **leak-free subject separation** over matched split sizes.

## 7. Image Characteristics and Preprocessing

The eye images are already cropped around the eyes, but they do not all have the same size. In the audited benchmark, image sizes ranged from:

- minimum: `52 x 52`
- maximum: `209 x 209`

The most common sizes were in the low-`80s` to mid-`90s` pixels.

For detector training, all images were resized to:

```text
224 x 224
```

This matches the transfer-learning backbone requirements.

## 8. Methodology Overview

The project was implemented in phases:

1. Download and reorganize the Kaggle dataset.
2. Audit subject IDs, class counts, and image sizes.
3. Create a subject-wise train/validation/test split.
4. Train a clean baseline detector.
5. Generate a synthetic low-light version of the clean split.
6. Measure the clean detector's drop under low light.
7. Fine-tune a detector only on low-light images.
8. Compare the clean detector and low-light detector.
9. Train a mixed clean + low-light detector.
10. Compare all three models on one held-out 5k test subset.
11. Repeat the same three-model comparison on a second held-out 5k subset for stability.

## 9. What Was Actually Implemented

### 9.1 Detector Architecture

The final detectors in this project use:

- backbone: `ResNet18`
- task: binary classification (`open`, `closed`)
- input size: `224 x 224`
- detector mode: single-input `raw`
- dual-input mode: available in codebase, but not used in the final Kaggle submission result

The detector code is implemented in:

```text
src/models/detector.py
```

For the final experiments:

- the clean detector used `ImageNet` pretrained `ResNet18`
- only the last residual stage was left trainable (`resnet_trainable_layers = 1`)
- the classifier head was replaced for 2-class prediction

### 9.2 Training Procedure

The training script used for the detector experiments is:

```text
src/train_transfer_detector.py
```

It handles:

- data loading
- deterministic seeding
- transfer-learning model creation
- training and validation loops
- automatic checkpoint saving
- threshold tuning for the `closed` class
- early stopping and learning-rate scheduling

Training choices used in the final detector pipeline:

- optimizer: `AdamW`
- loss: `FocalLoss` with `gamma = 2.0`
- scheduler: `ReduceLROnPlateau`
- monitor metric: validation `F1`
- threshold objective: validation `F1`
- early stopping: enabled
- deterministic seed: `42`

### 9.3 Data Augmentation and Normalization

The transform pipeline is defined in:

```text
src/utils/classifier_transforms.py
```

It includes:

- resize to `224 x 224`
- random horizontal flip
- random rotation (`12` degrees)
- color jitter for brightness and contrast
- optional Gaussian blur
- additive Gaussian noise
- ImageNet normalization

These augmentations improve robustness and reduce overfitting on a small subject-wise training set.

### 9.4 Dataset and Dataloader Code

The dataset and dataloader code used for the detector are:

```text
src/dataset.py
src/dataloader.py
```

These files handle:

- reading `open/closed` folder structures
- keeping a stable `class_to_idx` mapping
- creating train/validation datasets
- using predefined split roots when available
- reproducible dataloader workers

## 10. Low-Light Benchmark Construction

Instead of using a separate dark-image dataset, the project created synthetic low-light versions of the clean images. This allowed direct clean-vs-low-light comparison on matched subjects and labels.

The low-light generation code is:

```text
src/generate_lowlight_dataset.py
src/low_light_simulator.py
```

The selected degradation profile for the final benchmark was:

```text
eye_mid
```

This profile was chosen after visually comparing multiple degradation settings and selecting one that was dark enough to be meaningful but still preserved some eye structure.

### Final `eye_mid` Setting

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

The final low-light datasets were created for:

- `train_clean -> train_lowlight`
- `val_clean -> val_lowlight`
- `test_clean -> test_lowlight`

## 11. Models Trained in the Final Submission

Three detector models are important in the final report.

| Model | Training Data | Initialization | Best Epoch | Training Epoch Budget | Learning Rate | Best Threshold | Purpose |
|---|---|---|---:|---:|---:|---:|---|
| Clean detector | `train_clean_subject_class_balanced_20k` | ImageNet pretrained `ResNet18` | `9` | `15` | `1e-4` | `0.70` | Baseline detector trained on clean images only |
| Low-light detector | `train_lowlight_subject_class_balanced_20k_eye_mid` | Initialized from the clean detector checkpoint | `1` | `10` | `3e-5` | `0.40` | Adaptation to low-light only |
| Mixed detector | mixed clean + low-light bundle | Initialized from the clean detector checkpoint | `4` | `12` | `2e-5` | `0.50` | Final recommended model |

Additional implementation notes:

- requested batch size was `64`
- effective batch size on Colab settled at `24`
- all checkpoints stored model config, class mapping, threshold, and metrics

### How the Models Evolved After the Baseline

This part is important for understanding the project correctly.

The later models did **not** use a new architecture. All final models used the same `ResNet18` detector family.

The progression was:

1. Train the **clean detector** on `train_clean_subject_class_balanced_20k`.
2. Save the best clean checkpoint.
3. Initialize the **low-light detector** from that clean checkpoint and fine-tune it on `train_lowlight_subject_class_balanced_20k_eye_mid`.
4. Initialize the **mixed detector** from that same clean checkpoint and train it on a mixed bundle containing both clean and low-light images.

So the main change after the baseline was not the backbone, but the **training domain**:

- clean-only
- low-light-only
- mixed clean + low-light

## 12. Why the Enhancement Path Was Dropped

An enhancement-based route was explored earlier using a Zero-DCE-style enhancer. That code is still present in the broader development repository, but it was not kept as the main Kaggle result because the enhanced eye images became visually unrealistic for this grayscale eye-crop dataset.

In practice, the enhancement output often changed the eye appearance too strongly, so the final submission focuses on **detector adaptation** rather than image enhancement.

That decision made the final project:

- simpler to explain
- easier to reproduce
- more reliable for the Kaggle eye-crop setting

## 13. Evaluation Procedure

Evaluation was performed using:

```text
src/evaluate_transfer_detector.py
```

This script:

- loads a saved checkpoint
- evaluates on clean and optionally low-light roots
- exports metrics to CSV
- supports balanced subset evaluation through:
  - `--max-total-samples`
  - `--subset-seed`
- writes a `subset_manifest.csv` so repeated model comparisons use the exact same subset

The main reported metrics are:

- accuracy
- precision
- recall
- F1
- closed-eye recall

The repeated 5k evaluations were important because they showed the mixed detector's result was not a one-off random subset effect.

## 14. Code Files Used in This Submission

### Core Training and Evaluation Files

- `src/train_transfer_detector.py`
  - main training script for clean, low-light-only, and mixed detectors
- `src/evaluate_transfer_detector.py`
  - main evaluation script for clean-vs-low-light and final three-model comparison
- `src/generate_lowlight_dataset.py`
  - generates synthetic low-light splits
- `src/low_light_simulator.py`
  - defines the degradation operations and presets

### Model and Learning Components

- `src/models/detector.py`
  - detector architecture factory and `ResNet18` setup
- `src/models/baseline_cnn.py`
  - lightweight baseline CNN support
- `src/losses/focal_loss.py`
  - focal loss used during training

### Data and Utility Files

- `src/dataset.py`
  - eye-state dataset reader
- `src/dataloader.py`
  - dataloader construction and reproducible split handling
- `src/utils/classifier_transforms.py`
  - augmentations and normalization
- `src/utils/classifier_metrics.py`
  - metrics, threshold selection, and reporting helpers
- `src/utils/colab_runtime.py`
  - Colab/Drive path handling and checkpoint copying helpers

### Demo File

- `demo/run_demo.py`
  - simple clean-detector vs mixed-detector comparison on sample images

## 15. Notebooks Included

The experiment notebooks are included in:

```text
notebooks/kaggle_v1/
```

They document the project phase-by-phase:

- `phase0.ipynb` - dataset download and organization
- `phase1.ipynb` - dataset audit
- `phase2.ipynb` - subject-wise split creation
- `phase3.ipynb` - clean pilot training
- `phase4.ipynb` - full clean detector training
- `phase5.ipynb` - low-light generation
- `phase6.ipynb` - clean vs low-light baseline
- `phase7_updated.ipynb` - low-light detector training
- `phase8.ipynb` - clean vs low-light detector comparison
- `phase9.ipynb` - mixed detector training
- `phase10.ipynb` - first final 5k comparison
- `phase11.ipynb` - second final 5k comparison

## 16. Submission Folder Structure

```text
lowlight-eye-state-detection-submission/
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/
│   └── README.md
├── demo/
│   ├── README.md
│   ├── run_demo.py
│   └── sample_images/
├── docs/
│   ├── IMPLEMENTATION_DETAILS.md
│   └── GITHUB_SETUP.md
├── notebooks/
│   ├── README.md
│   ├── kaggle_v1/
│   └── reference/
├── paper/
│   ├── ieee_draft.tex
│   ├── EXAMINER_SCRIPT.md
│   └── figures/
├── results/
│   ├── README.md
│   ├── clean_vs_lowlight_baseline_5k.csv
│   ├── final_comparison_seed42.csv
│   └── final_comparison_seed314.csv
└── src/
    ├── dataset.py
    ├── dataloader.py
    ├── train_transfer_detector.py
    ├── evaluate_transfer_detector.py
    ├── generate_lowlight_dataset.py
    ├── low_light_simulator.py
    ├── inference_enhancer.py
    ├── losses/
    ├── models/
    └── utils/
```

## 17. What to Open First

If an examiner or reviewer opens this repository for the first time, the recommended reading order is:

1. `README.md` - project overview and final result
2. `results/` - final exported tables used in the report
3. `paper/` - draft paper, figures, and viva script
4. `demo/` - simple demonstration files
5. `src/` - core implementation files
6. `notebooks/kaggle_v1/` - full experimental workflow

## 18. How to Run the Demo

The repository already includes a curated set of demo images in:

```text
demo/sample_images/
```

So you only need to install the environment and place the checkpoints.

### Step 1: Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install requirements

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Step 3: Place the final checkpoints

Recommended file names:

```text
checkpoints/clean_detector_best.pt
checkpoints/mixed_detector_best.pt
checkpoints/lowlight_detector_best.pt
```

### Step 4: Run the demo

```bash
python3 demo/run_demo.py \
  --clean-checkpoint checkpoints/clean_detector_best.pt \
  --mixed-checkpoint checkpoints/mixed_detector_best.pt \
  --image-dir demo/sample_images \
  --output-dir demo_outputs
```

The demo creates:

```text
demo_outputs/predictions.csv
demo_outputs/demo_contact_sheet.png
demo_outputs/demo_summary.txt
```

For a viva or project presentation, the most useful file to show is:

```text
demo_outputs/demo_contact_sheet.png
```

It displays the same sample images with predictions from the clean detector and the mixed detector side by side.

The current curated demo set contains low-light samples selected from the final evaluation workflow so that the difference between the clean detector and mixed detector is easy to explain.

## 19. How to Provide the Checkpoints

Large model checkpoints should not normally be committed directly into a standard GitHub repository.

Recommended options:

### Option 1: Google Drive

Upload the final checkpoints to Google Drive and place the shared link in:

```text
checkpoints/README.md
```

### Option 2: ZIP Submission

If a ZIP file is allowed:

1. keep the GitHub repository lightweight,
2. copy the checkpoints into `checkpoints/`,
3. zip the folder locally,
4. submit the ZIP and the GitHub link together.

### Option 3: GitHub Releases or Git LFS

These are also valid, but for a college submission, `GitHub repo + Drive link` is usually the easiest.

## 20. What This Project Successfully Demonstrates

This project demonstrates that:

- low light significantly harms a clean eye-state detector
- low-light-only fine-tuning is not enough because it damages clean-domain performance
- mixed-domain training gives the most balanced and practical detector
- careful experimental design, especially subject-wise splitting, matters as much as model architecture

## 21. Final One-Line Summary

This project shows that **a mixed clean + synthetic low-light training strategy can produce a robust and practical eye-state detector for low-light drowsiness monitoring, outperforming both clean-only and low-light-only alternatives**.
