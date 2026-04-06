# Task-Driven Low-Light Enhancement

This repository currently contains two connected tracks:

1. A binary eye-state classification pipeline for `open` vs `closed`.
2. A Zero-DCE style low-light enhancement pipeline.

The project starts from raw video files, prepares an eye-state image dataset, trains a lightweight CNN baseline, evaluates the model on clean and degraded low-light images, and also includes a modular low-light enhancement model plus its training losses.

## Current Status

Implemented and smoke-tested:

- Raw video frame extraction
- Eye-state auto-labeling
- Dataset-quality auditing for label noise, crop quality, and sequence instability
- Balanced subset creation for fast Colab experiments
- Balanced train/validation dataset creation
- Dataset verification and visualization
- Lightweight CNN baseline model
- Configurable detector module with custom CNN, MobileNetV2, and ResNet18 options
- Transfer-learning detector training and evaluation pipeline with focal loss and threshold tuning
- Colab-friendly transfer-detector training flow with GPU auto-detection, adaptive batch sizing, tqdm progress bars, reproducible seeding, Google Drive checkpoint backups, and runtime-aware path handling
- Report-ready experiment comparison generation with CSV, Markdown, high-resolution tables, confusion matrices, training curves, and summary exports
- Joint enhancer-detector pipeline model
- Joint training loss and experiment config for end-to-end learning
- Full joint training and validation loop with checkpointing
- Baseline training loop
- Clean vs low-light evaluation with CSV and plots
- Zero-DCE style enhancement model
- Standalone Zero-DCE enhancer training loop with scheduler, early stopping, AMP support, checkpointing, and report-history export
- Zero-DCE inference and visualization
- Zero-DCE enhancement losses and tests
- Reusable enhancer + frozen detector evaluation pipeline for testing whether enhancement alone improves low-light detection without updating the detector
- The detector module now includes an explicit `freeze_model()` helper, `freeze_detector=True` builder option, frozen-layer printing, and gradient-state debug utilities so frozen inference can be verified rather than assumed

Latest smoke validation on the provided dataset slice:

- Source subset: `data/Fold3_part2/31`
- Extracted frames: `1822`
- Auto-labeled frames: `674 open`, `1148 closed`
- Balanced dataset used: `1348` images
- Train split: `1078`
- Validation split: `270`
- Baseline clean validation accuracy after 1 smoke epoch: `59.63%`
- Baseline mild low-light validation accuracy after 1 smoke epoch: `55.93%`
- Historical first severe low-light validation accuracy: `50.37%`
- Intermediate adjusted severe low-light validation accuracy: `52.59%`
- Current trainable-severe low-light validation accuracy: `52.59%`
- Baseline clean validation F1: `37.71%`
- Historical first severe low-light F1: `19.28%`
- Intermediate adjusted severe low-light F1: `23.81%`
- Current trainable-severe low-light F1: `32.63%`
- Mean clean validation brightness: `0.3294`
- Mean mild low-light brightness: `0.2028`
- Historical first severe brightness: `0.1530`
- Intermediate adjusted severe brightness: `0.1714`
- Current trainable-severe brightness: `0.1825`
- Detector smoke tests: passed for both `custom` and `mobilenetv2` backbones on dummy `224x224` RGB inputs
- Joint model smoke tests: passed for end-to-end `input -> enhancer -> detector -> logits`, including gradient flow from detector loss back into the enhancer
- Joint loss smoke check: passed for `L_total = L_detection + lambda * L_enhancement`, with separate logging for detection and enhancement terms and config-driven lambda tuning
- Joint training smoke run: completed 1 CPU epoch on a tiny synthetic low-light dataset, wrote `last` and `best` checkpoints, and `validate_joint.py` successfully reloaded the saved best checkpoint
- Joint enhancer-detector real-data smoke check: passed on a severe low-light mini subset derived from the real dataset slice, including training, checkpoint saving, and standalone validation reload
- Severe low-light preset review: the first version was too close to near-black on some frames, so the current `severe` profile is now tuned for joint training, staying difficult for the detector while preserving enough structure for enhancement-driven optimization; `extreme` remains the stress-test option
- Refreshed low-light sample previews were written to `artifacts/smoke_check/sample_lowlight_images_adjusted_v2`
- The canonical severe artifacts under `artifacts/smoke_check/lowlight_val_31_severe` and `artifacts/smoke_check/joint_lowlight_dataset_31_severe` were regenerated with the new trainable-severe profile. The older darker versions were preserved as `*_legacy_dark` backups.
- The latest clean-vs-current-severe detector comparison report was written to `artifacts/smoke_check/evaluation_report_31_severe_current`
- Dataset audit on `artifacts/smoke_check/labeled_31` flagged `272` low-quality samples and `216` short-run sequence-noise candidates, which supports the hypothesis that the current baseline is being hurt by noisy frame-level labels and weak eye crops
- Transfer-detector smoke training on a tiny real mini split completed with a ResNet18 backbone, focal loss, threshold tuning, and low-light validation. The mini validation run reached `87.50%` accuracy and `81.25%` closed-eye recall on clean validation, while the matching low-light mini validation remained much harder at `40.62%` accuracy and `50.00%` closed-eye recall. This is only a smoke check, not a full-dataset performance claim.
- A standalone transfer-detector evaluation script is included for report-ready clean vs low-light comparison, and checkpoint loading has been patched for PyTorch 2.6+ `weights_only=True` compatibility changes
- The baseline evaluation script now defaults to `num_workers=0` for better compatibility in restricted or sandboxed environments where PyTorch shared-memory workers may be unavailable
- The transfer-detector trainer now includes a Colab notebook starter, `requirements.txt`, Drive-mount support, runtime-aware relative paths for local-vs-Colab switching, and automatic batch-size fallback on memory pressure
- A balanced subsetting utility now supports fixed-count or percentage-based sampling, preserves dataset folder structure, writes a CSV manifest, and prints final per-class counts for quick Colab-sized experiments
- Baseline and joint training scripts now also export per-epoch history JSON files, and a report-generation script can combine baseline, low-light, enhancer-only, and joint metrics into report-ready tables and figures under a `results/` folder
- A standalone `train_enhancer.py` script is now included so the missing `enhancer only` experiment can be trained directly on low-light images and plugged into the final report pipeline
- `train_enhancer.py` has been smoke-tested on `artifacts/smoke_check/joint_lowlight_dataset_31_severe_mini`: it wrote compatible `enhancer_best.pt`, `enhancer_last.pt`, and `enhancer_training_history.json` files under `artifacts/smoke_check/enhancer_smoke`, and the saved checkpoint was verified through both `inference_enhancer.py` and `generate_report_results.py`
- The report generator now handles enhancer-only training histories cleanly even when they contain loss curves but no classifier accuracy/F1 metrics
- The transfer-detector evaluation pipeline now also exports a compact report-table CSV named `experiment_results.csv` with `Dataset / Accuracy / Precision / Recall / F1`, alongside annotated clean-vs-low-light degradation plots
- A dedicated `evaluate_enhancer_frozen_detector.py` script now compares raw low-light detection against `enhancer -> frozen clean-trained detector`, while explicitly freezing detector weights and keeping the detector in eval mode
- The frozen-detector evaluation now prints and verifies that detector parameters stay `requires_grad=False`, detector mode stays `eval`, and no stored detector gradients appear before or after inference
- The transfer-detector trainer now also supports training the detector on enhancer outputs using a frozen pretrained enhancer plus a clean-trained detector checkpoint as initialization, with the default enhanced-model output path `checkpoints/detector_enhanced.pth`
- The enhancer-detector comparison evaluation now supports three-way low-light comparison: raw low-light, enhancer + original detector, and enhancer + fine-tuned detector, with threshold sweeps over configurable candidate thresholds such as `0.3, 0.4, 0.5, 0.6`
- The enhancer + frozen detector pipeline has been smoke-tested on the real validation split. The freeze/eval guarantees passed, the report artifacts were written under `artifacts/smoke_check/enhancer_frozen_detector_eval_smoke`, and the current smoke-trained enhancer checkpoint did not improve the frozen detector yet, which is the correct outcome to report at this stage
- Enhanced-image detector fine-tuning has now been smoke-tested too: `train_transfer_detector.py` successfully trained a detector on frozen-enhancer outputs and saved `detector_enhanced.pth`, while `evaluate_enhancer_frozen_detector.py` produced the full three-way comparison plus `threshold_sweep.csv` under `artifacts/smoke_check/enhanced_detector_comparison_smoke`
- A larger real-data verification pass has now also been completed on the full prepared severe split (`1078` train / `270` val). The enhanced-image fine-tuning path ran end to end, saved checkpoints under `artifacts/smoke_check/detector_enhanced_fullslice`, and the matching three-way evaluation artifacts were written under `artifacts/smoke_check/enhanced_detector_comparison_fullslice`
- Threshold tuning now includes a prediction-rate guardrail so report scripts do not automatically pick collapsed operating points that predict almost everything as `closed`. By default, tuned thresholds are kept within a predicted-closed rate of `5%` to `95%` whenever a non-collapsed option exists
- After that fix, the larger three-way evaluation no longer collapses to threshold `0.30`: raw low-light now selects threshold `0.40` (`48.52%` accuracy, `61.06%` F1), enhancer + original detector selects threshold `0.40` (`47.78%` accuracy, `55.52%` F1), and enhancer + fine-tuned detector selects threshold `0.50` (`63.70%` accuracy, `62.88%` F1)
- The detector module now also supports dual-input classification, where raw low-light and enhanced images are passed through a shared or separate backbone, the extracted features are concatenated, and a shared classifier head predicts `open` vs `closed`
- `train_transfer_detector.py` now supports `--detector-input-mode raw|enhanced|dual`, can initialize a dual-input detector from a single-input checkpoint, and writes dual-input smoke checkpoints under `artifacts/smoke_check/detector_dual_smoke`
- `evaluate_enhancer_frozen_detector.py` now supports a `--dual-detector-checkpoint` path and can compare four branches on the same validation split: raw only, enhancer + original detector, enhancer + fine-tuned detector, and dual input detector
- The latest dual-input smoke comparison was written under `artifacts/smoke_check/dual_input_comparison_smoke`. On the current validation slice, raw low-light reached `48.52%` accuracy / `61.06%` F1, enhancer + original detector reached `47.78%` / `55.52%`, enhancer + fine-tuned detector reached `50.00%` / `65.99%`, and the dual-input detector reached `48.52%` / `62.53%`
- A dedicated `analyze_enhancement_recoveries.py` script now extracts report-ready examples where raw low-light detection fails but enhancement recovers the sample. On the current larger validation slice, it found `19` recoveries for enhancer + original detector, `101` recoveries for enhancer + fine-tuned detector, and `17` samples recovered by both paths. The CSVs, summary text, contact sheets, and detailed comparison figures were written under `artifacts/smoke_check/recovered_detection_cases_fullslice`
- A root `.gitignore` is now included for Python caches, notebook checkpoints, generated artifacts, checkpoints, local environment files, and dataset folders such as `data/`, `dataset/`, and `datasets/`

Smoke-check artifacts were written under [`artifacts/smoke_check`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/artifacts/smoke_check).

## Project Flow

The current pipeline is:

1. Raw videos are stored under folders like `data/Fold3_part2/<subject>/<condition>.mp4`.
2. [`extract_frames.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/extract_frames.py) samples frames from videos and saves cropped eye or upper-face regions.
3. [`label_eye_state.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/label_eye_state.py) labels the extracted frames as `open` or `closed`.
4. [`audit_dataset_quality.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/audit_dataset_quality.py) audits the labeled frame dataset for short label flips, low-quality crops, suspicious samples, and per-image statistics before model training.
5. [`create_balanced_subset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/create_balanced_subset.py) creates a smaller balanced dataset subset by fixed count or percentage per class, preserves folder structure, copies only selected files, and saves a CSV manifest.
6. [`prepare_dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/prepare_dataset.py) balances the classes and builds `train/` and `val/` folders.
7. [`verify_dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/verify_dataset.py) checks folder structure, image validity, class balance, and saves a report.
8. [`train_baseline.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_baseline.py) trains the baseline CNN classifier and now also exports epoch-wise history JSON for report plots.
9. [`train_transfer_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_transfer_detector.py) trains a stronger transfer-learning detector with augmentation, focal loss, early stopping, scheduler support, threshold tuning for the `closed` class, and Colab-friendly runtime features.
10. [`generate_lowlight_dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/generate_lowlight_dataset.py) creates a degraded low-light version of a dataset split.
11. [`evaluate.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/evaluate.py) compares the trained baseline classifier on clean vs low-light datasets and saves report-ready outputs.
12. [`evaluate_transfer_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/evaluate_transfer_detector.py) evaluates the stronger transfer-learning detector on clean and low-light datasets using a tuned closed-eye threshold.
13. [`models/zerodce.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/zerodce.py) provides a Zero-DCE style enhancement network for low-light image enhancement.
14. [`train_enhancer.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_enhancer.py) trains the enhancer alone on low-light images using the Zero-DCE losses, even when the dataset is stored in `open/closed` folders. It saves `enhancer_best.pt`, `enhancer_last.pt`, and epoch-wise history JSON for report plots.
15. [`inference_enhancer.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/inference_enhancer.py) and [`visualize_enhancement.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/visualize_enhancement.py) run the enhancer on images and save visual comparisons.
16. [`models/frozen_detector_pipeline.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/frozen_detector_pipeline.py) provides a reusable `low-light -> enhancer -> frozen detector` pipeline that keeps detector weights frozen and forces detector eval mode.
17. [`evaluate_enhancer_frozen_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/evaluate_enhancer_frozen_detector.py) evaluates whether enhancement alone improves low-light detection by comparing raw low-light detector performance against the enhancer-preprocessed version with the same clean-trained detector.
18. [`analyze_enhancement_recoveries.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/analyze_enhancement_recoveries.py) finds images that the raw low-light detector misses but an enhancement-based path gets right, then saves CSVs and detailed side-by-side visual comparisons for report use.
19. [`models/joint_model.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/joint_model.py) connects the enhancer and detector into a single differentiable pipeline.
20. [`train_joint.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_joint.py) trains the full enhancer-detector pipeline on low-light images and now also exports epoch-wise history JSON for report plots.
21. [`validate_joint.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/validate_joint.py) validates saved joint checkpoints and also provides the reusable validation loop used during training.
22. [`losses/enhancement_losses.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/enhancement_losses.py) provides the key Zero-DCE loss functions used during enhancement training.
23. [`losses/focal_loss.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/focal_loss.py) provides focal loss for imbalance-aware transfer-learning detector training.
24. [`losses/joint_loss.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/joint_loss.py) combines detection and enhancement objectives for end-to-end training.
25. [`configs/train_config.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/configs/train_config.py) stores tunable joint-training hyperparameters, including the enhancement lambda.
26. [`generate_report_results.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/generate_report_results.py) generates project-report tables and figures by evaluating baseline, low-light, enhancer-only, and joint-model checkpoints and exporting results into a `results/` folder.
27. [`notebooks/train_transfer_detector_colab.ipynb`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/notebooks/train_transfer_detector_colab.ipynb) provides a short Colab notebook with the setup cells for Drive mounting, dependency installation, path configuration, training, and evaluation.
28. [`requirements.txt`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/requirements.txt) lists the core packages needed to run the project locally or in Colab.

## Repository Structure

### Dataset and Dataloading

- [`dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/dataset.py)
  - Provides `EyeStateDataset`.
  - Supports either flat `open/closed` folders or split datasets like `train/open`, `train/closed`, `val/open`, `val/closed`.
  - Applies image resizing and ImageNet normalization.

- [`dataloader.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/dataloader.py)
  - Builds train and validation datasets.
  - Supports predefined `train/val` layout or automatic stratified splitting.
  - Seeds DataLoader workers for more reproducible transform sampling.
  - Returns PyTorch `DataLoader` objects in a `DataBundle`.

- [`utils.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/utils.py)
  - Dataset summary helpers.
  - Class distribution printing.
  - Visualization helpers for random samples.

### Dataset Preparation

- [`extract_frames.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/extract_frames.py)
  - Reads videos from a folder tree.
  - Samples frames by interval in frames or seconds.
  - Uses OpenCV Haar detection to focus on face, upper-face, or eyes.
  - Avoids saving nearly identical frames with pixel or SSIM-based change detection.
  - Best suited as the first step after raw video collection.

- [`label_eye_state.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/label_eye_state.py)
  - Auto-labels images as `open` or `closed`.
  - Supports three backends:
    - `mediapipe-solutions`: classic Face Mesh API when available
    - `mediapipe-tasks`: newer Face Landmarker API when a `.task` model is provided
    - `haar`: OpenCV fallback that works on cropped upper-face images
  - In `auto` mode it tries MediaPipe first, then falls back safely.
  - MediaPipe backends use EAR-based labeling.
  - Haar fallback uses eye detections and a configurable threshold on detected eyes.

- [`prepare_dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/prepare_dataset.py)
  - Collects labeled images.
  - Optionally balances class counts by downsampling the larger class.
  - Builds `train/` and `val/` folders.
  - Copies or moves files into a clean training dataset layout.

- [`verify_dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/verify_dataset.py)
  - Validates image integrity.
  - Detects flat vs split layout.
  - Flags folder naming issues.
  - Reports class imbalance.
  - Saves a text report and a class-count bar chart.

- [`audit_dataset_quality.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/audit_dataset_quality.py)
  - Audits a labeled frame dataset before training.
  - Computes brightness, contrast, sharpness, entropy, and Haar eye-count statistics per image.
  - Flags very short open/closed label runs that likely come from noisy frame-level labeling.
  - Flags low-quality crops using percentile-based thresholds.
  - Writes CSV tables, preview grids, and a text report for manual sanity checking.

- [`create_balanced_subset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/create_balanced_subset.py)
  - Creates a smaller balanced copy of a dataset for quick experiments or Colab runs.
  - Supports fixed-count sampling with `--samples-per-class`.
  - Supports percentage-based sampling with `--percent-per-class`.
  - For percentage mode, keeps the subset balanced by using the smallest requested per-class count across the available classes in each group.
  - Preserves the original folder structure, including `train/val` splits and nested sequence folders.
  - Writes a CSV manifest of selected source and destination files.

### Baseline Eye-State Classification

- [`models/baseline_cnn.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/baseline_cnn.py)
  - A lightweight CNN for binary classification.
  - Uses 3 convolution blocks.
  - Each block is `Conv2d -> BatchNorm -> ReLU -> MaxPool`.
  - Uses adaptive pooling and dropout before the final classifier.

- [`models/detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/detector.py)
  - Unified detector builder for binary eye-state classification.
  - Supports `custom`, `mobilenetv2`, and `resnet18` backbones in the same module.
  - MobileNetV2 and ResNet18 use pretrained ImageNet weights when available.
  - Early transfer-learning layers are frozen and only the last feature stages plus the classifier are fine-tuned.
  - Supports optional dual-input mode with either a shared backbone or separate raw/enhanced backbones.
  - In dual-input mode, raw low-light and enhanced images each have shape `(B, 3, 224, 224)`, their features are extracted independently, concatenated, and passed to a 2-class classifier head.
  - Also provides:
    - `freeze_model(model)`
    - `freeze_detector=True` in `build_detector(...)`
    - frozen-layer printing
    - gradient-state debug helpers for verifying truly frozen inference
  - Includes a compact model-summary printer and a dummy forward-pass helper for `224x224` RGB inputs.

- [`losses/focal_loss.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/focal_loss.py)
  - Implements multi-class focal loss for harder-example upweighting.
  - Supports class-wise `alpha`, configurable `gamma`, and optional label smoothing.
  - Used to counter the baseline detector's strong bias toward predicting `open`.

- [`utils/classifier_transforms.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/utils/classifier_transforms.py)
  - Builds augmentation pipelines for transfer-learning classifiers.
  - Training augmentations include brightness and contrast jitter, random blur, Gaussian noise, horizontal flips, and small rotations.
  - Validation transforms use deterministic resizing plus ImageNet normalization.

- [`utils/classifier_metrics.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/utils/classifier_metrics.py)
  - Collects logits and targets across an epoch.
  - Computes accuracy, precision, recall, F1, confusion matrix, and `closed_recall`.
  - Tunes the closed-eye probability threshold to optimize either F1 or recall on validation data.

- [`train_transfer_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_transfer_detector.py)
  - Full transfer-learning training loop for stronger eye-state classification.
  - Supports `resnet18` and `mobilenetv2`.
  - Uses focal loss to handle class imbalance.
  - Applies low-light-oriented augmentation for robustness.
  - Supports `--detector-input-mode raw|enhanced|dual`.
  - Can optionally train on enhancer outputs:
    - loads a frozen pretrained enhancer
    - enhances each training image on the fly
    - normalizes the enhanced image before detector inference
    - keeps enhancer weights frozen and verifies no enhancer gradients are stored
    - initializes the detector from a clean-trained detector checkpoint
    - saves the enhanced-detector checkpoint as `detector_enhanced.pth` by default
  - In dual-input mode, feeds both normalized raw low-light and normalized enhanced images into the detector, and can initialize the dual branch weights from a single-input checkpoint for faster adaptation.
  - Includes `ReduceLROnPlateau`, early stopping, threshold tuning, optional low-light validation every epoch, and tqdm progress bars.
  - Detects CUDA or MPS automatically.
  - Reduces the starting batch size for smaller GPUs and can retry with a smaller batch size after an out-of-memory error.
  - Supports runtime-aware relative paths so the same command can be used on a local Mac or in Colab.
  - Can mount Google Drive and copy best/last checkpoints into a Drive backup directory.
  - Saves best and last checkpoints with model config, threshold, and metric history.

- [`train_baseline.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_baseline.py)
  - Saves the best baseline checkpoint.
  - Can also export epoch-wise JSON history for training and validation curves.

- [`utils/colab_runtime.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/utils/colab_runtime.py)
  - Detects whether the script is running locally or in Colab.
  - Mounts Google Drive when requested.
  - Resolves relative paths against local or Colab workspace roots.
  - Copies checkpoints to a secondary backup directory, such as Google Drive.

- [`evaluate_transfer_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/evaluate_transfer_detector.py)
  - Loads a saved transfer-learning detector checkpoint.
  - Evaluates clean and optional low-light datasets with the saved or retuned threshold.
  - Saves:
    - `experiment_results.csv` for report tables
    - `evaluation_results.csv` for detailed metrics
    - confusion-matrix plots
    - metric comparison plots with explicit low-light drop annotations
    - a text summary

- [`generate_report_results.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/generate_report_results.py)
  - Evaluates the main experiments used in the final report:
    - baseline CNN on clean data
    - baseline CNN on low-light data
    - enhancer-only pipeline
    - joint enhancer-detector model
  - Computes accuracy, precision, recall, F1-score, and FPS.
  - Exports:
    - CSV comparison table
    - Markdown comparison table
    - high-resolution PNG and PDF table figure
    - high-resolution PNG and PDF confusion-matrix figure
    - high-resolution PNG and PDF training-curves figure
    - summary text

### Enhancement Training

- [`models/zerodce.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/zerodce.py)
  - Implements the Zero-DCE-style lightweight curve-estimation network.
  - Predicts iterative RGB curve maps and applies them directly in the forward pass.

- [`losses/enhancement_losses.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/enhancement_losses.py)
  - Implements exposure control, color constancy, illumination smoothness, and optional spatial consistency losses.
  - Provides the combined `enhancement_loss()` helper used during standalone enhancer training.

- [`train_enhancer.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_enhancer.py)
  - Trains the enhancer alone on low-light images without needing class supervision.
  - Reuses the same folder-based dataset layout as the detector scripts, but ignores the `open/closed` labels during loss computation.
  - Supports GPU auto-detection, CUDA AMP, gradient clipping, `ReduceLROnPlateau`, early stopping, epoch-wise tqdm progress bars, and configurable Zero-DCE loss weights.
  - Saves:
    - `enhancer_best.pt`
    - `enhancer_last.pt`
    - `enhancer_training_history.json`
  - These checkpoints are directly compatible with [`inference_enhancer.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/inference_enhancer.py) and can be passed into [`generate_report_results.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/generate_report_results.py) for the `Enhancer Only` row.

### Enhancer + Frozen Detector

- [`models/frozen_detector_pipeline.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/frozen_detector_pipeline.py)
  - Wraps an enhancer and a clean-trained detector into `low-light -> enhancer -> frozen detector`.
  - Freezes all detector parameters.
  - Keeps the detector in eval mode even if the outer pipeline is switched to train mode.
  - Prints detector freeze state and reuses the shared detector freeze/debug helpers.
  - Normalizes enhanced images before detector inference.
  - If the loaded detector is dual-input, it feeds both normalized raw low-light and normalized enhanced images into the frozen detector during evaluation.

- [`evaluate_enhancer_frozen_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/evaluate_enhancer_frozen_detector.py)
  - Loads a trained enhancer checkpoint and a clean-trained detector checkpoint.
  - Evaluates:
    - original detector on raw low-light images
    - enhancer + original detector on the same low-light images
    - enhancer + fine-tuned detector when an enhanced-detector checkpoint is provided
    - dual input detector when a dual-input detector checkpoint is provided
    - optional clean-detector reference if a clean dataset root is provided
  - Uses `torch.no_grad()` during the experiment so only forward passes are used.
  - Prints detector freeze/debug info before and after evaluation stages.
  - Verifies that no detector gradients are stored during the experiment.
  - Tries configurable threshold candidates such as `0.3, 0.4, 0.5, 0.6`, computes metrics for each threshold, and selects the best threshold by F1-score.
  - Saves:
    - `experiment_results.csv`
    - `evaluation_results.csv`
    - `threshold_sweep.csv`
    - `confusion_matrices.png`
    - `metric_comparison.png`
    - `evaluation_summary.txt`
  - This is the direct answer to the question: "does enhancement alone improve low-light detection if the detector itself stays fixed?"

### Colab Helpers

- [`notebooks/train_transfer_detector_colab.ipynb`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/notebooks/train_transfer_detector_colab.ipynb)
  - A short starter notebook for Colab training.
  - Includes the top setup cells for Drive mounting, dependency installation, path configuration, training, and evaluation.

- [`requirements.txt`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/requirements.txt)
  - Lists the core runtime dependencies needed for training and evaluation.

- [`models/joint_model.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/joint_model.py)
  - Builds a differentiable pipeline: `low-light input -> enhancer -> enhanced image -> detector -> class logits`.
  - Returns classification logits, enhanced image, and optionally Zero-DCE curve maps.
  - Includes strict shape checks for `224x224` RGB inputs.
  - Includes development-mode debug prints for tensor shapes and value ranges.
  - Keeps gradient flow intact from detector loss back through the enhancer.

- [`metrics.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/metrics.py)
  - Running loss and classification metrics.
  - Accuracy, precision, recall, F1.
  - Confusion matrix accumulation.
  - Formatting helpers for clean logging.

- [`configs/train_config.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/configs/train_config.py)
  - Central dataclasses for enhancement-loss, joint-loss, and joint-training settings.
  - Exposes `enhancement_lambda` as an experiment knob.
  - Includes helper methods to clone configs with a new lambda for quick sweeps.

- [`train_joint.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_joint.py)
  - Full end-to-end training loop for `input -> enhancer -> detector`.
  - Loads low-light training images directly.
  - Computes detection loss, enhancement loss, and total loss every step.
  - Backpropagates through the whole pipeline.
  - Uses GPU automatically when available.
  - Supports gradient clipping, CUDA mixed precision, last-checkpoint saving, and best-checkpoint saving by validation accuracy.

- [`validate_joint.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/validate_joint.py)
  - Reusable validation loop for the joint model.
  - Uses the same raw-image transform as joint training.
  - Can be run as a script to validate a saved joint checkpoint.
  - Reports total loss, detection loss, enhancement loss, accuracy, precision, recall, and F1.

- [`train_baseline.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_baseline.py)
  - Full PyTorch training script for the baseline CNN.
  - Uses `CrossEntropyLoss`.
  - Uses `Adam`.
  - Uses a `StepLR` scheduler.
  - Runs full train and validation loops.
  - Tracks best model by validation accuracy.
  - Saves checkpoint with model state, optimizer state, scheduler state, class mapping, and validation metrics.

- [`evaluate.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/evaluate.py)
  - Loads a trained baseline checkpoint.
  - Evaluates on a clean dataset and an optional low-light dataset.
  - Defaults to `num_workers=0` for maximum compatibility.
  - Computes:
    - loss
    - accuracy
    - precision
    - recall
    - F1
    - confusion matrix
  - Saves:
    - `evaluation_results.csv`
    - `confusion_matrices.png`
    - `metric_comparison.png`
    - `evaluation_summary.txt`
  - Prints a short report-ready explanation of performance degradation under low light.

### Low-Light Simulation

- [`low_light_simulator.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/low_light_simulator.py)
  - Reusable functions for simulating low-light effects.
  - Supports gamma darkening, brightness reduction, contrast reduction, black-level shifting, desaturation, motion blur, Gaussian noise, and Poisson noise.
  - The `black_level_shift` control pushes shadows toward near-black so the degraded set can better match the underexposed behavior described in the project brief.

- [`generate_lowlight_dataset.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/generate_lowlight_dataset.py)
  - Creates a degraded copy of a dataset.
  - Preserves class structure.
  - Supports `standard`, `severe`, and `extreme` low-light profiles through `--profile`.
  - The current `severe` profile is intentionally strong but not near-black; `extreme` is reserved for harsher stress testing.
  - Writes a CSV log describing which degradations were applied, including the selected profile and degradation parameters.

### Zero-DCE Style Enhancement

- [`models/zerodce.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/models/zerodce.py)
  - Implements a compact Zero-DCE style curve-estimation CNN.
  - Predicts iterative enhancement curve maps.
  - Applies the Zero-DCE style enhancement rule:
    - `x = x + r * (x^2 - x)`
  - Returns:
    - enhanced image
    - curve maps

- [`inference_enhancer.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/inference_enhancer.py)
  - Loads a single image.
  - Optionally loads a trained Zero-DCE checkpoint.
  - Runs enhancement.
  - Saves the enhanced output image.

- [`visualize_enhancement.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/visualize_enhancement.py)
  - Runs the enhancer on one image.
  - Saves a side-by-side input vs enhanced comparison figure.
  - Also optionally saves the enhanced image itself.

### Zero-DCE Losses

- [`losses/enhancement_losses.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/enhancement_losses.py)
  - `exposure_control_loss()`
  - `color_constancy_loss()`
  - `illumination_smoothness_loss()`
  - `spatial_consistency_loss()`
  - `enhancement_loss()`
  - All main hyperparameters are configurable.

- [`losses/joint_loss.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/losses/joint_loss.py)
  - Implements `L_total = L_detection + lambda * L_enhancement`.
  - Uses `CrossEntropyLoss` for detection.
  - Reuses the combined Zero-DCE enhancement loss.
  - Returns separate detection, enhancement, and weighted-enhancement terms for logging.
  - Includes a `JointTrainingLoss` wrapper and a helper to convert tensor losses into loggable floats.

- [`tests/test_losses.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/tests/test_losses.py)
  - Small tests plus usage examples for the enhancement losses.
  - Can be run directly to check the loss API and see example outputs.

- [`tests/test_detector.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/tests/test_detector.py)
  - Small smoke tests for the detector backbones.
  - Prints model summaries for the custom CNN and MobileNetV2 detector.
  - Runs dummy forward passes and checks that both backbones return a `(batch_size, 2)` output.

- [`tests/test_joint_model.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/tests/test_joint_model.py)
  - Smoke tests the full enhancer-detector pipeline on dummy tensors.
  - Checks output shapes for logits, enhanced image, and curve maps.
  - Verifies the optional no-curve-maps return path.
  - Confirms detector loss backpropagates into enhancer parameters.
  - Captures debug prints and validates clear shape-check errors.

### Joint Training Utilities

- [`utils/checkpoints.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/utils/checkpoints.py)
  - Shared helpers for writing and loading joint-training checkpoints.
  - Saves model state, optimizer state, AMP scaler state, configs, class mapping, and train/validation metrics.

- [`utils/metrics.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/utils/metrics.py)
  - Joint-training metric accumulator.
  - Tracks total loss, detection loss, enhancement loss, accuracy, precision, recall, F1, and confusion matrix.
  - Formats concise epoch log lines for train/validation output.

## How the Main Pieces Are Implemented

### 1. Frame Extraction

The raw dataset is video-based, so the classification pipeline first converts videos into images. The extractor:

- walks the input video tree
- samples frames at a specified interval
- removes near-duplicates
- detects a face region with Haar cascades
- crops a face, upper-face, or eye-focused region

This produces compact eye-relevant images instead of full-frame video stills.

### 2. Auto-Labeling

The labeler is designed to be resilient across environments:

- If classic MediaPipe Face Mesh is available, it uses facial landmarks and EAR.
- If the newer MediaPipe Tasks stack is used, it can work with a provided Face Landmarker `.task` model.
- If neither is practical, it falls back to OpenCV Haar eye detection.

This makes the labeling step usable even when MediaPipe API versions differ across systems.

### 3. Dataset Balancing and Splitting

After labeling:

- images are grouped into `open` and `closed`
- the larger class can be downsampled to the smaller class size
- the final balanced data is split into train and validation sets

This prevents the classifier from being dominated by one eye-state class.

### 4. Baseline Classification

The baseline model is intentionally lightweight:

- three convolution blocks
- batch normalization for stable training
- max-pooling for spatial downsampling
- dropout before the final linear layer

The training code logs the standard classification metrics and keeps the best checkpoint by validation accuracy.

The repository now also includes a detector builder module that can switch between:

- a compact custom CNN for lightweight experiments
- a MobileNetV2 classifier for transfer learning and fine-tuning

For MobileNetV2, the implementation freezes early feature layers and keeps only the last feature blocks and classifier trainable, which makes it suitable for low-data fine-tuning on the eye-state dataset.

### 5. Low-Light Robustness Evaluation

Instead of evaluating only on clean images, the project also measures robustness under degradation:

- a low-light version of the validation set is generated
- the degradation pipeline can now be scaled with `--profile standard|severe|extreme`
- stronger profiles reduce brightness further and crush shadows to create a more meaningful robustness gap
- the same classifier is evaluated on clean and low-light data
- metrics are saved to CSV
- confusion matrices and metric-comparison plots are generated

This makes the evaluation output directly useful for reports and ablation-style comparisons.

On the current smoke subset, the original mild low-light setting produced only a small drop from `59.63%` clean accuracy to `55.93%` low-light accuracy. The first severe profile made the task much harder, but review of real artifacts showed that some images had become too close to near-black noise. The current moderated severe profile still produces a meaningful drop to `52.59%` accuracy and `23.81%` F1 while preserving more recoverable visual structure. The `extreme` profile remains available for harsher stress testing.

### 6. Zero-DCE Enhancement

The enhancement track is separate from the classifier:

- a Zero-DCE style network predicts enhancement curves
- these curves are applied iteratively to brighten low-light images
- inference and visualization scripts support image-level testing
- enhancement losses are implemented for future model training

At the moment, the Zero-DCE architecture and losses are implemented, and inference works. A trained Zero-DCE checkpoint is not yet included in the repository.

### 7. Joint Enhancement + Detection

The repository now also includes a joint model that connects the two tracks directly:

- the Zero-DCE enhancer first processes a low-light RGB image
- the enhanced image is passed straight into the eye-state detector
- the model returns detector logits plus the enhancement outputs
- because the pipeline is fully differentiable, classification loss can update both the detector and the enhancer together

This is the core model structure needed for end-to-end training where low-light enhancement is optimized for the downstream eye-state classification task.

### 8. Joint Training Objective

For end-to-end training, the repository now includes a configurable total loss:

- `L_total = L_detection + lambda * L_enhancement`
- `L_detection` is standard cross-entropy on the eye-state logits
- `L_enhancement` is the weighted Zero-DCE enhancement objective
- `lambda` is stored in the training config so it can be tuned cleanly across experiments

This makes it easy to try settings such as stronger detection-first training with a small lambda or more enhancement-regularized training with a larger lambda.

### 9. Joint Training Loop

The repository now includes a full training loop for the end-to-end system:

- low-light images are loaded directly from the dataset
- each batch goes through `enhancer -> detector`
- the loop computes detection loss, enhancement loss, and total loss
- `total loss` is backpropagated through both modules together
- validation runs after every epoch
- checkpoints are saved for both the latest state and the best validation accuracy

For joint training, the image transform intentionally keeps tensors in the raw `[0, 1]` range instead of applying ImageNet normalization, because the Zero-DCE enhancement losses operate on image intensities directly.

## End-to-End Commands

### A. Prepare an Eye-State Dataset From Raw Videos

Extract frames:

```bash
python3 extract_frames.py \
  --input-dir data/Fold3_part2 \
  --output-dir artifacts/extracted_frames \
  --interval-seconds 1
```

Label frames:

```bash
python3 label_eye_state.py \
  --input-dir artifacts/extracted_frames \
  --output-dir artifacts/labeled_frames
```

Prepare a balanced train/val dataset:

```bash
python3 prepare_dataset.py \
  --input-dir artifacts/labeled_frames \
  --output-dir artifacts/dataset \
  --clear-output
```

Verify the dataset:

```bash
python3 verify_dataset.py \
  --dataset-root artifacts/dataset \
  --output-dir artifacts/dataset_report
```

### B. Train and Evaluate the Baseline CNN

Train:

```bash
python3 train_baseline.py artifacts/dataset \
  --epochs 20 \
  --batch-size 32 \
  --save-path checkpoints/baseline_cnn_best.pt
```

Generate a low-light validation set:

```bash
python3 generate_lowlight_dataset.py \
  --input-dir artifacts/dataset/val \
  --output-dir artifacts/lowlight_val \
  --strength 1.0 \
  --profile severe
```

Evaluate clean vs low-light:

```bash
python3 evaluate.py checkpoints/baseline_cnn_best.pt artifacts/dataset \
  --lowlight-data-root artifacts/lowlight_val \
  --split val \
  --output-dir artifacts/evaluation_report
```

The low-light generator supports three presets:

- `standard`: mild degradation for a small realism-focused shift
- `severe`: much darker images with stronger shadow crushing and noise
- `extreme`: stress-test setting for very dark synthetic inputs

### C. Run Zero-DCE Enhancement

Enhance a single image:

```bash
python3 inference_enhancer.py lowlight_image.png \
  --output-path outputs/enhanced.png
```

Save input vs enhanced visualization:

```bash
python3 visualize_enhancement.py lowlight_image.png \
  --enhanced-output-path outputs/enhanced.png \
  --output-path outputs/comparison.png
```

### D. Run Zero-DCE Loss Tests

```bash
python3 tests/test_losses.py
```

### E. Run Detector Smoke Tests

```bash
python3 tests/test_detector.py
```

### F. Run Joint-Model Smoke Tests

```bash
python3 tests/test_joint_model.py
```

### G. Smoke-Check the Joint Loss API

```bash
python3 - <<'PY'
import torch
from configs.train_config import JointTrainConfig
from losses.joint_loss import JointTrainingLoss, loss_dict_to_log_items

config = JointTrainConfig().with_joint_loss_lambda(0.25)
criterion = JointTrainingLoss(config)

logits = torch.randn(2, 2, requires_grad=True)
inputs = torch.rand(2, 3, 224, 224)
enhanced = torch.clamp(inputs * 0.9 + 0.05, 0.0, 1.0)
curve_maps = torch.rand(2, 8, 3, 224, 224) * 0.2 - 0.1
targets = torch.tensor([0, 1])

loss_dict = criterion(
    logits,
    targets,
    enhanced,
    curve_maps,
    input_image=inputs,
)
print(loss_dict_to_log_items(loss_dict))
PY
```

### H. Train the Joint Model

```bash
python3 train_joint.py /path/to/lowlight_dataset \
  --epochs 20 \
  --batch-size 16 \
  --detector-backbone custom \
  --enhancement-lambda 1.0 \
  --checkpoint-dir checkpoints/joint
```

### I. Validate a Saved Joint Checkpoint

```bash
python3 validate_joint.py checkpoints/joint/joint_best.pt /path/to/lowlight_dataset \
  --batch-size 16
```

## Important Outputs

Common outputs created by the pipeline:

- extracted frames: `artifacts/.../extracted_frames`
- labeled images: `artifacts/.../labeled_*`
- balanced dataset: `artifacts/.../dataset_*`
- dataset verification report: `dataset_report.txt`, `class_counts.png`
- baseline checkpoint: `*.pt`
- low-light dataset copy: `lowlight_*`
- evaluation report:
  - `evaluation_results.csv`
  - `confusion_matrices.png`
  - `metric_comparison.png`
  - `evaluation_summary.txt`
- Zero-DCE outputs:
  - enhanced image
  - comparison figure

## Notes and Current Caveats

- The repository currently uses a lightweight baseline classifier for smoke validation, not a final production classifier.
- `models/detector.py` adds a MobileNetV2 fine-tuning option. The new joint trainer can use it, but the older baseline-only training script still uses `BaselineCNN`.
- `train_joint.py` and `validate_joint.py` are implemented and smoke-tested, but so far they have only been exercised on a tiny synthetic dataset and not yet on the full real low-light dataset.
- `train_joint.py` and `validate_joint.py` now also pass a real-data mini smoke check on severe low-light images. A full real-dataset epoch on CPU is still slow in this sandbox, so the practical verification here used a smaller real subset for turnaround.
- `label_eye_state.py` now handles MediaPipe API differences, but the `mediapipe-tasks` path requires a Face Landmarker `.task` file if you want to use that backend explicitly.
- The OpenCV Haar fallback is less semantically precise than landmark-based EAR labeling, but it makes the pipeline runnable in environments where the old MediaPipe Face Mesh API is unavailable.
- The first low-light simulation was too mild for the project objective. The current pipeline now includes stronger `severe` and `extreme` degradation profiles, which better expose the classifier's weakness under dark conditions.
- `extreme` is useful as a stress test, but it may be unrealistically dark for the final report. The `severe` preset was adjusted after artifact review so it remains challenging without collapsing as many samples into near-black noise.
- In this sandbox, MobileNetV2 pretrained weights could not be downloaded because external network access is restricted. The code is set up to use pretrained weights when available and falls back safely for local offline smoke tests.
- Joint training intentionally uses raw `[0, 1]` image tensors for Zero-DCE compatibility. If you later rely heavily on pretrained MobileNetV2, it may be worth experimenting with a detector-side normalization step after enhancement.
- Zero-DCE inference is implemented and working, but meaningful enhancement quality depends on training the model and providing a trained checkpoint.

## Recommended Next Steps

- Run the repaired dataset-preparation pipeline across all raw videos, not just the smoke-check subset.
- Regenerate the full degraded dataset with `--profile severe` and use it as the main low-light benchmark.
- Train the baseline classifier for more epochs and inspect confusion-matrix trends across clean, severe low-light, and enhanced images.
- Train the Zero-DCE model with the implemented enhancement losses.
- Run [`train_joint.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/train_joint.py) on the real severe low-light dataset and compare checkpoints across different lambda values.
- Use [`validate_joint.py`](/Users/sarthakbaghel/Documents/Projects/Task-driven low-light enhancement/validate_joint.py) to compare the best joint checkpoints against the baseline detector.
- Re-evaluate classifier robustness after applying trained enhancement before classification.
