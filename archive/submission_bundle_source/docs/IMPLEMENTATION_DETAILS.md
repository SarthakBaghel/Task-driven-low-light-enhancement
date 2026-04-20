# Implementation Details

This document gives a concise technical explanation of how the final submission was implemented.

## 1. Data Preparation

The project uses a Kaggle eye-image dataset already labeled into `open` and `closed`.

Main steps:

1. reorganize the downloaded dataset into a clean root,
2. extract subject IDs from filenames,
3. create a subject-wise train/validation/test split,
4. materialize those splits into separate folders.

This avoids data leakage between train and test.

## 2. Clean Baseline Training

The first model trained was a clean baseline detector using:

- ResNet18 backbone
- transfer learning
- binary classification
- threshold tuning for the `closed` class

Main script:

```text
src/train_transfer_detector.py
```

## 3. Low-Light Data Generation

Synthetic low-light data was created from the clean split using:

```text
src/generate_lowlight_dataset.py
src/low_light_simulator.py
```

The selected setting was `eye_mid`, which gives a moderate but meaningful degradation.

## 4. Low-Light Evaluation

The clean detector was then evaluated on:

- clean test images
- low-light test images

Main script:

```text
src/evaluate_transfer_detector.py
```

This showed that low light reduces low-light closed-eye recall significantly.

## 5. Detector Adaptation Experiments

Two adaptation strategies were compared:

### Low-light-only detector

Train on low-light data only.

Result:

- strong low-light recall
- poor clean-image performance
- overfits to degraded input

### Mixed detector

Train on both clean and low-light data.

Result:

- strongest overall clean + low-light balance
- best final model

## 6. Final Validation Strategy

The final three-model comparison was run on:

- one balanced held-out 5k subset
- a second balanced held-out 5k subset

This was done to verify that the result was not due to only one sampled subset.

## 7. Demo Strategy

The final demo is intentionally simple:

- take a few sample eye images,
- run the clean detector,
- run the mixed detector,
- show both predictions side by side.

Main demo script:

```text
demo/run_demo.py
```

This makes the project easy to explain in a college presentation.
