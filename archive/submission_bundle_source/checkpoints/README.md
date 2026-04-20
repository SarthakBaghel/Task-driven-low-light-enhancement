# Checkpoint Guide

This folder is a placeholder for the trained model checkpoints.

## Recommended Files

Place these files here when preparing a final ZIP/demo bundle:

```text
clean_detector_best.pt
mixed_detector_best.pt
lowlight_detector_best.pt
```

## Exact Final Checkpoints From This Project

If you already downloaded the final checkpoints from the development workspace, these are the original files to use:

```text
kaggle_v1_checkpoints/full_clean_detector_subject_class_balanced_20k_resnet18/kaggle_v1_clean_full_subject_class_balanced_20k_best.pt
kaggle_v1_checkpoints/detector_mixed_subject_class_balanced_20k_eye_mid_resnet18/detector_mixed_best.pt
kaggle_v1_checkpoints/detector_lowlight_subject_class_balanced_20k_eye_mid_resnet18/detector_lowlight_best.pt
```

Recommended rename/copy mapping for this submission bundle:

```text
kaggle_v1_clean_full_subject_class_balanced_20k_best.pt -> checkpoints/clean_detector_best.pt
detector_mixed_best.pt -> checkpoints/mixed_detector_best.pt
detector_lowlight_best.pt -> checkpoints/lowlight_detector_best.pt
```

## Best Way to Share Them

For GitHub submission:

- keep this folder empty in the public repository,
- upload the actual checkpoint files to Google Drive,
- share the Google Drive link with the examiner,
- mention that link in your report or email.

## If You Are Submitting a ZIP

Before zipping the project locally, copy the checkpoint files into this folder so the demo runs directly.

Example local copy commands from the downloaded root project:

```bash
cp ../kaggle_v1_checkpoints/full_clean_detector_subject_class_balanced_20k_resnet18/kaggle_v1_clean_full_subject_class_balanced_20k_best.pt checkpoints/clean_detector_best.pt
cp ../kaggle_v1_checkpoints/detector_mixed_subject_class_balanced_20k_eye_mid_resnet18/detector_mixed_best.pt checkpoints/mixed_detector_best.pt
cp ../kaggle_v1_checkpoints/detector_lowlight_subject_class_balanced_20k_eye_mid_resnet18/detector_lowlight_best.pt checkpoints/lowlight_detector_best.pt
```

Example demo command:

```bash
python demo/run_demo.py \
  --clean-checkpoint checkpoints/clean_detector_best.pt \
  --mixed-checkpoint checkpoints/mixed_detector_best.pt \
  --image-dir demo/sample_images \
  --output-dir demo_outputs
```
