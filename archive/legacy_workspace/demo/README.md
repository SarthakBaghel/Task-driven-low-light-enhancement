# Demo Guide

This folder contains a simple examiner-friendly demo for the final project.

The demo compares:

- clean detector
- mixed clean + low-light detector

on the same sample eye images.

## What You Need

Provide these separately because checkpoints are too large for normal GitHub commits:

```text
clean detector checkpoint
mixed detector checkpoint
sample images
```

Recommended sample image layout:

```text
demo/sample_images/open/*.png
demo/sample_images/closed/*.png
```

If class folders are used, the script can report whether each model is correct.

## Run

From the project root:

```bash
python demo/run_demo.py \
  --clean-checkpoint /path/to/clean_detector_best.pt \
  --mixed-checkpoint /path/to/detector_mixed_best.pt \
  --image-dir demo/sample_images \
  --output-dir demo_outputs
```

## Outputs

```text
demo_outputs/predictions.csv
demo_outputs/demo_contact_sheet.png
demo_outputs/demo_summary.txt
```

For a college presentation, open `demo_outputs/demo_contact_sheet.png`. It shows each image with both model predictions.
