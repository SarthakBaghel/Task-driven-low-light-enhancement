# Demo

This demo compares:

- clean detector
- mixed detector

on a small curated set of low-light eye images.

It is designed for a quick viva or project presentation from the **repository root**.

## What Is Already Prepared

The repository already contains:

- a curated demo image folder in `sample_images/`
- a demo script in `run_demo.py`

So you only need to place the checkpoints correctly and run the command.

## Required Checkpoints

Place these files in:

```text
checkpoints/
```

Expected names:

```text
checkpoints/clean_detector_best.pt
checkpoints/mixed_detector_best.pt
```

Optional:

```text
checkpoints/lowlight_detector_best.pt
```

If the files are not already present locally, use the Google Drive link documented in:

```text
checkpoints/README.md
```

## Exact Steps to Run the Demo

### Step 1

Open a terminal inside:

```text
Task-driven low-light enhancement/
```

### Step 2

Make sure the checkpoint files are present:

```text
checkpoints/clean_detector_best.pt
checkpoints/mixed_detector_best.pt
```

### Step 3

Create and activate a virtual environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Step 4

Run:

```bash
python demo/run_demo.py \
  --clean-checkpoint checkpoints/clean_detector_best.pt \
  --mixed-checkpoint checkpoints/mixed_detector_best.pt \
  --image-dir demo/sample_images \
  --output-dir demo_outputs
```

### Step 5

Open the generated files in:

```text
demo_outputs/
```

## Output Files

```text
demo_outputs/predictions.csv
demo_outputs/demo_contact_sheet.png
demo_outputs/demo_summary.txt
```

## Best File to Show During Presentation

The easiest file to show during the viva is:

```text
demo_outputs/demo_contact_sheet.png
```

## What the Examiner Will See

The contact sheet shows:

- the low-light eye image
- the ground-truth class
- the clean detector prediction
- the mixed detector prediction
- the closed-eye probability for both detectors

This makes it easy to explain that:

- the clean detector works well on normal images but can fail in low light
- the mixed detector is more robust in low light

## If You Want to Reuse the Curated Demo Set

The current demo samples were selected from the final low-light benchmark and are already placed in:

```text
demo/sample_images/open/
demo/sample_images/closed/
```

So you do not need to add new demo images unless you want to customize the presentation.
