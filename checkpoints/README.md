# Checkpoints

The trained model binaries are intentionally **not** meant to be versioned in a normal GitHub repository.

## Recommended Submission Method

Use the GitHub repository for code, paper, notebooks, and results, and provide the model files through Google Drive.

Shared Drive folder:

- [Kaggle V1 checkpoints and experiment assets](https://drive.google.com/drive/folders/1TaIUtn7WlPQCqN11l_nM7N4b0PR1s2Qp?usp=sharing)

## Final Files Expected in This Folder

The demo and paper-related instructions expect these file names:

```text
checkpoints/clean_detector_best.pt
checkpoints/mixed_detector_best.pt
checkpoints/lowlight_detector_best.pt   # optional for comparison only
```

## Current Mapping Used in This Project

- `clean_detector_best.pt`
  - best clean baseline detector trained on `train_clean_subject_class_balanced_20k`
- `mixed_detector_best.pt`
  - final recommended detector trained on mixed clean + low-light data
- `lowlight_detector_best.pt`
  - optional comparison model trained only on low-light data

## How an Examiner Can Use Them

1. Download the checkpoints from the Google Drive folder.
2. Place the `.pt` files into this `checkpoints/` folder using the expected names above.
3. Run the demo from the repository root:

```bash
python3 demo/run_demo.py \
  --clean-checkpoint checkpoints/clean_detector_best.pt \
  --mixed-checkpoint checkpoints/mixed_detector_best.pt \
  --image-dir demo/sample_images \
  --output-dir demo_outputs
```

## Why They Are Shared Separately

Checkpoint files are large and can make a GitHub submission unnecessarily heavy. For a college submission, the most practical setup is:

- GitHub repository for the code and documentation
- Google Drive folder for the trained model binaries
