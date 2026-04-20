# Source Code

This folder contains the core files needed for the final detector pipeline.

## Important Files

- `train_transfer_detector.py`
  - train clean, low-light-only, or mixed detectors
- `evaluate_transfer_detector.py`
  - evaluate detector on clean and low-light datasets
- `generate_lowlight_dataset.py`
  - create low-light versions of clean eye datasets
- `dataset.py`
  - dataset loader
- `dataloader.py`
  - train/validation dataloader creation

## Support Modules

- `models/`
  - detector architectures
- `losses/`
  - loss functions
- `utils/`
  - metrics, transforms, and runtime helpers
