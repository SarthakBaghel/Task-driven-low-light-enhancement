# Examiner Script

This note is a short speaking script for viva or project demonstration.

## 1. If the Examiner Asks: What Is the Base Paper?

The main recent base paper I used for positioning the project is:

**O. F. Hassan, A. F. Ibrahim, A. Gomaa, M. A. Makhlouf, and B. Hafiz, "Real-time driver drowsiness detection using transformer architectures: a novel deep learning approach," Scientific Reports, 2025.**

Why this is the base paper:

- it is recent
- it is directly about driver drowsiness detection
- it uses open-eye and closed-eye classification
- it gives a strong modern baseline for eye-state-based drowsiness monitoring

Low-light-specific supporting paper:

**F. Alzami et al., "Time Distributed MobileNetV2 with Auto-CLAHE for Eye Region Drowsiness Detection in Low Light Conditions," IJACSA, 2024.**

What to say:

> My base paper is the 2025 Scientific Reports paper on deep-learning-based driver drowsiness detection using eye-state analysis. I used that as the main recent reference for the eye-state detection problem. For the low-light angle, I also referred to the 2024 Auto-CLAHE low-light drowsiness paper as related work.

## 2. If the Examiner Asks: What Is the Novelty of Your Project?

The safest and most honest novelty statement is:

> The novelty of my work is not a new model architecture. The novelty is in creating a subject-wise low-light evaluation setup and showing, through controlled experiments, that mixed-domain training gives a better low-light eye-state detector than clean-only or low-light-only training.

Short version:

- subject-wise split to avoid leakage
- paired clean and synthetic low-light benchmark
- controlled comparison of three training strategies
- evidence that mixed-domain training gives the best trade-off

## 3. If the Examiner Asks: How Was the Model Trained?

Short spoken answer:

> I first created a clean benchmark from the Kaggle eye dataset by organizing it into open and closed classes and then splitting it subject-wise into train, validation, and test sets. After that, I trained a clean baseline detector using ResNet18 with transfer learning. Then I generated synthetic low-light versions of the same eye images using a moderate degradation profile. After that, I trained two more detectors: one only on low-light images and one on a mixed dataset containing both clean and low-light images. Finally, I compared all three models on held-out low-light and clean test subsets.

## 4. If the Examiner Asks: What Was the Base Clean Detector?

Use this answer:

> The base clean detector was a ResNet18 transfer-learning classifier trained on the clean subject-wise training split. It used 224 by 224 input images, AdamW optimizer, focal loss with gamma equal to 2, validation-based threshold tuning, and early stopping. The clean detector was initialized with ImageNet pretrained weights and trained only on clean eye crops.

Important technical details:

- backbone: `ResNet18`
- input size: `224 x 224`
- classes: `open`, `closed`
- training data: `train_clean_subject_class_balanced_20k`
- optimizer: `AdamW`
- loss: `FocalLoss`, `gamma = 2.0`
- monitor metric: validation `F1`
- threshold objective: validation `F1`
- best threshold: `0.70`
- training budget: `15` epochs
- best clean checkpoint found at epoch `9`

## 5. If the Examiner Asks: What Improved in the Final Model?

Use this answer:

> The main improvement was not changing the backbone, but changing the training domain. The clean detector worked well on normal images, but its low-light performance dropped. The low-light-only detector improved low-light accuracy but overfit and became poor on clean images. The final mixed detector was trained on both clean and low-light data, so it learned to stay robust in the dark without forgetting the clean domain.

Important performance points:

- Clean detector:
  - clean F1: `92.86`
  - low-light F1: `88.13`
- Low-light-only detector:
  - clean F1: `69.87`
  - low-light F1: `89.83`
- Mixed detector:
  - clean F1: `93.82`
  - low-light F1: `93.98`

Most important improvement:

- low-light F1 improved from `88.13` to `93.98`
- low-light closed-eye recall improved from `79.94` to `98.86`

## 6. Ready-to-Speak Viva Script

> My project focuses on low-light eye-state detection for drowsiness monitoring. I started from a public Kaggle eye dataset and created a subject-wise benchmark so that the same subject never leaks between training and testing. Then I trained a clean baseline detector using ResNet18 transfer learning. After that, I generated synthetic low-light versions of the same eye images and compared three strategies: training only on clean images, training only on low-light images, and training on a mixed clean plus low-light dataset. The main finding was that the mixed detector gave the best balance. The clean detector dropped in low light, while the low-light-only detector overfit and became weak on clean images. The mixed detector improved low-light F1 to 93.98 while also keeping strong clean performance. So the key contribution of my work is a controlled low-light robustness study showing that mixed-domain training is more effective than clean-only or low-light-only training for this task.

## 7. Very Short Version for Fast Questions

If the examiner wants a 20-second answer:

> I used ResNet18 as the clean baseline detector, trained it on a subject-wise clean eye-state split, generated synthetic low-light data, and compared clean-only, low-light-only, and mixed training. The novelty is the subject-wise low-light benchmark and the finding that mixed-domain training gives the best low-light robustness without hurting clean performance.
