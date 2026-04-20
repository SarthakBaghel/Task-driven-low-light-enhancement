# Figures Folder

This folder contains paper figures and placeholders for the IEEE draft.

## Included Files

- `final_results_overview.png`
  - publication-style summary plot created from the final two-seed comparison results
- `model_tradeoff_analysis.png`
  - additional publication-style plot showing clean-vs-low-light F1 trade-off and closed-eye recall by domain
- `pipeline_overview_real.png`
  - real project pipeline diagram for the paper
- `resnet18_architecture_real.png`
  - correct ResNet18 detector architecture diagram used in the methodology section
- `qualitative_recovery_real.png`
  - real low-light recovery sheet generated from the saved checkpoints and the first 5k held-out subset
- `qualitative_recovery_real_cases.csv`
  - metadata for the four recovery examples used in the real qualitative sheet
- `demo_contact_sheet_real.png`
  - real demo contact sheet generated from curated low-light sample images
- `demo_contact_sheet_real_predictions.csv`
  - predictions used for the real demo contact sheet
- `pipeline_overview_placeholder.png`
  - placeholder for the end-to-end project pipeline diagram
- `qualitative_recovery_placeholder.png`
  - placeholder for low-light examples recovered by the mixed detector
- `demo_contact_sheet_placeholder.png`
  - placeholder for the final demo contact sheet

## Recommended Replacements Before Final Submission

Replace the placeholder files with final exported visuals when available:

1. `pipeline_overview_placeholder.png`
   - kept only as a fallback; the paper now uses `pipeline_overview_real.png`
2. `demo_contact_sheet_placeholder.png`
   - kept only as a fallback; the paper now uses `demo_contact_sheet_real.png`

## Notes

- The LaTeX draft already references these filenames directly.
- The LaTeX draft now uses:
  - `pipeline_overview_real.png`
  - `resnet18_architecture_real.png`
  - `model_tradeoff_analysis.png`
  - `qualitative_recovery_real.png`
  - `demo_contact_sheet_real.png`
- The LaTeX draft now uses `qualitative_recovery_real.png` as the real qualitative figure.
- If you replace a placeholder, keep the same filename to avoid changing the paper source.
