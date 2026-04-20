# Paper Draft

This folder contains a draft IEEE conference-style paper for the project.

## Files

- `ieee_draft.tex`
  - main LaTeX draft
- `EXAMINER_SCRIPT.md`
  - short viva-style script covering base paper, novelty, training pipeline, and final improvements

## What You Should Edit Before Submission

1. Replace the placeholder title/author block with your real details.
2. Add real figure files if you want to replace the current placeholders.
3. Verify the conference template requirements of your target venue.
4. Update the acknowledgment/funding section if needed.

## Suggested Figures to Add Later

- project pipeline overview
- sample clean vs low-light images
- examples where the mixed detector recovers a missed low-light case
- bar chart of clean / low-light F1 for the three models

## Compile

Example:

```bash
pdflatex ieee_draft.tex
```

If your editor runs BibTeX automatically, that is not needed here because the draft uses a self-contained `thebibliography` section.
