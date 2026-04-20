# GitHub Setup Guide

Yes, this submission folder can become its **own GitHub repository**.

## Important Note

Your current project root is already a Git repository. Because of that, the cleanest way is:

- keep this submission folder as a prepared bundle here,
- copy or move it outside the parent repository,
- initialize Git there,
- then push it as a separate GitHub repo.

This is better than creating a nested Git repo inside another Git repo.

## Recommended Workflow

### 1. Copy this folder outside the parent repo

Example:

```bash
cp -R lowlight-eye-state-detection-submission ~/Desktop/lowlight-eye-state-detection-submission
```

or move it:

```bash
mv lowlight-eye-state-detection-submission ~/Desktop/
```

### 2. Go inside the copied folder

```bash
cd ~/Desktop/lowlight-eye-state-detection-submission
```

### 3. Initialize a new Git repository

```bash
git init
git add .
git commit -m "initial college submission"
```

### 4. Create a new empty GitHub repo

Suggested repo name:

```text
lowlight-eye-state-detection-submission
```

### 5. Connect and push

```bash
git branch -M main
git remote add origin <YOUR_NEW_GITHUB_REPO_URL>
git push -u origin main
```

## If You Want to Keep It Inside the Current Repo

It is technically possible to run `git init` inside this folder while it still sits inside the parent repo, but that creates a **nested repository**, which is usually confusing and not recommended for submission.

## Best Practice for Your Case

For the examiner:

1. keep this folder as the clean prepared bundle,
2. copy it outside the current repo,
3. initialize a new Git repo there,
4. push that new folder as the final submission repository,
5. share checkpoints separately through Google Drive.
