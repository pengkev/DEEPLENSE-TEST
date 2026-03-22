# DeepLense Test Repository

This repository contains my solutions for multiple **DeepLense / ML4SCI test tasks**. Each task lives in its own subdirectory and is documented separately with its own notebook, results, saved artifacts, and task-specific discussion.

The goal of this top-level README is to provide a quick orientation to the repository structure. For the full methodology, experimental setup, findings, figures, and model-specific conclusions, please refer to the README inside each task folder.

## Repository Structure

```text
DEEPLENSE-TEST/
|-- README.md
|-- Task-1/
|   |-- README.md
|   `-- Task_1.ipynb
`-- Task-6/
    |-- README.md
    `-- Task_6.ipynb
```

## Available Tasks

### Task 1: Multi-Class Lensing Classification

[Task-1](./Task-1) contains a transfer-learning study for gravitational lensing classification. The work compares multiple pretrained vision backbones on a three-class substructure-labeling problem and evaluates them with accuracy, ROC curves, and macro ROC-AUC.

For the detailed writeup, results tables, figures, and conclusions, see [Task-1/README.md](./Task-1/README.md).

### Task 6: Strong-Lensing Super-Resolution

[Task-6](./Task-6) contains a multi-model super-resolution study on paired low-resolution / high-resolution lensing data. The work includes both simulated supervised training and real-data fine-tuning, comparing CNN and transformer-based models with `MSE`, `PSNR`, and `SSIM`.

For the detailed writeup, reconstruction results, saved checkpoints, and task-specific interpretation, see [Task-6/README.md](./Task-6/README.md).

## Notebooks

Each task directory includes a Jupyter notebook containing the full experimental workflow:

- [Task-1/Task_1.ipynb](./Task-1/Task_1.ipynb)
- [Task-6/Task_6.ipynb](./Task-6/Task_6.ipynb)

These notebooks include the code used for data loading, preprocessing, model construction, training, evaluation, and visualization.

## What to Look At

If you are reviewing the repository quickly, the best path is:

1. start with the task-level README for the problem you care about
2. open the corresponding notebook for the implementation details
3. review the saved figures and checkpoints inside that task directory if needed

## Notes

- Task-specific findings are intentionally kept inside each subdirectory README rather than duplicated here.
- The task folders contain the most important context for interpreting the results.
- Large model weight files are tracked with **Git LFS** where applicable.
