# Dataset Pipeline for v4 Model

This document outlines the dataset preparation pipeline for the v4 model of the Intelligent Image Compression & Analysis System.

## Architecture

The pipeline consists of the following components:
- `src/dataset/collect.py`: Downloads the legitimate CIFAR-10 dataset (torchvision) representing our exact 10 target classes.
- `src/dataset/validate.py`: Checks for corruption, minimum resolution (32x32), and ensures RGB format.
- `src/dataset/deduplicate.py`: Prevents exact and near-duplicates using MD5 and perceptual hashing (`imagehash`).
- `src/dataset/split.py`: Deterministically shuffles and splits images into train, validation, and test sets.
- `src/dataset/statistics.py`: Generates the dataset statistics and checks for data leakage.
- `src/dataset/build_dataset.py`: The main orchestrator that executes the pipeline.
- `scripts/check_dataset.py`: A lightweight validation script to verify the generated dataset structure and report.

## Execution

1. **Build Dataset**:
   ```bash
   python src/dataset/build_dataset.py
   ```
   *Note: This will download CIFAR-10 (~162 MB) to `data/raw_dataset/images`, process the images, and populate `data/dataset/`.*

2. **Validate Dataset**:
   ```bash
   python scripts/check_dataset.py
   ```

## Targets
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- **Train**: 4,000 images per class
- **Validation**: 1,000 images per class
- **Test**: 1,000 images per class
- **Total Dataset Size**: 60,000 images

## Safety and Git
- No model files from v3 or experimental versions have been modified.
- No massive downloads are automatically triggered without explicit user action.
- The `data/raw_dataset` and `data/dataset` directories are in `.gitignore`.
