# Session 7 — Iterative MNIST Model Design

This assignment develops nine CNN variants to reach at least **99.4% MNIST validation accuracy** with fewer than **8,000 parameters** in no more than **15 epochs**.

## Core concepts

- **Target-driven experimentation:** each notebook states a target, records results, analyzes limitations, and motivates the next model.
- **Receptive field:** early experiments show that a classifier must see enough of the 28×28 image before making a decision.
- **Parameter efficiency:** channel reduction, 1×1 squeeze layers, and global average pooling replace oversized convolutional or dense layers.
- **Batch normalization and dropout:** experiments compare optimization stability against regularization and expose the cost of excessive dropout.
- **Data augmentation:** geometric variation improves validation performance without increasing model size.
- **Learning-rate scheduling:** `ReduceLROnPlateau` lowers the learning rate when validation progress stalls.
- **Squeeze–expand design:** the final network balances narrow bottlenecks with richer feature extraction.

## Experiment progression

The models shrink from millions of parameters to a compact architecture while recovering accuracy. Model 9 combines a squeeze–expand CNN, augmentation, and learning-rate scheduling to achieve **99.46% validation accuracy with 7,356 parameters**.

## Contents

- `models/model_1.py` through `models/model_9.py` contain each architecture iteration.
- `Code1.ipynb` through `Code9.ipynb` train and evaluate the matching model; `Code8WithDataAugmentation.ipynb` isolates the augmentation experiment.
- `utils.py` contains the shared data, training, evaluation, and plotting workflow.
