# Session 6 — Backpropagation and MNIST CNN

This session connects the mathematics of gradient descent with a practical convolutional neural network trained on MNIST.

## Core concepts

- **Forward propagation:** weighted inputs, activations, and loss are calculated through a small neural network.
- **Backpropagation:** the chain rule propagates output error backward to obtain every weight gradient.
- **Gradient descent:** spreadsheet experiments compare how learning rate affects convergence.
- **CNN feature extraction:** convolutions learn spatial patterns while pooling expands receptive field and reduces resolution.
- **Regularization and normalization:** batch normalization and dropout stabilize training and reduce overfitting.
- **Global average pooling:** spatial feature maps are reduced without a parameter-heavy fully connected head.
- **PyTorch training workflow:** data transforms, loaders, training/validation loops, loss tracking, and device selection are separated into reusable modules.

## Contents

- `BackPropogation/` contains the calculation workbook, network diagram, formulas, and learning-rate plots.
- `model.py` defines the MNIST CNN.
- `utils.py` provides transforms, plotting, device, training, and testing helpers.
- `S6.ipynb` loads MNIST, summarizes the model, trains it, and plots metrics.

The CNN has **17,462 parameters**, keeping the model compact while demonstrating a complete image-classification pipeline.
