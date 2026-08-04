# Session 9 — Efficient CIFAR-10 CNN

This assignment designs a CIFAR-10 classifier that exceeds 85% accuracy with fewer than 200,000 parameters and no max-pooling layers.

## Core concepts

- **Dilated convolution:** increases receptive field without pooling or a proportional increase in parameters.
- **Depthwise-separable convolution:** splits spatial and channel mixing to reduce computation and parameter count.
- **Strided/downsampling alternatives:** spatial resolution is controlled through convolutional design rather than max pooling.
- **1×1 convolution:** channel projection creates efficient transition and bottleneck layers.
- **Global average pooling:** replaces a large dense classification head.
- **Batch normalization and dropout:** improve optimization and regularize the compact network.
- **Receptive-field planning:** dilation and kernel placement give the final features sufficient image context.

## Implementation

- `model.py` defines the custom CNN.
- `utils.py` provides CIFAR-10 transforms, loaders, training/testing loops, and plots.
- `S9.ipynb` runs training and evaluation.

The resulting model contains **170,592 parameters**. The notebook documents the training outcome and the trade-off made when the planned Albumentations pipeline could not be integrated.
