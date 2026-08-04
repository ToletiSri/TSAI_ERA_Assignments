# Session 18 — U-Net Segmentation and Conditional VAEs

This two-part assignment studies dense prediction with U-Net and conditional generation with variational autoencoders.

## Core concepts

### U-Net segmentation

- An encoder contracts spatial features while a decoder restores resolution.
- Skip connections preserve fine detail by joining encoder features to matching decoder stages.
- Four experiments compare max pooling with strided convolution, transposed convolution with interpolation, and binary cross-entropy with Dice loss.
- Dice loss directly optimizes overlap and is useful when foreground/background pixels are imbalanced.

### Conditional variational autoencoders

- An encoder learns a distribution in latent space rather than a single deterministic code.
- The reparameterization trick permits gradients through stochastic latent sampling.
- Reconstruction and KL-divergence terms balance fidelity against a smooth, sampleable latent space.
- Labels condition MNIST and CIFAR-10 reconstruction; deliberately incorrect image/label pairs demonstrate how repeated optimization shifts output toward the supplied class.

## Contents

- `Part1/` contains shared U-Net code plus four notebooks for the downsampling, upsampling, and loss combinations.
- `Part2/` contains the conditional VAE, Lightning data modules, utilities, and MNIST/CIFAR-10 experiments.
- Result images are stored beneath each part and displayed by the notebooks.

## Observations

The segmentation runs provide side-by-side qualitative comparisons of architectural and loss choices. In the VAE experiments, a digit 5 conditioned as 9 gradually acquires features of a 9, while a horse conditioned as a bird progressively loses horse-like structure over 25 iterations.
