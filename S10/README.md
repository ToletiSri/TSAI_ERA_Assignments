# Session 10 — Custom ResNet on CIFAR-10

This assignment trains a purpose-built residual network for CIFAR-10 using modern augmentation and learning-rate policies.

## Core concepts

- **Residual learning:** skip connections add a block input to learned residual features, improving gradient flow through deeper models.
- **Stage-wise feature hierarchy:** convolutional stages increase channels from 64 to 512 while pooling reduces spatial resolution.
- **Data augmentation:** padded random crop, horizontal flip, and CutOut improve robustness.
- **One Cycle policy:** an LR finder selects the maximum learning rate, which rises early and anneals over 24 epochs.
- **Optimization:** Adam and cross-entropy train the classifier with a large batch size.
- **Reusable experiment structure:** architecture and training utilities live outside the notebook.

## Architecture

A 64-channel preparation layer feeds 128-, 256-, and 512-channel stages. Residual blocks are used at the 128- and 512-channel stages, followed by 4×4 pooling and a 10-class linear head. The network has **6,573,130 trainable parameters**.

## Contents and result

- `custom_resnet.py` defines the model.
- `utils.py` provides Albumentations transforms, training/evaluation, LR finding, and plotting helpers.
- `S10.ipynb` assembles and runs the experiment.

The recorded validation accuracy reaches **90.63%** at epoch 24.
