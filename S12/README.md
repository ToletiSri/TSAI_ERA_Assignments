# Session 12 — PyTorch Lightning and Interactive CIFAR-10 Inference

This assignment migrates the custom CIFAR-10 ResNet workflow to PyTorch Lightning and prepares it for an interactive Gradio deployment.

## Core concepts

- **Lightning abstraction:** model steps, optimization, metrics, and trainer configuration replace hand-written orchestration loops.
- **Residual classification:** the Session 10 custom ResNet remains the inference backbone.
- **Checkpointing:** trained weights are serialized and reused independently of the training notebook.
- **Model interpretability:** Grad-CAM can target a selected layer and blend an adjustable heatmap over an uploaded image.
- **Error analysis:** users can inspect a configurable number of misclassified examples.
- **Top-k prediction:** inference reports configurable class probabilities rather than only the winning class.
- **Gradio deployment:** image upload, sample inputs, and controls turn the model into an accessible application.

## Contents

- `custom_resnet.py` defines the classifier.
- `utils.py` contains data, training, visualization, misclassification, and Grad-CAM helpers.
- `S12.ipynb` trains with Lightning, saves the state dictionary, plots metrics, and inspects errors.

The deployed workflow is available in the [TSAI S12 Hugging Face Space](https://huggingface.co/spaces/ToletiSri/TSAI_S12).
