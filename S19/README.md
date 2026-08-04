# Session 19 — Zero-Shot Image Classification with CLIP

This assignment builds a small interactive application around a pretrained CLIP model.

## Core concepts

- **Vision–language embeddings:** CLIP maps images and text into the same representation space.
- **Zero-shot classification:** user-provided captions act as candidate classes, so no task-specific classifier training is required.
- **Similarity-based inference:** normalized image/text features are compared and converted into caption likelihoods with softmax.
- **Prompt sensitivity:** changing the candidate captions changes the classification task at inference time.
- **Interactive deployment:** Gradio accepts an image and captions and presents ranked probabilities.

## Contents

- `S19_Clip.ipynb` loads the pretrained model, preprocesses inputs, calculates similarities, and builds the demo.
- `cats.jpg` and `personBicycle.jpg` are example inputs.

Try the [Hugging Face Space](https://huggingface.co/spaces/ToletiSri/TSAI_S19).
