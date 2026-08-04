# Session 17 — Transformer Variants: BERT, GPT, and ViT

This session applies the Transformer building blocks to three modalities and learning objectives.

## Core concepts

### BERT

- Bidirectional encoder attention builds contextual token representations.
- Masked-language modeling hides selected tokens and predicts their identities.
- Learned token and positional embeddings feed stacked encoder layers.

### GPT

- Decoder-only, causal attention supports next-token prediction.
- Autoregressive sampling generates text one token at a time.
- Checkpoint and loss-estimation utilities support iterative training.

### Vision Transformer

- Images are split into fixed-size patches and projected into token embeddings.
- A learnable class token aggregates global image information.
- Transformer encoders replace convolutional feature extraction for classification.

## Contents

- `transformer.py` implements shared attention components and the BERT/GPT models.
- `utils.py` provides text encoding, batching, evaluation, and checkpoint helpers.
- `BERT.ipynb`, `GPT.ipynb`, and `ViT.ipynb` contain the three experiments.
- `BERT_data/` and `GPT_data/` contain their training corpora.
- `super_repo/` contains reusable data-loading, training, plotting, and prediction helpers for ViT.
