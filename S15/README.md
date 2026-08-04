# Session 15 — Transformer for English–Italian Translation

This assignment implements an encoder–decoder Transformer from scratch and trains it for neural machine translation with PyTorch Lightning.

## Core concepts

- Word-level source and target tokenizers with beginning, end, padding, and unknown tokens.
- Token embeddings plus sinusoidal positional encodings.
- Scaled dot-product multi-head attention.
- Encoder self-attention and decoder masked self-attention/cross-attention.
- Position-wise feed-forward networks, residual connections, normalization, and dropout.
- Teacher forcing with causal and padding masks.
- Autoregressive decoding and checkpoint-based training.

## Implementation map

- `model.py` contains the complete Transformer and its building blocks.
- `dataset.py` prepares bilingual sequences and attention masks.
- `LightningModel.py` manages tokenizers, data loaders, optimization, training, and validation.
- `config.py` centralizes dataset, model, and checkpoint settings.
- `S15.ipynb` launches the training workflow.

The session provides the baseline architecture that is optimized and adapted to English–French translation in Session 16.
