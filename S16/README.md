# Session 16 — Accelerated English–French Translation

This session extends the encoder–decoder Transformer from Session 15 and trains it on the English–French subset of `opus_books` using PyTorch Lightning.

## Core concepts

- **Transformer encoder–decoder:** self-attention encodes the source sentence, while masked self-attention and cross-attention generate the translation.
- **Multi-head attention:** parallel attention heads learn complementary token relationships.
- **Token and positional embeddings:** learned word representations are combined with sinusoidal position information.
- **Causal masking:** the decoder cannot inspect future target tokens during teacher-forced training.
- **Bilingual tokenization:** separate word-level tokenizers and special tokens prepare source and target sequences.
- **Sequence filtering and padding:** long or highly imbalanced sentence pairs are removed, then batches are padded to fixed lengths.
- **Training acceleration:** Lightning, mixed precision, manual optimization, and One Cycle learning-rate scheduling streamline training.

## Implementation map

- `model.py` implements attention, encoder/decoder blocks, normalization, residual connections, and projection to the target vocabulary.
- `dataset.py` builds padded bilingual examples and encoder/decoder masks.
- `LightningModel.py` loads data, trains tokenizers, performs training and validation, and prints sample translations.
- `config.py` stores language, sequence-length, checkpoint, and optimization settings.
- `S16.ipynb` configures and launches the Lightning trainer.

## Recorded result

Training reached a mean loss of **1.6602** at epoch 41. Validation examples in the notebook show autoregressive French translations generated from English source sentences.
