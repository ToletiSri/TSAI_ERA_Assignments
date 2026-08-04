# Session 21 — Character-Level GPT

This assignment trains a decoder-only Transformer on Shakespeare text and exposes text generation through Gradio.

## Core concepts

- **Autoregressive language modeling:** the model predicts the next character from all previous characters in a fixed context window.
- **Causal self-attention:** a triangular mask prevents tokens from attending to the future.
- **Transformer blocks:** multi-head attention, feed-forward layers, residual connections, and layer normalization refine token representations.
- **Character tokenization:** a compact vocabulary maps the corpus characters to integer IDs and back.
- **Sampling:** generated logits are converted to probabilities and sampled repeatedly to extend a prompt.
- **Train/validation evaluation:** periodic average losses track generalization during training.
- **Interactive inference:** the Gradio notebook accepts a starting prompt and desired output length.

## Contents

- `gpt.py` implements the attention heads, Transformer blocks, and language model.
- `bigram.py` provides the simpler baseline used before the full GPT model.
- `config.py` contains model and training hyperparameters.
- `S21.ipynb` trains and samples from the model.
- `GradioInterface.ipynb` builds the user interface.
- `input.txt` contains the Shakespeare corpus.

Try the [Hugging Face Space](https://huggingface.co/spaces/ToletiSri/TSAI_S21).
