# Session 27 — Parameter-Efficient Fine-Tuning of Phi-2

This assignment fine-tunes Microsoft's Phi-2 causal language model on an instruction-style dataset and evaluates the resulting adapter.

## Core concepts

- **Supervised fine-tuning (SFT):** prompt/response examples are formatted as training text for causal language modeling.
- **Quantization:** bitsandbytes loads the base model at reduced precision to lower accelerator memory requirements.
- **LoRA/PEFT:** small trainable low-rank adapters update model behavior while the large base model remains frozen.
- **Trainer orchestration:** TRL's `SFTTrainer` integrates tokenization, batching, training arguments, and adapter training.
- **Experiment tracking:** Weights & Biases records fine-tuning metrics.
- **Adapter inference:** the saved adapter is loaded with the base model and tested on unseen questions.

## Contents

`s27-phi2-openaidataset.ipynb` installs the required libraries, prepares the dataset, configures quantization and LoRA, trains the adapter, and compares generated responses during inference.
