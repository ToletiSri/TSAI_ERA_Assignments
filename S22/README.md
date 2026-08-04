# Session 22 — Large-Scale GPT Pretraining

This assignment develops a GPT pretraining loop designed for efficient distributed execution with Lightning Fabric.

## Core concepts

- **Decoder-only Transformer:** causal self-attention and MLP blocks perform next-token prediction.
- **Packed datasets:** memory-mapped token shards reduce padding and keep the input pipeline efficient.
- **Distributed training:** Lightning Fabric abstracts device placement, process launch, and collective operations.
- **Fully sharded data parallelism (FSDP):** model parameters and training state can be sharded across accelerators.
- **Mixed precision and fused optimization:** lower-precision arithmetic and optimized AdamW improve throughput.
- **Learning-rate scheduling:** linear warm-up is followed by cosine decay.
- **Operational tooling:** checkpoint resume, validation, FLOP estimation, speed monitoring, and CSV logging make long training runs observable and recoverable.

## Contents

- `main.ipynb` defines setup, training, validation, data-loader, and scheduler routines.
- `tsai_gpt/model.py` defines GPT configuration and architecture.
- `tsai_gpt/packed_dataset.py` streams packed binary datasets.
- `tsai_gpt/speed_monitor.py` reports throughput and utilization.
- `tsai_gpt/tokenizer.py`, `rmsnorm.py`, and `utils.py` provide supporting components.
