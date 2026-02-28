# yeen-gpt

A small GPT language model trained from scratch with BPE tokenization, built with PyTorch.

## Setup

```bash
uv sync
```

### CUDA (GPU) support

The default install pulls CPU-only PyTorch. To use a GPU, reinstall torch from the appropriate CUDA index:

```bash
# CUDA 12.8 (RTX 40/50 series)
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Usage

**Train:**

```bash
uv run python -m yeen_gpt train
uv run python -m yeen_gpt train --n-embd 512 --n-head 8 --dropout 0.3
```

**Generate from a saved checkpoint:**

```bash
uv run python -m yeen_gpt generate checkpoints/model.pt
uv run python -m yeen_gpt generate checkpoints/model.pt --prompt "ROMEO:" --max-tokens 1000
```

## Lint

```bash
uv run ruff check src/
uv run ruff format src/
```
