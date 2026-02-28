"""Hyperparameters and configuration."""

from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    """All hyperparameters in one place."""

    # data
    data_path: str = "data/input.txt"
    num_merges: int = 256  # BPE merge operations

    # model
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    block_size: int = 256
    dropout: float = 0.4

    # training
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 250
    eval_iters: int = 200
    learning_rate: float = 3e-4
    patience: int = 5  # early stopping evals without improvement
    seed: int = 1337

    # runtime
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # checkpoints
    save_path: str = "checkpoints/model.pt"
