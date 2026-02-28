"""Data loading, encoding, and batching utilities."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import Config


def load_text(path: str | Path = "data/input.txt") -> str:
    """Read the training corpus from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# BPE helpers
# ---------------------------------------------------------------------------

def _get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count every adjacent pair in *ids*."""
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of *pair* in *ids* with *new_id*."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    """Byte-Pair Encoding tokenizer trained on a corpus string."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab_size: int = 256

    # ----- training --------------------------------------------------------

    def train(self, text: str, n_merges: int = 256) -> None:
        """Learn *n_merges* BPE merges from *text*."""
        ids = list(text.encode("utf-8"))
        for i in range(n_merges):
            counts = _get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)  # type: ignore[arg-type]
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.vocab_size = 256 + len(self.merges)

    # ----- encoding --------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* into a list of BPE token ids."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = _get_pair_counts(ids)
            pair = min(
                counts,
                key=lambda p: self.merges.get(p, float("inf")),  # type: ignore[arg-type]
            )
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    # ----- decoding --------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of BPE token ids back to a string."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ----- serialisation ---------------------------------------------------

    def state_dict(self) -> dict:
        """Return serialisable state."""
        return {
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "vocab_size": self.vocab_size,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a previously saved state dict."""
        self.merges = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in state["merges"].items()
        }
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.vocab_size = state["vocab_size"]


class Dataset:
    """Holds train/val splits and provides batching."""

    def __init__(self, cfg: Config, tokenizer: Tokenizer, text: str) -> None:
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self._cfg = cfg

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a small batch of inputs *x* and targets *y*."""
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self._cfg.block_size, (self._cfg.batch_size,))
        x = torch.stack([data[i : i + self._cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self._cfg.block_size + 1] for i in ix])
        return x.to(self._cfg.device), y.to(self._cfg.device)
