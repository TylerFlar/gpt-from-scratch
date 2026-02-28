"""Training, generation, and CLI entry point."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path

import torch

from .config import Config
from .data import Dataset, Tokenizer, load_text
from .model import GPTLanguageModel


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    cfg: Config,
    model: GPTLanguageModel,
    tokenizer: Tokenizer,
) -> None:
    """Persist model weights, tokenizer state, and config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str | Path,
    device: str | None = None,
) -> tuple[Config, GPTLanguageModel, Tokenizer]:
    """Load a previously saved checkpoint."""
    data = torch.load(path, map_location=device or "cpu", weights_only=False)
    cfg = Config(**data["config"])
    if device is not None:
        cfg.device = device

    tokenizer = Tokenizer()
    tokenizer.load_state_dict(data["tokenizer"])

    model = GPTLanguageModel(cfg, tokenizer.vocab_size)
    model.load_state_dict(data["model_state_dict"])
    model.to(cfg.device)
    model.eval()
    return cfg, model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    cfg: Config,
    model: GPTLanguageModel,
    dataset: Dataset,
) -> dict[str, float]:
    """Estimate mean loss on train and val splits."""
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = dataset.get_batch(split)
            _logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: Config) -> None:
    """Train a GPT language model."""
    torch.manual_seed(cfg.seed)

    # --- data ---
    text = load_text(cfg.data_path)
    tokenizer = Tokenizer()
    print("Training BPE tokenizer...")
    tokenizer.train(text, n_merges=cfg.num_merges)
    print(f"Vocab size: {tokenizer.vocab_size}")
    dataset = Dataset(cfg, tokenizer, text)

    # --- model ---
    model = GPTLanguageModel(cfg, tokenizer.vocab_size).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params / 1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # --- training loop ---
    best_val_loss = float("inf")
    best_state: dict | None = None
    evals_without_improvement = 0

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
            losses = estimate_loss(cfg, model, dataset)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
                if evals_without_improvement >= cfg.patience:
                    print(
                        f"Early stopping at step {step} "
                        f"(best val loss {best_val_loss:.4f})"
                    )
                    break

        xb, yb = dataset.get_batch("train")
        _logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss {best_val_loss:.4f})")

    # --- save ---
    save_checkpoint(cfg.save_path, cfg, model, tokenizer)

    # --- sample ---
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
    generated_ids = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_ids))


# ---------------------------------------------------------------------------
# Generation from a saved checkpoint
# ---------------------------------------------------------------------------

def generate(
    checkpoint: str,
    prompt: str = "",
    max_tokens: int = 500,
    device: str | None = None,
) -> None:
    """Load a checkpoint and generate text."""
    cfg, model, tokenizer = load_checkpoint(checkpoint, device=device)

    if prompt:
        ids = tokenizer.encode(prompt)
        context = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)

    generated_ids = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    print(tokenizer.decode(generated_ids))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yeen-gpt",
        description="Train or generate with a small GPT language model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- train ----
    tp = sub.add_parser("train", help="Train a new model")
    tp.add_argument("--data", dest="data_path", default=None, help="Path to training text file")
    tp.add_argument("--save", dest="save_path", default=None, help="Path to save checkpoint")

    # hyperparams – auto-generate flags from Config fields
    for f in fields(Config):
        if f.name in ("data_path", "save_path", "device"):
            continue
        flag = f"--{f.name.replace('_', '-')}"
        tp.add_argument(flag, dest=f.name, type=f.type, default=None, help=f"{f.name} (default: {f.default})")

    tp.add_argument("--device", default=None, help="Device (default: auto)")

    # ---- generate ----
    gp = sub.add_parser("generate", help="Generate text from a saved checkpoint")
    gp.add_argument("checkpoint", help="Path to a saved .pt checkpoint")
    gp.add_argument("--prompt", "-p", default="", help="Optional prompt text")
    gp.add_argument("--max-tokens", "-n", type=int, default=500, help="Number of tokens to generate")
    gp.add_argument("--device", default=None, help="Device (default: auto)")

    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cfg = Config()
        # override with any CLI-provided values
        for f in fields(Config):
            val = getattr(args, f.name, None)
            if val is not None:
                setattr(cfg, f.name, val)
        train(cfg)

    elif args.command == "generate":
        generate(
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            device=args.device,
        )
