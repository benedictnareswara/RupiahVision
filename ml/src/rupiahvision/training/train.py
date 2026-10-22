"""Train one model on the v1 public dataset.

Keeps the checkpoint with the best validation accuracy. Writes:
    models/<run>/checkpoint.pt      weights + everything needed to rebuild the model (git-ignored)
    results/runs/<run>.json         config and per-epoch history (committed)

Usage (from ml/):
    uv run python -m rupiahvision.training.train --model mobilenet_v3_small
    uv run python -m rupiahvision.training.train --model custom_cnn --epochs 40 --lr 3e-3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rupiahvision import REPO_ROOT
from rupiahvision.data.inventory import DATA_DIR
from rupiahvision.training.data import (
    OUTPUTS,
    ImageDataset,
    eval_transform,
    load_split_rows,
    train_transform,
)
from rupiahvision.training.models import (
    MODEL_NAMES,
    build_model,
    count_parameters,
    parameter_groups,
)

MODELS_DIR = REPO_ROOT / "models"
RUNS_DIR = REPO_ROOT / "results" / "runs"


@dataclass(frozen=True)
class TrainConfig:
    model: str
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    image_size: int = 224
    pretrained: bool = True
    seed: int = 0
    workers: int = 4
    max_images_per_split: int | None = None  # for quick smoke tests only

    def __post_init__(self):
        if self.model == "custom_cnn":  # trained from scratch: no ImageNet weights exist
            object.__setattr__(self, "pretrained", False)

    @property
    def run_name(self) -> str:
        return self.model


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights (mean 1), so rare classes count as much as common ones."""
    counts = Counter(targets)
    weights = torch.tensor([1.0 / max(counts[c], 1) for c in range(num_classes)])
    return weights * num_classes / weights.sum()


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> tuple[float, float]:
    """(mean loss, accuracy); trains if an optimizer is given, else just evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            seen += len(targets)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(
    config: TrainConfig,
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    runs_dir: Path = RUNS_DIR,
    device: torch.device | None = None,
) -> dict:
    seed_everything(config.seed)
    device = device or pick_device()
    rows = load_split_rows(data_dir)
    if config.max_images_per_split:
        rows = {s: r[: config.max_images_per_split] for s, r in rows.items()}
    train_set = ImageDataset(rows["train"], train_transform(config.image_size), data_dir)
    val_set = ImageDataset(rows["val"], eval_transform(config.image_size), data_dir)
    if not len(train_set) or not len(val_set):
        raise SystemExit("train or val split is empty; run `python -m rupiahvision.public split`")

    generator = torch.Generator().manual_seed(config.seed)
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=config.workers,
        persistent_workers=config.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = build_model(config.model, len(OUTPUTS), config.pretrained).to(device)
    scale = config.backbone_lr_scale if config.pretrained else 1.0
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.lr, scale), weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(train_set.targets, len(OUTPUTS)).to(device))

    out_dir = models_dir / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], {"val_accuracy": -1.0, "epoch": 0}
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(
            f"epoch {epoch:3d}/{config.epochs}  train loss {train_loss:.3f} acc {train_acc:.3f}  "
            f"val loss {val_loss:.3f} acc {val_acc:.3f}",
            flush=True,
        )
        if val_acc > best["val_accuracy"]:
            best = {"val_accuracy": val_acc, "epoch": epoch}
            torch.save(
                {
                    "model": config.model,
                    "outputs": OUTPUTS,
                    "image_size": config.image_size,
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                },
                out_dir / "checkpoint.pt",
            )

    record = {
        "run": config.run_name,
        "config": asdict(config),
        "device": device.type,
        "parameters": count_parameters(model),
        "images": {"train": len(train_set), "val": len(val_set)},
        "best": best,
        "train_minutes": round((time.perf_counter() - started) / 60, 1),
        "history": history,
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{config.run_name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"best val accuracy {best['val_accuracy']:.3f} at epoch {best['epoch']}; "
        f"checkpoint in {out_dir}"
    )
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a v1 model.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    defaults = TrainConfig(model="custom_cnn")
    for name in ("epochs", "batch_size", "image_size", "seed", "workers"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name)
        )
    for name in ("lr", "backbone_lr_scale", "weight_decay"):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--device", help="cpu, mps or cuda (default: best available)")
    args = vars(parser.parse_args(argv))
    device = pick_device(args.pop("device"))
    train(TrainConfig(**args), device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
