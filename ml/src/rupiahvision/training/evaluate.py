"""Evaluate a trained model: calibration and "unknown" threshold on val, metrics on test.

Steps:
  1. Fit the temperature on val logits (metrics.fit_temperature).
  2. Choose the lowest threshold whose risk on val is at most --target-risk.
  3. On test, report accuracy and per-class accuracy with group-bootstrap 95% CIs, the
     confusion matrix, and coverage/risk at the chosen threshold.

Writes results/eval/<run>.json plus <run>_confusion.png and <run>_risk_coverage.png.

Usage (from ml/):
    uv run python -m rupiahvision.training.evaluate --run mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from rupiahvision import REPO_ROOT
from rupiahvision.data.inventory import DATA_DIR
from rupiahvision.training import metrics
from rupiahvision.training.data import OUTPUTS, ImageDataset, eval_transform, load_split_rows
from rupiahvision.training.models import build_model, count_parameters
from rupiahvision.training.train import MODELS_DIR, pick_device

EVAL_DIR = REPO_ROOT / "results" / "eval"
DEFAULT_TARGET_RISK = 0.01


def load_checkpoint(run: str, models_dir: Path = MODELS_DIR) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(models_dir / run / "checkpoint.pt", map_location="cpu")
    if tuple(checkpoint["outputs"]) != OUTPUTS:
        raise SystemExit(f"{run} was trained with different outputs: {checkpoint['outputs']}")
    model = build_model(checkpoint["model"], len(OUTPUTS), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


@torch.no_grad()
def predict_logits(model, rows, image_size, data_dir, device, batch_size=64) -> np.ndarray:
    dataset = ImageDataset(rows, eval_transform(image_size), data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    return np.concatenate([model(x.to(device)).float().cpu().numpy() for x, _ in loader])


def per_class(labels, preds, groups, seed) -> list[dict]:
    out = []
    for index, name in enumerate(OUTPUTS):
        mask = labels == index
        correct = (preds[mask] == index).astype(float)
        low, high = metrics.proportion_ci(correct, groups[mask], seed=seed)
        out.append(
            {
                "label": name,
                "n": int(mask.sum()),
                "groups": int(len(set(groups[mask]))),
                "accuracy": float(correct.mean()) if mask.any() else None,
                "ci95": [low, high] if mask.any() else None,
            }
        )
    return out


def plot_confusion(matrix: np.ndarray, path: Path, title: str) -> None:
    fig = Figure(figsize=(6.5, 5.5))
    ax = fig.subplots()
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ticks = range(len(OUTPUTS))
    ax.set_xticks(ticks, OUTPUTS, rotation=45, ha="right")
    ax.set_yticks(ticks, OUTPUTS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in ticks:
        for j in ticks:
            if matrix[i, j]:
                color = "white" if shares[i, j] > 0.5 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_risk_coverage(curve: dict, chosen: dict, path: Path, title: str) -> None:
    fig = Figure(figsize=(5.5, 4))
    ax = fig.subplots()
    ax.plot(curve["coverage"], curve["risk"], color="#1f5fa8")
    ax.scatter(
        [chosen["coverage"]],
        [chosen["risk"]],
        color="#c0392b",
        zorder=3,
        label=f"threshold {chosen['threshold']:.2f}",
    )
    ax.set_xlabel("coverage (share of notes announced)")
    ax.set_ylabel("risk (share of announcements wrong)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def evaluate(
    run: str,
    target_risk: float = DEFAULT_TARGET_RISK,
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
    eval_dir: Path = EVAL_DIR,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    device = device or pick_device()
    model, checkpoint = load_checkpoint(run, models_dir)
    size = checkpoint["image_size"]
    rows = load_split_rows(data_dir)
    limit = checkpoint["config"].get("max_images_per_split")
    if limit:
        rows = {s: r[:limit] for s, r in rows.items()}

    def labels_of(split):
        return np.array([OUTPUTS.index(r["label"]) for r in rows[split]])

    val_logits = predict_logits(model, rows["val"], size, data_dir, device)
    temperature = metrics.fit_temperature(val_logits, labels_of("val"))
    val_probs = metrics.softmax(val_logits, temperature)
    threshold = metrics.choose_threshold(val_probs, labels_of("val"), target_risk)

    test_labels = labels_of("test")
    test_groups = np.array([r["group"] for r in rows["test"]])
    test_probs = metrics.softmax(
        predict_logits(model, rows["test"], size, data_dir, device), temperature
    )
    preds = test_probs.argmax(axis=1)
    correct = (preds == test_labels).astype(float)
    classes = per_class(test_labels, preds, test_groups, seed)

    def macro_accuracy(indices: np.ndarray) -> float:
        """Mean of per-class accuracies over the classes present in `indices`."""
        labels = test_labels[indices]
        return float(np.mean([correct[indices][labels == c].mean() for c in np.unique(labels)]))

    everything = np.arange(len(correct))
    # Widened by Wilson-over-groups like the other CIs (see metrics.proportion_ci).
    boot = metrics.cluster_bootstrap_ci(everything, test_groups, macro_accuracy, seed=seed)
    n_groups = len(set(test_groups))
    wilson = metrics.wilson_ci(macro_accuracy(everything) * n_groups, n_groups)
    macro_ci = (min(boot[0], wilson[0]), max(boot[1], wilson[1]))
    matrix = metrics.confusion_matrix(test_labels, preds, len(OUTPUTS))
    selective = metrics.selective_metrics(test_probs, test_labels, threshold)
    val_selective = metrics.selective_metrics(val_probs, labels_of("val"), threshold)

    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(matrix, eval_dir / f"{run}_confusion.png", f"{run}: test confusion matrix")
    plot_risk_coverage(
        metrics.risk_coverage_curve(test_probs, test_labels),
        selective,
        eval_dir / f"{run}_risk_coverage.png",
        f"{run}: test risk vs coverage",
    )

    result = {
        "run": run,
        "model": checkpoint["model"],
        "parameters": count_parameters(model),
        "image_size": size,
        "outputs": list(OUTPUTS),
        "images": {"val": len(rows["val"]), "test": len(rows["test"])},
        "test_groups": int(len(set(test_groups))),
        "calibration": {
            "temperature": temperature,
            "target_risk": target_risk,
            "threshold": threshold,
            "val": val_selective,
        },
        "test": {
            "accuracy": float(correct.mean()),
            "accuracy_ci95": list(metrics.proportion_ci(correct, test_groups, seed=seed)),
            "macro_accuracy": macro_accuracy(everything),
            "macro_accuracy_ci95": list(macro_ci),
            "per_class": classes,
            "confusion_matrix": matrix.tolist(),
            "selective": selective,
        },
    }
    (eval_dir / f"{run}.json").write_text(json.dumps(result, indent=2) + "\n")
    t = result["test"]
    print(
        f"{run}: test accuracy {t['accuracy']:.3f} (95% CI {t['accuracy_ci95'][0]:.3f}-"
        f"{t['accuracy_ci95'][1]:.3f}), macro {t['macro_accuracy']:.3f}; at threshold "
        f"{threshold:.3f}: coverage {selective['coverage']:.3f}, risk {selective['risk']:.3f}"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained v1 model.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--target-risk", type=float, default=DEFAULT_TARGET_RISK)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    evaluate(args.run, args.target_risk, device=pick_device(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
