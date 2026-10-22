"""Export a trained model to ONNX for the browser app, and check and time the export.

Writes:
    models/<run>/model.onnx       input "input" (1x3xSxS, ImageNet-normalised RGB),
                                  output "logits" (1x8)
    models/<run>/model.json       what the app needs: outputs, input size, normalisation,
                                  temperature and threshold (from results/eval/<run>.json)
    results/export/<run>.json     ONNX size, max |PyTorch - ONNX| difference, CPU latency

Latency here is ONNX Runtime on this computer's CPU from Python. It is not the app's
latency: that gets measured in the browser on the phone (milestone M4).

Usage (from ml/):
    uv run python -m rupiahvision.training.export --run mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from rupiahvision import REPO_ROOT
from rupiahvision.training.data import IMAGENET_MEAN, IMAGENET_STD, OUTPUTS
from rupiahvision.training.evaluate import EVAL_DIR, load_checkpoint
from rupiahvision.training.train import MODELS_DIR

EXPORT_DIR = REPO_ROOT / "results" / "export"
WEB_MODEL_DIR = REPO_ROOT / "web" / "public" / "model"
OPSET = 18
PARITY_TOLERANCE = 1e-3


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def export_onnx(model: torch.nn.Module, image_size: int, path: Path) -> None:
    """One self-contained .onnx file (weights embedded), which the browser loads in one fetch."""
    dummy = torch.zeros(1, 3, image_size, image_size)
    path.with_name(path.name + ".data").unlink(missing_ok=True)  # left by older exports
    torch.onnx.export(
        model.eval().cpu(),
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
    )


def parity(model: torch.nn.Module, session: ort.InferenceSession, image_size: int) -> float:
    """Largest |PyTorch logit - ONNX logit| over a few random inputs."""
    rng = np.random.default_rng(0)
    worst = 0.0
    with torch.no_grad():
        for _ in range(5):
            x = rng.standard_normal((1, 3, image_size, image_size)).astype(np.float32)
            expected = model(torch.from_numpy(x)).numpy()
            actual = session.run(["logits"], {"input": x})[0]
            worst = max(worst, float(np.abs(expected - actual).max()))
    return worst


def latency_ms(session: ort.InferenceSession, image_size: int, runs: int = 100) -> dict:
    x = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    for _ in range(10):  # warm-up
        session.run(["logits"], {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(["logits"], {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(times)), "p95": float(np.percentile(times, 95)), "runs": runs}


def export(
    run: str,
    models_dir: Path = MODELS_DIR,
    eval_dir: Path = EVAL_DIR,
    export_dir: Path = EXPORT_DIR,
) -> dict:
    model, checkpoint = load_checkpoint(run, models_dir)
    size = checkpoint["image_size"]
    out_dir = models_dir / run
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, size, onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_diff = parity(model, session, size)
    if max_diff > PARITY_TOLERANCE:
        raise SystemExit(
            f"ONNX output differs from PyTorch by {max_diff:.2e} (tolerance {PARITY_TOLERANCE:.0e})"
        )

    eval_path = eval_dir / f"{run}.json"
    calibration = json.loads(eval_path.read_text())["calibration"] if eval_path.exists() else None
    metadata = {
        "run": run,
        "outputs": list(OUTPUTS),
        "input": {
            "name": "input",
            "size": size,
            "layout": "NCHW",
            "color": "RGB",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "temperature": calibration["temperature"] if calibration else 1.0,
        "threshold": calibration["threshold"] if calibration else None,
    }
    (out_dir / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    result = {
        "run": run,
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": OPSET,
        "max_abs_logit_difference": max_diff,
        "cpu_latency_ms": latency_ms(session, size),
        "measured_on": {
            "cpu": cpu_name(),
            "runtime": f"onnxruntime {ort.__version__} (Python, CPU)",
        },
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / f"{run}.json").write_text(json.dumps(result, indent=2) + "\n")
    lat = result["cpu_latency_ms"]
    print(
        f"{run}: {onnx_path} ({result['onnx_bytes'] / 1e6:.1f} MB), parity {max_diff:.1e}, "
        f"CPU latency median {lat['median']:.1f} ms, p95 {lat['p95']:.1f} ms"
    )
    return result


def publish(run: str, models_dir: Path = MODELS_DIR, web_dir: Path = WEB_MODEL_DIR) -> None:
    """Copy an exported model and its metadata into the web app (served as model/...)."""
    web_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "model.json"):
        shutil.copy2(models_dir / run / name, web_dir / name)
    print(f"Published {run} to {web_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a trained v1 model to ONNX.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--publish", action="store_true", help="also copy it into the web app")
    args = parser.parse_args(argv)
    export(args.run)
    if args.publish:
        publish(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
