"""Generate the README's tables from files in the repo, and check they are up to date.

Every table in README.md sits between a pair of markers:

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

Only this script writes between them. CI runs it with --check, which fails if the
README differs from what the source files (labels.json, results/*.json) say.

Usage:
    uv run python -m rupiahvision.reporting.readme_tables          # rewrite README.md
    uv run python -m rupiahvision.reporting.readme_tables --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rupiahvision import REPO_ROOT
from rupiahvision.labels import load_schema

BLOCK_RE = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def classes_table(repo_root: Path) -> str:
    schema = load_schema(repo_root / "data" / "labels.json")
    rows = [
        (
            rupiah(d),
            ", ".join(schema.series),
            schema.speech["id-ID"][str(d)],
            schema.speech["en-US"][str(d)],
        )
        for d in schema.denominations
    ]
    rows.append(
        (
            "(no note / not confident)",
            "-",
            schema.speech["id-ID"]["unknown"],
            schema.speech["en-US"]["unknown"],
        )
    )
    return markdown_table(["Denomination", "Series", "Spoken (id-ID)", "Spoken (en-US)"], rows)


# How independent a class's test images are; see rupiahvision.data.splits.
SEPARATION_LEVELS = {
    "note_disjoint": "note-disjoint",
    "val_by_session": "test note-disjoint, val by session",
    "session_only": "**session only**",
    "no_test_data": "no test images",
    "no_notes": "no notes yet",
}


def rupiah(amount: int) -> str:
    return f"Rp{amount:,}".replace(",", ".")


def dataset_table(repo_root: Path) -> str:
    summary = json.loads(
        (repo_root / "results" / "data" / "dataset_summary.json").read_text(encoding="utf-8")
    )
    rows = [
        (
            rupiah(c["denomination"]),
            c["series"],
            c["notes"],
            SEPARATION_LEVELS[c["level"]],
            c["own_images"]["train"],
            c["own_images"]["val"],
            c["own_images"]["test"],
            c["public_images"],
        )
        for c in summary["classes"]
    ]
    negatives = summary["negatives"]
    rows.append(
        (
            "unknown (no note)",
            "-",
            f"{negatives['objects']} objects",
            "-",
            negatives["images"]["train"],
            negatives["images"]["val"],
            negatives["images"]["test"],
            "-",
        )
    )
    totals = summary["total_images"]
    sessions = summary["sessions"]
    footer = (
        f"\nTotal: {totals['train']} train / {totals['val']} val / {totals['test']} test images "
        f"from {sessions['train']} / {sessions['val']} / {sessions['test']} capture sessions.\n"
    )
    headers = ["Denomination", "Series", "Notes", "Test separation", "Train", "Val", "Test",
               "Public (train only)"]  # fmt: skip
    return markdown_table(headers, rows) + footer


def not_measured(command: str) -> str:
    return f"_Not measured yet. Produced by `{command}`._\n"


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def public_dataset_table(repo_root: Path) -> str:
    summary = load_json(repo_root / "results" / "data" / "public_summary.json")
    if summary is None:
        return not_measured("python -m rupiahvision.public split")
    rows = [(label, *counts.values()) for label, counts in summary["labels"].items()]
    rows.append(("**total**", *summary["total"].values()))
    rows.append(("photo groups", *summary["groups"].values()))
    notes = "".join(f"\n- {note}" for note in summary["notes"])
    return markdown_table(["Label", "Train", "Val", "Test"], rows) + (
        f"\nNear-duplicate check:{notes}\n" if notes else ""
    )


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def pct_ci(value: float, ci: list[float]) -> str:
    return f"{pct(value)} ({pct(ci[0])}–{pct(ci[1])})"


def evaluated_runs(repo_root: Path) -> list[tuple[dict, dict | None, dict | None]]:
    """(eval, export, training run) for every evaluated model, ordered by parameter count."""
    out = []
    for path in sorted((repo_root / "results" / "eval").glob("*.json")):
        result = load_json(path)
        run = result["run"]
        out.append(
            (
                result,
                load_json(repo_root / "results" / "export" / f"{run}.json"),
                load_json(repo_root / "results" / "runs" / f"{run}.json"),
            )
        )
    return sorted(out, key=lambda t: t[0]["parameters"])


def said_on_no_note(test: dict) -> str:
    """False-announcement rate on no-note images, with the count it rests on."""
    n = next(c["n"] for c in test["per_class"] if c["label"] == "unknown")
    rate = test["selective"]["false_announcement_rate_on_unknown"]
    return f"{pct(rate)} ({round(rate * n)}/{n})"


def models_table(repo_root: Path) -> str:
    runs = evaluated_runs(repo_root)
    if not runs:
        return not_measured("python -m rupiahvision.training.evaluate --run <model>")
    rows = []
    for result, exported, trained in runs:
        test, selective = result["test"], result["test"]["selective"]
        latency = exported["cpu_latency_ms"] if exported else None
        rows.append(
            (
                f"`{result['run']}`",
                f"{result['parameters'] / 1e6:.2f} M",
                f"{exported['onnx_bytes'] / 1e6:.1f} MB" if exported else "not measured",
                pct(trained["best"]["val_accuracy"]) if trained else "not measured",
                pct_ci(test["accuracy"], test["accuracy_ci95"]),
                pct_ci(test["macro_accuracy"], test["macro_accuracy_ci95"]),
                f"{result['calibration']['threshold']:.2f}",
                pct(selective["coverage"]),
                f"{pct(selective['risk'])} ({selective['wrong_announcements']}"
                f"/{selective['announcements']})",
                said_on_no_note(test),
                f"{latency['median']:.1f} ({latency['p95']:.1f})" if latency else "not measured",
            )
        )
    headers = ["Model", "Params", "ONNX", "Val acc", "Test acc (95% CI)", "Macro acc (95% CI)",
               "Threshold", "Coverage", "Risk (wrong/said)", "Said on no-note",
               "CPU ms median (p95)"]  # fmt: skip
    cpu = next((e["measured_on"]["cpu"] for _, e, _ in runs if e), None)
    footer = f"\nCPU latency: ONNX Runtime from Python on {cpu}, one image.\n" if cpu else ""
    return markdown_table(headers, rows) + footer


def selected_run(repo_root: Path) -> dict | None:
    """The model with the best *validation* accuracy; test results never pick the model."""
    runs = [(r, t) for r, _, t in evaluated_runs(repo_root) if t]
    return max(runs, key=lambda rt: rt[1]["best"]["val_accuracy"])[0] if runs else None


def per_class_table(repo_root: Path) -> str:
    result = selected_run(repo_root)
    if result is None:
        return not_measured("python -m rupiahvision.training.evaluate --run <model>")
    rows = [
        (c["label"], c["n"], c["groups"], pct_ci(c["accuracy"], c["ci95"]) if c["n"] else "-")
        for c in result["test"]["per_class"]
    ]
    header = f"Selected model (best validation accuracy): `{result['run']}`.\n\n"
    confusion = f"\n![Confusion matrix](results/eval/{result['run']}_confusion.png)\n"
    return (
        header
        + markdown_table(["Label", "Test images", "Photo groups", "Accuracy (95% CI)"], rows)
        + confusion
    )


TABLES: dict[str, Callable[[Path], str]] = {
    "classes": classes_table,
    "dataset": dataset_table,
    "public_dataset": public_dataset_table,
    "models": models_table,
    "per_class": per_class_table,
}


def render(readme: str, tables: dict[str, str]) -> str:
    """Replace each generated block's body; every block and every table must match up."""
    found = [m.group("name") for m in BLOCK_RE.finditer(readme)]
    if len(found) != len(set(found)):
        raise ValueError(f"duplicate generated blocks in README: {found}")
    unknown = set(found) - set(tables)
    if unknown:
        raise ValueError(f"README has blocks with no generator: {sorted(unknown)}")
    missing = set(tables) - set(found)
    if missing:
        raise ValueError(f"README is missing generated blocks: {sorted(missing)}")
    return BLOCK_RE.sub(lambda m: m.group(1) + tables[m.group("name")] + m.group(4), readme)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is stale")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    readme_path = args.repo_root / "README.md"
    current = readme_path.read_text(encoding="utf-8")
    updated = render(current, {name: fn(args.repo_root) for name, fn in TABLES.items()})

    if args.check:
        if updated != current:
            print(
                "README.md is out of date. Run: "
                "uv run python -m rupiahvision.reporting.readme_tables",
                file=sys.stderr,
            )
            return 1
        print("README.md tables are up to date.")
        return 0

    if updated != current:
        readme_path.write_text(updated, encoding="utf-8")
        print(f"Updated {readme_path}")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
