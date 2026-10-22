"""Label schema shared with the web app (data/labels.json).

The model is trained on fine-grained classes (denomination x series x side, plus
"unknown"); the app only ever announces a denomination. `LabelSchema.aggregate`
does that reduction by summing class probabilities per denomination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from rupiahvision import REPO_ROOT

DEFAULT_SCHEMA_PATH = REPO_ROOT / "data" / "labels.json"


def class_name(denomination: int, series: str, side: str) -> str:
    return f"{denomination}_{series}_{side}"


@dataclass(frozen=True)
class LabelSchema:
    unknown_class: str
    denominations: tuple[int, ...]
    series: tuple[str, ...]
    sides: tuple[str, ...]
    classes: tuple[str, ...]
    speech: dict[str, dict[str, str]]
    conditions: dict[str, tuple[str, ...]]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def outputs(self) -> tuple[str, ...]:
        """What the app can announce: "unknown" followed by each denomination."""
        return (self.unknown_class, *(str(d) for d in self.denominations))

    def index_of(self, name: str) -> int:
        return self.classes.index(name)

    def denomination_of(self, name: str) -> int | None:
        """Denomination of a class name, or None for the unknown class."""
        if name == self.unknown_class:
            return None
        if name not in self.classes:
            raise KeyError(f"unknown class name: {name!r}")
        return int(name.split("_", 1)[0])

    def aggregation_matrix(self) -> np.ndarray:
        """(num_classes, len(outputs)) 0/1 matrix mapping classes to announceable outputs."""
        matrix = np.zeros((self.num_classes, len(self.outputs)), dtype=np.float32)
        for i, name in enumerate(self.classes):
            denom = self.denomination_of(name)
            matrix[i, 0 if denom is None else self.denominations.index(denom) + 1] = 1.0
        return matrix

    def aggregate(self, probs: np.ndarray) -> np.ndarray:
        """Sum class probabilities (..., num_classes) into outputs (..., len(outputs))."""
        probs = np.asarray(probs, dtype=np.float32)
        if probs.shape[-1] != self.num_classes:
            raise ValueError(f"expected last dim {self.num_classes}, got {probs.shape[-1]}")
        return probs @ self.aggregation_matrix()


def expected_classes(
    unknown_class: str,
    denominations: tuple[int, ...],
    series: tuple[str, ...],
    sides: tuple[str, ...],
) -> tuple[str, ...]:
    """Canonical class order: unknown first, then denomination-major, series, side."""
    return (
        unknown_class,
        *(class_name(d, s, side) for d, s, side in product(denominations, series, sides)),
    )


def load_schema(path: Path | str = DEFAULT_SCHEMA_PATH) -> LabelSchema:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = LabelSchema(
        unknown_class=raw["unknown_class"],
        denominations=tuple(raw["denominations"]),
        series=tuple(raw["series"]),
        sides=tuple(raw["sides"]),
        classes=tuple(raw["classes"]),
        speech=raw["speech"],
        conditions={k: tuple(v) for k, v in raw["conditions"].items()},
    )
    _validate(schema)
    return schema


def _validate(schema: LabelSchema) -> None:
    # The class order is baked into every trained model's output layer, so it must
    # match the canonical order exactly rather than merely contain the same names.
    expected = expected_classes(
        schema.unknown_class, schema.denominations, schema.series, schema.sides
    )
    if schema.classes != expected:
        raise ValueError("labels.json 'classes' does not match the canonical class order")
    for lang, phrases in schema.speech.items():
        missing = set(schema.outputs) - set(phrases)
        if missing:
            raise ValueError(f"speech[{lang!r}] is missing phrases for {sorted(missing)}")
