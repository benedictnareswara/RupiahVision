"""Builders for small, fake data/ directories so tests never need real images."""

import json
import random
import shutil

import pytest

from rupiahvision import REPO_ROOT
from rupiahvision.data.inventory import (
    CLIP_COLUMNS,
    MANIFEST_COLUMNS,
    NEGATIVE_COLUMNS,
    NOTE_COLUMNS,
    SESSION_COLUMNS,
    write_csv,
)

SPLIT_CONFIG = {
    "seed": 7,
    "val_fraction": 0.2,
    "test_fraction": 0.2,
    "open_set_object_types": [],
    "max_dhash_distance": 4,
}


def note(note_id, denomination=50000, series="TE2022", wear="used"):
    return dict(
        note_id=note_id, denomination=str(denomination), series=series, wear=wear, owner="me",
        comment="",
    )  # fmt: skip


def negative(object_id, object_type="receipt"):
    return dict(object_id=object_id, object_type=object_type, comment="")


def session(session_id, split, location="home_desk"):
    return dict(
        session_id=session_id, split=split, date="2026-10-24", location=location,
        device="iphone", comment="",
    )  # fmt: skip


def image(path, label, subject_id, session_id, dhash, side="front", source="own"):
    not_note = label == "unknown"
    return dict(
        path=path, label=label, source=source, session_id=session_id, subject_id=subject_id,
        side="-" if not_note else side, state="-" if not_note else "flat",
        lighting="daylight", background="plain", dhash=dhash,
    )  # fmt: skip


def distinct_hash(i: int) -> str:
    """Seeded random 64-bit hash. Two of these differ in ~32 bits, so they never look like
    near-duplicates (P(distance <= 4) is about 1e-13 per pair)."""
    return f"{random.Random(i).getrandbits(64):016x}"


class DataDir:
    def __init__(self, root):
        self.root = root
        shutil.copy(REPO_ROOT / "data" / "labels.json", root / "labels.json")
        self.config(**SPLIT_CONFIG)
        self.write(notes=[], negatives=[], sessions=[], clips=[])

    def config(self, **values):
        (self.root / "split_config.json").write_text(json.dumps({**SPLIT_CONFIG, **values}))

    def write(self, *, notes=None, negatives=None, sessions=None, clips=None, manifest=None):
        files = {
            "notes.csv": (NOTE_COLUMNS, notes),
            "negatives.csv": (NEGATIVE_COLUMNS, negatives),
            "sessions.csv": (SESSION_COLUMNS, sessions),
            "clips.csv": (CLIP_COLUMNS, clips),
            "manifest.csv": (MANIFEST_COLUMNS, manifest),
        }
        for name, (columns, rows) in files.items():
            if rows is not None:
                write_csv(self.root / name, columns, rows)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    return DataDir(root)
