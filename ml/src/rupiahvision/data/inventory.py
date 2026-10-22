"""Load and validate the hand-written data inventory in data/.

Hand-written, one row each (columns and rules in docs/data-collection.md):
    notes.csv      a physical banknote
    negatives.csv  a non-banknote object
    sessions.csv   a capture session (one location and lighting, one day, one split)
    clips.csv      a recorded video or photo

Generated:
    manifest.csv   one row per image        (see MANIFEST_COLUMNS)

Every problem found is collected and raised together in an InventoryError, so one run
lists everything that needs fixing instead of stopping at the first bad row.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from rupiahvision import REPO_ROOT
from rupiahvision.labels import LabelSchema, load_schema

DATA_DIR = REPO_ROOT / "data"
SPLITS = ("train", "val", "test")
EMPTY_SUBJECT = "EMPTY"  # a scene with no object at all; allowed in any split
NOT_APPLICABLE = "-"  # side/state for anything that isn't a banknote
OWN_SOURCE = "own"
DEVICES = ("iphone", "webcam", "other")

NOTE_ID = re.compile(r"N\d{3,}")
OBJECT_ID = re.compile(r"O\d{3,}")
SESSION_ID = re.compile(r"S\d{3,}")
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]*")
DHASH = re.compile(r"[0-9a-f]{16}")

NOTE_COLUMNS = ("note_id", "denomination", "series", "wear", "owner", "comment")
NEGATIVE_COLUMNS = ("object_id", "object_type", "comment")
SESSION_COLUMNS = ("session_id", "split", "date", "location", "device", "comment")
CLIP_COLUMNS = ("file", "session_id", "subject_id", "side", "state", "lighting", "background")
MANIFEST_COLUMNS = (
    "path", "label", "source", "session_id", "subject_id",
    "side", "state", "lighting", "background", "dhash",
)  # fmt: skip


class InventoryError(ValueError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s):\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class Note:
    note_id: str
    denomination: int
    series: str
    wear: str
    owner: str
    comment: str

    @property
    def group(self) -> str:
        """Notes of the same denomination and series compete for the same split roles."""
        return f"{self.denomination}_{self.series}"


@dataclass(frozen=True)
class NegativeObject:
    object_id: str
    object_type: str
    comment: str

    @property
    def group(self) -> str:
        return f"negative:{self.object_type}"


@dataclass(frozen=True)
class Session:
    session_id: str
    split: str
    date: date
    location: str
    device: str
    comment: str


@dataclass(frozen=True)
class Clip:
    file: str
    session_id: str
    subject_id: str
    side: str
    state: str
    lighting: str
    background: str


@dataclass(frozen=True)
class ManifestRow:
    path: str
    label: str
    source: str  # "own", or "public:<dataset>" for images we did not take
    session_id: str
    subject_id: str
    side: str
    state: str
    lighting: str
    background: str
    dhash: str

    @property
    def is_own(self) -> bool:
        return self.source == OWN_SOURCE


@dataclass(frozen=True)
class Inventory:
    notes: dict[str, Note]
    negatives: dict[str, NegativeObject]
    sessions: dict[str, Session]

    def group_of(self, subject_id: str) -> str | None:
        """Split group of a note or object; None for EMPTY scenes."""
        if subject_id in self.notes:
            return self.notes[subject_id].group
        if subject_id in self.negatives:
            return self.negatives[subject_id].group
        if subject_id == EMPTY_SUBJECT:
            return None
        raise KeyError(f"unknown subject {subject_id!r}")

    def groups(self) -> dict[str, list[str]]:
        """Group -> sorted subject ids, for every note and negative object."""
        out: dict[str, list[str]] = {}
        for subject_id, subject in [*self.notes.items(), *self.negatives.items()]:
            out.setdefault(subject.group, []).append(subject_id)
        return {group: sorted(ids) for group, ids in sorted(out.items())}


def read_csv(path: Path, columns: tuple[str, ...], *, required: bool = True) -> list[dict]:
    """Rows of a CSV whose header must be exactly `columns`. Blank rows are skipped."""
    if not path.exists():
        if required:
            raise InventoryError([f"{path.name}: file not found at {path}"])
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != columns:
            raise InventoryError(
                [f"{path.name}: header must be {','.join(columns)}, got {reader.fieldnames}"]
            )
        return [
            {k: (v or "").strip() for k, v in row.items()}
            for row in reader
            if any((v or "").strip() for v in row.values())
        ]


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _check_unique(rows: list[dict], key: str, file: str, problems: list[str]) -> None:
    seen: set[str] = set()
    for line, row in enumerate(rows, start=2):
        if row[key] in seen:
            problems.append(f"{file}:{line}: duplicate {key} {row[key]!r}")
        seen.add(row[key])


def _choice(value: str, allowed: tuple, what: str, where: str, problems: list[str]) -> None:
    if value not in allowed:
        problems.append(f"{where}: {what} {value!r} must be one of {', '.join(map(str, allowed))}")


def load_inventory(data_dir: Path = DATA_DIR, schema: LabelSchema | None = None) -> Inventory:
    schema = schema or load_schema(data_dir / "labels.json")
    problems: list[str] = []

    note_rows = read_csv(data_dir / "notes.csv", NOTE_COLUMNS)
    _check_unique(note_rows, "note_id", "notes.csv", problems)
    notes = {}
    for line, row in enumerate(note_rows, start=2):
        where = f"notes.csv:{line}"
        if not NOTE_ID.fullmatch(row["note_id"]):
            problems.append(f"{where}: note_id {row['note_id']!r} must look like N001")
        denom = int(row["denomination"]) if row["denomination"].isdigit() else row["denomination"]
        _choice(denom, schema.denominations, "denomination", where, problems)
        _choice(row["series"], schema.series, "series", where, problems)
        _choice(row["wear"], schema.conditions["wear"], "wear", where, problems)
        notes[row["note_id"]] = Note(
            row["note_id"], denom, row["series"], row["wear"], row["owner"], row["comment"]
        )

    negative_rows = read_csv(data_dir / "negatives.csv", NEGATIVE_COLUMNS)
    _check_unique(negative_rows, "object_id", "negatives.csv", problems)
    negatives = {}
    for line, row in enumerate(negative_rows, start=2):
        where = f"negatives.csv:{line}"
        if not OBJECT_ID.fullmatch(row["object_id"]):
            problems.append(f"{where}: object_id {row['object_id']!r} must look like O001")
        if not SLUG.fullmatch(row["object_type"]):
            problems.append(f"{where}: object_type {row['object_type']!r} must be a lowercase slug")
        negatives[row["object_id"]] = NegativeObject(
            row["object_id"], row["object_type"], row["comment"]
        )

    session_rows = read_csv(data_dir / "sessions.csv", SESSION_COLUMNS)
    _check_unique(session_rows, "session_id", "sessions.csv", problems)
    sessions = {}
    for line, row in enumerate(session_rows, start=2):
        where = f"sessions.csv:{line}"
        if not SESSION_ID.fullmatch(row["session_id"]):
            problems.append(f"{where}: session_id {row['session_id']!r} must look like S001")
        _choice(row["split"], SPLITS, "split", where, problems)
        _choice(row["device"], DEVICES, "device", where, problems)
        if not SLUG.fullmatch(row["location"]):
            problems.append(f"{where}: location {row['location']!r} must be a lowercase slug")
        try:
            day = date.fromisoformat(row["date"])
        except ValueError:
            problems.append(f"{where}: date {row['date']!r} must be YYYY-MM-DD")
            day = date.min
        sessions[row["session_id"]] = Session(
            row["session_id"], row["split"], day, row["location"], row["device"], row["comment"]
        )

    if problems:
        raise InventoryError(problems)
    return Inventory(notes, negatives, sessions)


def load_clips(
    inventory: Inventory, data_dir: Path = DATA_DIR, schema: LabelSchema | None = None
) -> list[Clip]:
    schema = schema or load_schema(data_dir / "labels.json")
    rows = read_csv(data_dir / "clips.csv", CLIP_COLUMNS)
    problems: list[str] = []
    _check_unique(rows, "file", "clips.csv", problems)
    for line, row in enumerate(rows, start=2):
        where = f"clips.csv:{line}"
        if row["session_id"] not in inventory.sessions:
            problems.append(f"{where}: session {row['session_id']!r} is not in sessions.csv")
        subject = row["subject_id"]
        if subject in inventory.notes:
            _choice(row["side"], schema.sides, "side", where, problems)
            _choice(row["state"], schema.conditions["state"], "state", where, problems)
        elif subject in inventory.negatives or subject == EMPTY_SUBJECT:
            for column in ("side", "state"):
                if row[column] != NOT_APPLICABLE:
                    problems.append(f"{where}: {column} must be '-' for non-banknote {subject}")
        else:
            problems.append(f"{where}: subject {subject!r} is not in notes.csv or negatives.csv")
        _choice(row["lighting"], schema.conditions["lighting"], "lighting", where, problems)
        _choice(row["background"], schema.conditions["background"], "background", where, problems)
    if problems:
        raise InventoryError(problems)
    return [Clip(**row) for row in rows]


def load_manifest(path: Path = DATA_DIR / "manifest.csv") -> list[ManifestRow]:
    """Generated image manifest; an absent file means no images yet."""
    rows = read_csv(path, MANIFEST_COLUMNS, required=False)
    problems = [
        f"manifest.csv:{line}: dhash {row['dhash']!r} must be 16 hex digits"
        for line, row in enumerate(rows, start=2)
        if not DHASH.fullmatch(row["dhash"])
    ]
    _check_unique(rows, "path", "manifest.csv", problems)
    if problems:
        raise InventoryError(problems)
    return [ManifestRow(**row) for row in rows]
