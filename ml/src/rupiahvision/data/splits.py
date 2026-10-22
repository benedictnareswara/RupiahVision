"""Plan, assign and check train/val/test splits so the test set never leaks.

Two rules keep test images independent of training images:

1. Every capture session belongs to exactly one split (the `split` column of
   sessions.csv), and all its images land in that split. A background, lighting setup or
   day therefore never spans two splits.
2. Every physical note (and negative object) has a *role* that says which splits it may
   be photographed for. `plan` assigns roles before capture, from how many notes the
   class (denomination x series) has:

       3+ notes  separate test, val and train notes           level "note_disjoint"
       2 notes   one test note; the other for train and val   level "val_by_session"
       1 note    the only note appears in every split         level "session_only"

   Negative object types listed in split_config.json "open_set_object_types" are
   test-only, so the test set contains kinds of objects the model has never seen.

Public-dataset images always go to train: we can't tell which physical note they show.

The level reported per class comes from the images actually captured, not from the plan,
so results can never claim more separation than the data has.

Commands (run from ml/):
    uv run python -m rupiahvision.data.splits plan   # roles -> data/subject_roles.csv
    uv run python -m rupiahvision.data.splits make   # -> data/splits.csv + results summary
    uv run python -m rupiahvision.data.splits check  # verify committed files (CI runs this)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from rupiahvision import REPO_ROOT
from rupiahvision.data.dhash import near_duplicates
from rupiahvision.data.inventory import (
    DATA_DIR,
    EMPTY_SUBJECT,
    SPLITS,
    Inventory,
    InventoryError,
    ManifestRow,
    load_inventory,
    load_manifest,
    read_csv,
    write_csv,
)
from rupiahvision.labels import LabelSchema, class_name, load_schema

RESULTS_DIR = REPO_ROOT / "results" / "data"
ROLE_COLUMNS = ("subject_id", "group", "role")
SPLIT_COLUMNS = ("path", "split")

# Which splits a subject with each role may appear in.
ROLE_SPLITS: dict[str, frozenset[str]] = {
    "train": frozenset({"train"}),
    "val": frozenset({"val"}),
    "test": frozenset({"test"}),
    "trainval": frozenset({"train", "val"}),
    "any": frozenset(SPLITS),
    "open_set": frozenset({"test"}),
}
# New subjects fill a class's missing roles in this order, so test is covered first.
ROLE_PRIORITY = ("test", "val", "trainval", "train", "any")
MAX_LISTED = 20  # cap on near-duplicate pairs printed per split pair


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    val_fraction: float
    test_fraction: float
    open_set_object_types: tuple[str, ...]
    max_dhash_distance: int

    @classmethod
    def load(cls, path: Path) -> SplitConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=raw["seed"],
            val_fraction=raw["val_fraction"],
            test_fraction=raw["test_fraction"],
            open_set_object_types=tuple(raw["open_set_object_types"]),
            max_dhash_distance=raw["max_dhash_distance"],
        )


@dataclass
class LeakageReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    levels: dict[str, str] = field(default_factory=dict)  # group -> separation level


def target_roles(n: int, config: SplitConfig) -> list[str]:
    """Roles for a class with n subjects (see module docstring)."""
    if n == 0:
        return []
    if n == 1:
        return ["any"]
    if n == 2:
        return ["test", "trainval"]
    n_test = max(1, round(n * config.test_fraction))
    n_val = max(1, round(n * config.val_fraction))
    n_val = min(n_val, n - n_test - 1)  # always keep at least one training subject
    return ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)


def plan_roles(
    inventory: Inventory, existing: dict[str, str], config: SplitConfig
) -> tuple[dict[str, str], list[str]]:
    """Give every subject a role. Existing roles are kept, so photos already taken stay valid.

    New subjects take the roles their class is still missing, in a seeded random order.
    """
    roles: dict[str, str] = {}
    warnings: list[str] = []
    open_set = {f"negative:{t}" for t in config.open_set_object_types}

    for group, ids in inventory.groups().items():
        if group in open_set:
            roles.update(dict.fromkeys(ids, "open_set"))
            continue

        kept = {sid: existing[sid] for sid in ids if sid in existing}
        new = [sid for sid in ids if sid not in existing]
        random.Random(f"{config.seed}:{group}").shuffle(new)

        remaining = Counter(target_roles(len(ids), config))
        for sid, role in kept.items():
            if role == "any" and len(ids) > 1:
                role = "trainval" if remaining["trainval"] else "train"
                warnings.append(
                    f"{group}: {sid} has role 'any' from when it was the class's only note. "
                    f"From now on photograph it only in {role} sessions, and change its role "
                    f"to {role!r} in subject_roles.csv. Its earlier val/test images keep this "
                    f"class from being note-disjoint until they are removed."
                )
            if remaining[role] > 0:
                remaining[role] -= 1

        queue = [role for role in ROLE_PRIORITY for _ in range(remaining[role])]
        fallback = "trainval" if len(ids) == 2 else "train"
        for sid in new:
            roles[sid] = queue.pop(0) if queue else fallback
        roles.update(kept)

    return dict(sorted(roles.items())), warnings


def assign_splits(
    inventory: Inventory, manifest: Sequence[ManifestRow], roles: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Split of every image (from its session), plus every rule the images break."""
    splits: dict[str, str] = {}
    problems: list[str] = []
    for row in manifest:
        if not row.is_own:
            splits[row.path] = "train"
            continue
        session = inventory.sessions.get(row.session_id)
        if session is None:
            problems.append(f"{row.path}: session {row.session_id!r} is not in sessions.csv")
            continue
        splits[row.path] = session.split

        if row.subject_id == EMPTY_SUBJECT:
            expected_label = "unknown"
        elif row.subject_id in inventory.notes:
            note = inventory.notes[row.subject_id]
            expected_label = class_name(note.denomination, note.series, row.side)
        elif row.subject_id in inventory.negatives:
            expected_label = "unknown"
        else:
            problems.append(f"{row.path}: subject {row.subject_id!r} is not in the inventory")
            continue
        if row.label != expected_label:
            problems.append(f"{row.path}: label {row.label!r} should be {expected_label!r}")

        if row.subject_id == EMPTY_SUBJECT:
            continue
        role = roles.get(row.subject_id)
        if role is None:
            problems.append(f"{row.path}: {row.subject_id} has no role yet; run `splits plan`")
        elif session.split not in ROLE_SPLITS[role]:
            allowed = "/".join(sorted(ROLE_SPLITS[role]))
            problems.append(
                f"{row.path}: {row.subject_id} has role {role!r} (only {allowed}) "
                f"but session {session.session_id} is a {session.split} session"
            )
    return splits, problems


def separation_level(subject_splits: dict[str, set[str]]) -> str:
    """How independent a class's test (and val) images are, from where its subjects appear."""
    in_test = [s for s, splits in subject_splits.items() if "test" in splits]
    if not in_test:
        return "no_test_data"
    if any(subject_splits[s] - {"test"} for s in in_test):
        return "session_only"
    if any({"val", "train"} <= splits for splits in subject_splits.values()):
        return "val_by_session"
    return "note_disjoint"


def check_leakage(
    schema: LabelSchema,
    inventory: Inventory,
    manifest: Sequence[ManifestRow],
    splits: dict[str, str],
    config: SplitConfig,
) -> LeakageReport:
    report = LeakageReport()

    session_splits: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row.is_own:
            session_splits[row.session_id].add(splits[row.path])
            if row.subject_id != EMPTY_SUBJECT:
                subject_splits[row.subject_id].add(splits[row.path])
    for session_id, found in sorted(session_splits.items()):
        if len(found) > 1:
            report.errors.append(f"session {session_id} has images in {sorted(found)}")

    by_split: dict[str, list[ManifestRow]] = {s: [] for s in SPLITS}
    for row in manifest:
        by_split[splits[row.path]].append(row)
    for split_a, split_b in combinations(SPLITS, 2):
        rows_a, rows_b = by_split[split_a], by_split[split_b]
        pairs = near_duplicates(
            [r.dhash for r in rows_a], [r.dhash for r in rows_b], config.max_dhash_distance
        )
        if pairs:
            report.errors.append(
                f"{len(pairs)} near-duplicate image pair(s) between {split_a} and {split_b} "
                f"(dHash distance <= {config.max_dhash_distance}):"
            )
            report.errors += [
                f"    {rows_a[i].path} ~ {rows_b[j].path} (distance {d})"
                for i, j, d in pairs[:MAX_LISTED]
            ]

    groups = inventory.groups()
    note_groups = [f"{d}_{s}" for d in schema.denominations for s in schema.series]
    for group in note_groups + [g for g in groups if g.startswith("negative:")]:
        ids = groups.get(group, [])
        report.levels[group] = (
            separation_level({s: subject_splits.get(s, set()) for s in ids}) if ids else "no_notes"
        )

    used = {inventory.sessions[s] for s in session_splits if s in inventory.sessions}
    train_locations = {s.location for s in used if s.split == "train"}
    for location in sorted({s.location for s in used if s.split == "test"} & train_locations):
        report.warnings.append(f"location {location!r} is used by both train and test sessions")
    return report


def summarize(
    schema: LabelSchema,
    inventory: Inventory,
    manifest: Sequence[ManifestRow],
    splits: dict[str, str],
    roles: dict[str, str],
    report: LeakageReport,
    config: SplitConfig,
) -> dict:
    """Per-class counts and separation levels; the README dataset table is built from this."""

    def zero() -> dict[str, int]:
        return dict.fromkeys(SPLITS, 0)

    own_images: dict[str, dict[str, int]] = defaultdict(zero)
    public_images: Counter[str] = Counter()
    negative_images = zero()
    for row in manifest:
        if row.label == schema.unknown_class:
            negative_images[splits[row.path]] += 1
            continue
        group = row.label.rsplit("_", 1)[0]  # "50000_TE2022_back" -> "50000_TE2022"
        if row.is_own:
            own_images[group][splits[row.path]] += 1
        else:
            public_images[group] += 1

    groups = inventory.groups()
    classes = []
    for denomination in schema.denominations:
        for series in schema.series:
            group = f"{denomination}_{series}"
            ids = groups.get(group, [])
            classes.append(
                {
                    "denomination": denomination,
                    "series": series,
                    "notes": len(ids),
                    "planned_roles": dict(
                        sorted(Counter(roles[s] for s in ids if s in roles).items())
                    ),
                    "level": report.levels[group],
                    "own_images": own_images[group],
                    "public_images": public_images[group],
                }
            )

    sessions = zero()
    for session in inventory.sessions.values():
        sessions[session.split] += 1
    totals = zero()
    for split in splits.values():
        totals[split] += 1

    return {
        "classes": classes,
        "negatives": {
            "objects": len(inventory.negatives),
            "object_types": sorted({o.object_type for o in inventory.negatives.values()}),
            "open_set_object_types": list(config.open_set_object_types),
            "images": negative_images,
        },
        "sessions": sessions,
        "total_images": totals,
        "warnings": report.warnings,
    }


@dataclass(frozen=True)
class Paths:
    data_dir: Path = DATA_DIR
    results_dir: Path = RESULTS_DIR

    @property
    def roles(self) -> Path:
        return self.data_dir / "subject_roles.csv"

    @property
    def splits(self) -> Path:
        return self.data_dir / "splits.csv"

    @property
    def summary(self) -> Path:
        return self.results_dir / "dataset_summary.json"


def load_roles(path: Path) -> dict[str, str]:
    rows = read_csv(path, ROLE_COLUMNS, required=False)
    unknown = [r["role"] for r in rows if r["role"] not in ROLE_SPLITS]
    if unknown:
        raise InventoryError([f"{path.name}: unknown role(s) {sorted(set(unknown))}"])
    return {r["subject_id"]: r["role"] for r in rows}


def _summary_json(summary: dict) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False) + "\n"


def _split_rows(splits: dict[str, str]) -> list[dict]:
    return [{"path": p, "split": s} for p, s in sorted(splits.items())]


def cmd_plan(paths: Paths) -> int:
    inventory = load_inventory(paths.data_dir)
    config = SplitConfig.load(paths.data_dir / "split_config.json")
    roles, warnings = plan_roles(inventory, load_roles(paths.roles), config)
    write_csv(
        paths.roles,
        ROLE_COLUMNS,
        [{"subject_id": s, "group": inventory.group_of(s), "role": r} for s, r in roles.items()],
    )
    print(f"Wrote {paths.roles} ({len(roles)} subjects)")

    schema = load_schema(paths.data_dir / "labels.json")
    groups = inventory.groups()
    print("\nNotes per class (3+ gives fully note-disjoint train/val/test):")
    for denomination in schema.denominations:
        for series in schema.series:
            ids = groups.get(f"{denomination}_{series}", [])
            assigned = ", ".join(f"{s}={roles[s]}" for s in ids) or "none yet"
            need = f"  <- need {3 - len(ids)} more" if len(ids) < 3 else ""
            print(f"  Rp{denomination:>7} {series}: {len(ids)} ({assigned}){need}")
    for warning in warnings:
        print(f"\nWARNING: {warning}", file=sys.stderr)
    return 0


def _build(paths: Paths) -> tuple[dict[str, str], dict, list[str]]:
    """Everything `make` writes, plus the problems that should stop it."""
    schema = load_schema(paths.data_dir / "labels.json")
    config = SplitConfig.load(paths.data_dir / "split_config.json")
    inventory = load_inventory(paths.data_dir, schema)
    manifest = load_manifest(paths.data_dir / "manifest.csv")
    roles = load_roles(paths.roles)

    problems = [
        f"{s} has no role yet; run `splits plan`"
        for s in [*inventory.notes, *inventory.negatives]
        if s not in roles
    ]
    splits, assign_problems = assign_splits(inventory, manifest, roles)
    problems += assign_problems
    if problems:
        return splits, {}, problems

    report = check_leakage(schema, inventory, manifest, splits, config)
    summary = summarize(schema, inventory, manifest, splits, roles, report, config)
    return splits, summary, report.errors


def _report_problems(problems: list[str]) -> int:
    print(f"{len(problems)} problem(s):", file=sys.stderr)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    return 1


def cmd_make(paths: Paths) -> int:
    splits, summary, problems = _build(paths)
    if problems:
        return _report_problems(problems)
    write_csv(paths.splits, SPLIT_COLUMNS, _split_rows(splits))
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(_summary_json(summary), encoding="utf-8")
    print(f"Wrote {paths.splits} ({len(splits)} images) and {paths.summary}")
    print(f"Images per split: {summary['total_images']}")
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


def cmd_check(paths: Paths) -> int:
    splits, summary, problems = _build(paths)
    if not problems:
        if read_csv(paths.splits, SPLIT_COLUMNS, required=False) != _split_rows(splits):
            problems.append(f"{paths.splits.name} is out of date; run `splits make`")
        current = paths.summary.read_text(encoding="utf-8") if paths.summary.exists() else ""
        if current != _summary_json(summary):
            problems.append(f"{paths.summary.name} is out of date; run `splits make`")
    if problems:
        return _report_problems(problems)
    print(f"Splits OK: {summary['total_images']}, no leakage found.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan, make and check dataset splits.")
    parser.add_argument("command", choices=("plan", "make", "check"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    paths = Paths(args.data_dir, args.results_dir)
    command = {"plan": cmd_plan, "make": cmd_make, "check": cmd_check}[args.command]
    try:
        return command(paths)
    except InventoryError as error:
        return _report_problems(error.problems)


if __name__ == "__main__":
    raise SystemExit(main())
