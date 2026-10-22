"""Import public datasets and split them into train/val/test (the v1 dataset, PLAN.md).

Labels are the app's outputs: "unknown" plus the 7 denominations. The public sets don't
label series or side.

    kaggle_rupiah22    Kaggle "rupiah banknotes 2022": folders jpg1000 ... jpg100000 and
                       jpgbackground (-> unknown); jpgmulticlass is skipped (several notes
                       per photo, so no single label).
    roboflow_bangkit   Roboflow "Uang Rupiah Emisi 2022", COCO JSON export. Each box is
                       cropped to one image. One crop per image is also taken away from all
                       boxes as an "unknown" example from the same scenes.

Images are copied to data/public/<dataset>/ (git-ignored; the Kaggle license is unknown,
so these images are never committed). The committed files are in data/v1/:
import_<dataset>.csv (one row per imported image) and index.csv (the same, with splits).

Splitting. Photos are grouped so near-identical shots stay together:
  - Roboflow exports augmented copies named <original>_jpg.rf.<hash>.jpg; only one copy
    per original is kept (our training augmentation replaces Roboflow's).
  - Numbered camera files (IMG_6640.jpg) are grouped in blocks of GROUP_BLOCK consecutive
    numbers, since a run of shots is usually one note on one surface.
Whole groups are assigned to splits, stratified by dataset and label. Afterwards, any
val/test image that is a near-duplicate (dHash) of an image in another split is dropped.
This is weaker than the by-physical-note split in rupiahvision.data.splits: test notes are
probably also in training, so test accuracy is optimistic.

Usage (from ml/):
    uv run python -m rupiahvision.public import-kaggle ~/Downloads/archive.zip
    uv run python -m rupiahvision.public import-roboflow ~/Downloads/Uang-Rupiah.coco.zip
    uv run python -m rupiahvision.public split
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from rupiahvision import REPO_ROOT
from rupiahvision.data.dhash import dhash, near_duplicates, to_hex
from rupiahvision.data.frames import resize_to_max_side
from rupiahvision.data.inventory import DATA_DIR, SPLITS, read_csv, write_csv
from rupiahvision.labels import load_schema

INDEX_COLUMNS = ("path", "label", "dataset", "group", "dhash", "split")
IMPORT_COLUMNS = INDEX_COLUMNS[:-1]
SUMMARY_PATH = REPO_ROOT / "results" / "data" / "public_summary.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MAX_SIDE = 1024
MIN_CROP_SIDE = 64
GROUP_BLOCK = 10
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MAX_DHASH_DISTANCE = 4
SEED = 20261022

KAGGLE = "kaggle_rupiah22"
ROBOFLOW = "roboflow_bangkit"


@contextmanager
def opened(source: Path) -> Iterator[Path]:
    """A directory with the dataset's files: `source` itself, or a zip unpacked to a temp dir."""
    if source.is_dir():
        yield source
        return
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(source) as archive:
        archive.extractall(tmp)
        yield Path(tmp)


def group_key(stem: str) -> str:
    """Group for a photo: the Roboflow original name, numbered files bucketed in blocks.

    WhatsApp exports ("WhatsApp-Image-2023-07-30-at-17-17-28-1-") are grouped by the minute
    they were sent: a burst of photos sent together is usually one note on one surface.
    """
    original = re.split(r"_(?:jpe?g|png)\.rf\.", stem, maxsplit=1, flags=re.IGNORECASE)[0]
    whatsapp = re.match(r"WhatsApp-Image-\d{4}-\d{2}-\d{2}-at-\d{2}-\d{2}", original)
    if whatsapp:
        return whatsapp.group(0)
    original = original.rstrip("-_ ")  # Roboflow turns "(1)" and spaces into trailing dashes
    match = re.fullmatch(r"(.*?)(\d+)", original)
    if match:
        return f"{match.group(1)}{int(match.group(2)) // GROUP_BLOCK}"
    return original


def save(image: np.ndarray, out_dir: Path, name: str, data_dir: Path) -> tuple[str, str]:
    """Write an image (longer side capped at MAX_SIDE); returns (path under data/, dhash)."""
    image = resize_to_max_side(image, MAX_SIDE)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jpg"
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path.relative_to(data_dir).as_posix(), to_hex(dhash(image))


def import_kaggle(source: Path, data_dir: Path = DATA_DIR) -> list[dict]:
    rows = []
    out_root = data_dir / "public" / KAGGLE
    shutil.rmtree(out_root, ignore_errors=True)
    with opened(source) as root:
        folders = sorted(
            p
            for p in root.rglob("jpg*")
            if p.is_dir() and re.fullmatch(r"jpg(\d+|background)", p.name)
        )
        if not folders:
            raise SystemExit(f"no jpg<denomination> folders found in {source}")
        for folder in folders:
            tag = folder.name.removeprefix("jpg")
            label = "unknown" if tag == "background" else tag
            for file in sorted(folder.iterdir()):
                if file.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = cv2.imread(str(file), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"skipping unreadable {file.name}", file=sys.stderr)
                    continue
                path, h = save(image, out_root / label, file.stem, data_dir)
                group = f"{KAGGLE}:{label}:{group_key(file.stem)}"
                rows.append(dict(path=path, label=label, dataset=KAGGLE, group=group, dhash=h))
    return rows


def roboflow_label(category: str) -> str | None:
    """'50k' -> '50000'; the dataset-wide parent category maps to None."""
    match = re.fullmatch(r"(\d+)\s*k", category.strip(), flags=re.IGNORECASE)
    return str(int(match.group(1)) * 1000) if match else None


def background_crop(
    image: np.ndarray, boxes: list[tuple[int, int, int, int]], rng: random.Random
) -> np.ndarray | None:
    """A random square crop that touches no box, or None if none fits after a few tries."""
    height, width = image.shape[:2]
    side = min(height, width) // 3
    if side < MIN_CROP_SIDE:
        return None
    for _ in range(20):
        x, y = rng.randint(0, width - side), rng.randint(0, height - side)
        if all(
            x + side <= bx or bx + bw <= x or y + side <= by or by + bh <= y
            for bx, by, bw, bh in boxes
        ):
            return image[y : y + side, x : x + side]
    return None


def import_roboflow(source: Path, data_dir: Path = DATA_DIR) -> list[dict]:
    rows = []
    out_root = data_dir / "public" / ROBOFLOW
    shutil.rmtree(out_root, ignore_errors=True)
    rng = random.Random(SEED)
    seen_originals: set[str] = set()
    with opened(source) as root:
        annotation_files = sorted(root.rglob("_annotations.coco.json"))
        if not annotation_files:
            raise SystemExit(f"no _annotations.coco.json found in {source}")
        for annotations in annotation_files:
            coco = json.loads(annotations.read_text(encoding="utf-8"))
            labels = {c["id"]: roboflow_label(c["name"]) for c in coco["categories"]}
            boxes_by_image = defaultdict(list)
            for a in coco["annotations"]:
                boxes_by_image[a["image_id"]].append((labels.get(a["category_id"]), a["bbox"]))

            for info in sorted(coco["images"], key=lambda i: i["file_name"]):
                stem = Path(info["file_name"]).stem
                original = group_key(stem)
                key = re.split(r"_(?:jpe?g|png)\.rf\.", stem, maxsplit=1, flags=re.I)[0]
                if key in seen_originals:
                    continue  # an augmented copy of an image we already have
                seen_originals.add(key)
                image = cv2.imread(str(annotations.parent / info["file_name"]), cv2.IMREAD_COLOR)
                if image is None:
                    print(f"skipping unreadable {info['file_name']}", file=sys.stderr)
                    continue
                group = f"{ROBOFLOW}:{original}"
                height, width = image.shape[:2]
                int_boxes = []
                for n, (label, (x, y, w, h)) in enumerate(boxes_by_image[info["id"]]):
                    int_boxes.append((int(x), int(y), int(np.ceil(w)), int(np.ceil(h))))
                    if label is None or min(w, h) < MIN_CROP_SIDE:
                        continue
                    pad_x, pad_y = 0.1 * w, 0.1 * h  # keep a little context around the note
                    x0, y0 = max(0, int(x - pad_x)), max(0, int(y - pad_y))
                    x1, y1 = min(width, int(x + w + pad_x)), min(height, int(y + h + pad_y))
                    path, hsh = save(image[y0:y1, x0:x1], out_root / label, f"{key}_{n}", data_dir)
                    rows.append(
                        dict(path=path, label=label, dataset=ROBOFLOW, group=group, dhash=hsh)
                    )
                crop = background_crop(image, int_boxes, rng)
                if crop is not None:
                    path, hsh = save(crop, out_root / "unknown", f"{key}_bg", data_dir)
                    rows.append(
                        dict(path=path, label="unknown", dataset=ROBOFLOW, group=group, dhash=hsh)
                    )
    return rows


def assign_groups(rows: list[dict], seed: int = SEED) -> dict[str, str]:
    """Group -> split, aiming for VAL/TEST_FRACTION of every (dataset, label) in val/test.

    One group can hold several labels (a session photographing several notes), so groups
    are assigned jointly, not one label at a time. Groups are visited in seeded random
    order. Each goes to the held-out split where it does the most good: the images that
    fill a label's remaining quota, minus those that overshoot it. If neither split
    gains, the group goes to train. Labels with fewer than 3 groups can't be split
    without leaving train empty, so their groups always train.
    """
    contents: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for row in rows:
        contents[row["group"]][(row["dataset"], row["label"])] += 1
    totals: Counter[tuple[str, str]] = Counter()
    group_count: Counter[tuple[str, str]] = Counter()
    for counts in contents.values():
        totals.update(counts)
        group_count.update(counts.keys())
    fractions = {"test": TEST_FRACTION, "val": VAL_FRACTION}
    remaining = {s: {k: f * n for k, n in totals.items()} for s, f in fractions.items()}

    order = sorted(contents)
    random.Random(seed).shuffle(order)
    split_of: dict[str, str] = {}
    for group in order:
        counts = contents[group]
        if any(group_count[k] < 3 for k in counts):
            split_of[group] = "train"
            continue
        gains = {
            split: sum(min(c, left[k]) - max(0.0, c - left[k]) for k, c in counts.items())
            for split, left in remaining.items()
        }
        best = max(gains, key=gains.get)
        split_of[group] = best if gains[best] > 0 else "train"
        if gains[best] > 0:
            for k, c in counts.items():
                remaining[best][k] -= c

    # Repair: a label made of a few big groups can end up with no val or test images at
    # all, so nothing measures it. Move its smallest training group there, as long as at
    # least one group stays in train.
    for key in sorted(totals):
        if group_count[key] < 3:
            continue
        mine = [g for g in sorted(contents) if key in contents[g]]
        for split in ("test", "val"):
            if any(split_of[g] == split for g in mine):
                continue
            training = sorted((g for g in mine if split_of[g] == "train"),
                              key=lambda g: sum(contents[g].values()))  # fmt: skip
            if len(training) >= 2:
                split_of[training[0]] = split
    return split_of


def split_rows(rows: list[dict], seed: int = SEED) -> tuple[list[dict], list[str]]:
    """Rows with a split column, minus val/test near-duplicates of other splits."""
    split_of = assign_groups(rows, seed)
    rows = [{**row, "split": split_of[row["group"]]} for row in rows]
    by_split = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    dropped: set[str] = set()
    notes = []
    # Compare val and test against every other split; drop the val/test side of each pair.
    for held_out, others in (("test", ("train", "val")), ("val", ("train",))):
        for other in others:
            pairs = near_duplicates(
                [r["dhash"] for r in by_split[held_out]],
                [r["dhash"] for r in by_split[other] if r["path"] not in dropped],
                MAX_DHASH_DISTANCE,
            )
            newly = {by_split[held_out][i]["path"] for i, _, _ in pairs} - dropped
            if newly:
                notes.append(f"dropped {len(newly)} {held_out} image(s) near-duplicating {other}")
            dropped |= newly
    kept = sorted((r for r in rows if r["path"] not in dropped), key=lambda r: r["path"])
    return kept, notes


def summarize(rows: list[dict], notes: list[str]) -> dict:
    schema = load_schema()
    counts = {label: dict.fromkeys(SPLITS, 0) for label in schema.outputs}
    datasets: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(SPLITS, 0))
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        counts[row["label"]][row["split"]] += 1
        datasets[row["dataset"]][row["split"]] += 1
        groups[row["split"]].add(row["group"])
    return {
        "labels": counts,
        "datasets": dict(sorted(datasets.items())),
        "groups": {s: len(groups[s]) for s in SPLITS},
        "total": {s: sum(c[s] for c in counts.values()) for s in SPLITS},
        "notes": notes,
    }


def import_path(dataset: str, data_dir: Path) -> Path:
    return data_dir / "v1" / f"import_{dataset}.csv"


def index_path(data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "v1" / "index.csv"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import and split public datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("import-kaggle", "import-roboflow"):
        sub.add_parser(name).add_argument("source", type=Path, help="zip file or unpacked folder")
    sub.add_parser("split")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)

    if args.command.startswith("import"):
        dataset, importer = (
            (KAGGLE, import_kaggle)
            if args.command == "import-kaggle"
            else (ROBOFLOW, import_roboflow)
        )
        rows = importer(args.source.expanduser(), args.data_dir)
        write_csv(import_path(dataset, args.data_dir), IMPORT_COLUMNS, rows)
        print(
            f"Imported {len(rows)} images from {dataset}: {dict(Counter(r['label'] for r in rows))}"
        )
        print("Next: uv run python -m rupiahvision.public split")
        return 0

    rows = []
    for dataset in (KAGGLE, ROBOFLOW):
        rows += read_csv(import_path(dataset, args.data_dir), IMPORT_COLUMNS, required=False)
    if not rows:
        print(
            "No imported datasets found; run import-kaggle / import-roboflow first.",
            file=sys.stderr,
        )
        return 1
    kept, notes = split_rows(rows)
    write_csv(index_path(args.data_dir), INDEX_COLUMNS, kept)
    summary = summarize(kept, notes)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {index_path(args.data_dir)}: {summary['total']} images, {summary['groups']} groups"
    )
    for note in notes:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
