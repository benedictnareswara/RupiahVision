"""Extract frames from recorded clips and build data/manifest.csv.

For every row of clips.csv, reads data/raw/<file>:
  - videos (.mov, .mp4, .m4v) are sampled at FPS frames per second;
  - photos (.jpg, .jpeg, .png) give one image each.
A frame whose dHash is within DEDUPE_DISTANCE bits of the previous kept frame of the same
clip is dropped (the phone was held still), so the dataset isn't padded with copies.

Frames are saved to data/frames/<session>/<clip>_<n>.jpg with the longer side scaled down
to MAX_SIDE. Each clip also gets a sidecar <clip>.json listing its frames and hashes, so
re-running skips clips that are already extracted with the same settings.

The manifest is rebuilt from scratch on every run: one row per extracted frame, plus the
rows of any committed data/public_<dataset>.csv files (public images, train only).

Usage (from ml/):
    uv run python -m rupiahvision.data.frames           # extract new clips, write manifest
    uv run python -m rupiahvision.data.frames --force   # re-extract everything
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from rupiahvision.data.dhash import dhash, to_hex
from rupiahvision.data.inventory import (
    DATA_DIR,
    EMPTY_SUBJECT,
    MANIFEST_COLUMNS,
    OWN_SOURCE,
    Clip,
    Inventory,
    InventoryError,
    load_clips,
    load_inventory,
    read_csv,
    write_csv,
)
from rupiahvision.labels import class_name, load_schema

VIDEO_SUFFIXES = (".mov", ".mp4", ".m4v")
PHOTO_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class FrameSettings:
    fps: float = 2.0
    max_side: int = 1024
    dedupe_distance: int = 2
    jpeg_quality: int = 90


DEFAULT_SETTINGS = FrameSettings()


def resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image
    size = (round(width * scale), round(height * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def sample_video(path: Path, fps: float) -> Iterator[np.ndarray]:
    """Frames at roughly `fps` per second, using grab() to skip decoding unused frames."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = source_fps / fps
    next_index = 0.0
    index = 0
    try:
        while capture.grab():
            if index + 1e-6 >= next_index:
                ok, frame = capture.retrieve()
                if ok:
                    yield frame
                next_index += step
            index += 1
    finally:
        capture.release()


def read_clip(path: Path, fps: float) -> Iterator[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        yield from sample_video(path, fps)
    elif suffix in PHOTO_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read image {path}")
        yield image
    else:
        hint = " (set iPhone Camera > Formats to Most Compatible)" if suffix == ".heic" else ""
        raise OSError(f"unsupported file type {suffix!r} for {path}{hint}")


def extract_clip(
    clip: Clip, raw_dir: Path, frames_dir: Path, settings: FrameSettings, force: bool = False
) -> list[dict]:
    """Extract one clip; returns [{"file": <name>, "dhash": <hex>}, ...] in frame order."""
    source = raw_dir / clip.file
    if not source.exists():
        raise OSError(f"clip not found: {source}")
    out_dir = frames_dir / clip.session_id
    stem = Path(clip.file).stem
    sidecar = out_dir / f"{stem}.json"
    fingerprint = {"source_bytes": source.stat().st_size, "settings": asdict(settings)}

    if sidecar.exists() and not force:
        cached = json.loads(sidecar.read_text(encoding="utf-8"))
        if cached["fingerprint"] == fingerprint and all(
            (out_dir / f["file"]).exists() for f in cached["frames"]
        ):
            return cached["frames"]

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{stem}_*.jpg"):
        if old.stem[len(stem) + 1 :].isdigit():  # not another clip named "<stem>_x"
            old.unlink()
    frames: list[dict] = []
    last_hash: int | None = None
    for frame in read_clip(source, settings.fps):
        frame = resize_to_max_side(frame, settings.max_side)
        h = dhash(frame)
        if last_hash is not None and (h ^ last_hash).bit_count() <= settings.dedupe_distance:
            continue
        last_hash = h
        name = f"{stem}_{len(frames):04d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
        frames.append({"file": name, "dhash": to_hex(h)})
    if not frames:
        raise OSError(f"no frames could be read from {source}")
    sidecar.write_text(
        json.dumps({"fingerprint": fingerprint, "frames": frames}, indent=1), encoding="utf-8"
    )
    return frames


def clip_label(clip: Clip, inventory: Inventory) -> str:
    if clip.subject_id in inventory.notes:
        note = inventory.notes[clip.subject_id]
        return class_name(note.denomination, note.series, clip.side)
    assert clip.subject_id in inventory.negatives or clip.subject_id == EMPTY_SUBJECT
    return "unknown"


def public_rows(data_dir: Path) -> list[dict]:
    """Rows of every committed data/public_<dataset>.csv, already in manifest format."""
    rows = []
    for path in sorted(data_dir.glob("public_*.csv")):
        for row in read_csv(path, MANIFEST_COLUMNS):
            if row["source"] == OWN_SOURCE:
                raise InventoryError([f"{path.name}: public rows must not have source 'own'"])
            rows.append(row)
    return rows


def build_manifest(
    data_dir: Path = DATA_DIR, settings: FrameSettings = DEFAULT_SETTINGS, force: bool = False
) -> tuple[list[dict], list[str]]:
    """Extract every clip; returns manifest rows and the clips that failed."""
    schema = load_schema(data_dir / "labels.json")
    inventory = load_inventory(data_dir, schema)
    clips = load_clips(inventory, data_dir, schema)
    raw_dir, frames_dir = data_dir / "raw", data_dir / "frames"

    rows: list[dict] = []
    failures: list[str] = []
    owner_of: dict[tuple[str, str], str] = {}
    for clip in clips:
        key = (clip.session_id, Path(clip.file).stem)
        if key in owner_of:
            failures.append(f"{clip.file}: same name as {owner_of[key]} in {clip.session_id}")
            continue
        owner_of[key] = clip.file
        try:
            frames = extract_clip(clip, raw_dir, frames_dir, settings, force)
        except OSError as error:
            failures.append(f"{clip.file}: {error}")
            continue
        label = clip_label(clip, inventory)
        for frame in frames:
            rows.append(
                {
                    "path": f"frames/{clip.session_id}/{frame['file']}",
                    "label": label,
                    "source": OWN_SOURCE,
                    "session_id": clip.session_id,
                    "subject_id": clip.subject_id,
                    "side": clip.side,
                    "state": clip.state,
                    "lighting": clip.lighting,
                    "background": clip.background,
                    "dhash": frame["dhash"],
                }
            )
    rows += public_rows(data_dir)
    return sorted(rows, key=lambda r: r["path"]), failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frames and build data/manifest.csv.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="re-extract every clip")
    args = parser.parse_args(argv)
    try:
        rows, failures = build_manifest(args.data_dir, force=args.force)
    except InventoryError as error:
        print(error, file=sys.stderr)
        return 1
    if failures:
        # Don't write a manifest with clips silently missing.
        print(f"{len(failures)} clip(s) failed; manifest not written:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    write_csv(args.data_dir / "manifest.csv", MANIFEST_COLUMNS, rows)
    own = sum(r["source"] == OWN_SOURCE for r in rows)
    print(f"Wrote manifest.csv: {own} own frames, {len(rows) - own} public images.")
    print("Next: uv run python -m rupiahvision.data.splits make")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
