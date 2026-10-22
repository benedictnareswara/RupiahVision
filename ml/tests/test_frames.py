import json

import cv2
import numpy as np
import pytest
from conftest import distinct_hash, image, note, session

from rupiahvision.data import frames, splits
from rupiahvision.data.frames import FrameSettings, build_manifest, extract_clip
from rupiahvision.data.inventory import MANIFEST_COLUMNS, Clip, load_manifest, write_csv

SETTINGS = FrameSettings()


def scene(seed: int, size=(240, 320)) -> np.ndarray:
    small = np.random.default_rng(seed).integers(0, 256, (12, 16, 3), dtype=np.uint8)
    return cv2.resize(small, size[::-1], interpolation=cv2.INTER_CUBIC)


def write_video(path, moving_seconds=3, still_seconds=2, fps=30):
    """A new scene every 0.5 s while "moving", then one scene held still."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    assert writer.isOpened()
    for i in range(moving_seconds * fps):
        writer.write(scene(i // (fps // 2)))
    for _ in range(still_seconds * fps):
        writer.write(scene(1000))
    writer.release()


def make_clip(file, session_id="S001", subject_id="N001", side="front"):
    not_note = not subject_id.startswith("N")
    return Clip(
        file=file, session_id=session_id, subject_id=subject_id,
        side="-" if not_note else side, state="-" if not_note else "flat",
        lighting="daylight", background="plain",
    )  # fmt: skip


def test_video_is_sampled_and_still_frames_deduplicated(tmp_path):
    write_video(tmp_path / "raw" / "S001" / "IMG_1.mp4")
    result = extract_clip(
        make_clip("S001/IMG_1.mp4"), tmp_path / "raw", tmp_path / "frames", SETTINGS
    )
    # 3 s moving at 2 fps = 6 distinct frames; the 2 s still segment adds only 1.
    assert len(result) == 7
    assert [f["file"] for f in result] == [f"IMG_1_{i:04d}.jpg" for i in range(7)]
    for f in result:
        saved = cv2.imread(str(tmp_path / "frames" / "S001" / f["file"]))
        assert saved.shape == (240, 320, 3)
    sidecar = json.loads((tmp_path / "frames" / "S001" / "IMG_1.json").read_text())
    assert sidecar["frames"] == result


def test_photo_gives_one_resized_frame(tmp_path):
    path = tmp_path / "raw" / "S001" / "IMG_2.jpg"
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), scene(5, size=(1500, 2000)))
    result = extract_clip(
        make_clip("S001/IMG_2.jpg"), tmp_path / "raw", tmp_path / "frames", SETTINGS
    )
    assert len(result) == 1
    saved = cv2.imread(str(tmp_path / "frames" / "S001" / result[0]["file"]))
    assert saved.shape == (768, 1024, 3)


def test_extraction_is_cached_until_forced(tmp_path):
    write_video(tmp_path / "raw" / "S001" / "IMG_1.mp4", moving_seconds=1, still_seconds=0)
    args = (make_clip("S001/IMG_1.mp4"), tmp_path / "raw", tmp_path / "frames", SETTINGS)
    first = extract_clip(*args)
    frame = tmp_path / "frames" / "S001" / first[0]["file"]
    frame.write_bytes(b"sentinel")  # a cache hit leaves this untouched
    assert extract_clip(*args) == first
    assert frame.read_bytes() == b"sentinel"
    assert extract_clip(*args, force=True) == first
    assert frame.read_bytes() != b"sentinel"


def test_changed_settings_invalidate_cache(tmp_path):
    write_video(tmp_path / "raw" / "S001" / "IMG_1.mp4", moving_seconds=2, still_seconds=0)
    clip = make_clip("S001/IMG_1.mp4")
    at_2fps = extract_clip(clip, tmp_path / "raw", tmp_path / "frames", SETTINGS)
    at_1fps = extract_clip(clip, tmp_path / "raw", tmp_path / "frames", FrameSettings(fps=1.0))
    assert (len(at_2fps), len(at_1fps)) == (4, 2)
    assert len(list((tmp_path / "frames" / "S001").glob("IMG_1_*.jpg"))) == 2


def test_heic_gets_a_helpful_error(tmp_path):
    path = tmp_path / "raw" / "S001" / "IMG_3.HEIC"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    with pytest.raises(OSError, match="Most Compatible"):
        extract_clip(make_clip("S001/IMG_3.HEIC"), tmp_path / "raw", tmp_path / "frames", SETTINGS)


def setup_three_sessions(data_dir):
    data_dir.write(
        notes=[note("N001"), note("N002"), note("N003")],
        sessions=[session("S001", "train"), session("S002", "val", "kitchen"),
                  session("S003", "test", "cafe")],
    )  # fmt: skip
    assert splits.main(["plan", "--data-dir", str(data_dir.root)]) == 0
    return splits.load_roles(data_dir.root / "subject_roles.csv")


def clip_row(file, session_id, subject_id, side="front"):
    return vars(make_clip(file, session_id, subject_id, side))


def test_build_manifest_then_make_splits(data_dir, tmp_path):
    roles = setup_three_sessions(data_dir)
    session_of = {"train": "S001", "val": "S002", "test": "S003"}
    clips = []
    for n, (note_id, role) in enumerate(sorted(roles.items())):
        sid = session_of[role]
        # A different scene seed per clip, so no clip looks like another across splits.
        path = data_dir.root / "raw" / sid / f"IMG_{n}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        for i in range(30):
            writer.write(scene(100 * n + i // 15))
        writer.release()
        clips.append(clip_row(f"{sid}/IMG_{n}.mp4", sid, note_id, side="back"))
    empty = data_dir.root / "raw" / "S001" / "IMG_9.jpg"
    cv2.imwrite(str(empty), scene(999))
    clips.append(clip_row("S001/IMG_9.jpg", "S001", "EMPTY"))
    data_dir.write(clips=clips)
    public = image("public/x/1.jpg", "50000_TE2022_front", "", "", distinct_hash(5),
                   source="public:x")  # fmt: skip
    write_csv(data_dir.root / "public_x.csv", MANIFEST_COLUMNS, [public])

    assert frames.main(["--data-dir", str(data_dir.root)]) == 0
    manifest = load_manifest(data_dir.root / "manifest.csv")
    own = [r for r in manifest if r.is_own]
    assert len(own) == 3 * 2 + 1  # 1 s clips at 2 fps, plus one photo
    assert {r.label for r in own} == {"50000_TE2022_back", "unknown"}
    assert [r.path for r in manifest if not r.is_own] == ["public/x/1.jpg"]

    results = ["--data-dir", str(data_dir.root), "--results-dir", str(tmp_path / "results")]
    assert splits.main(["make", *results]) == 0
    summary = json.loads((tmp_path / "results" / "dataset_summary.json").read_text())
    row = next(
        c for c in summary["classes"] if c["denomination"] == 50000 and c["series"] == "TE2022"
    )
    assert row["own_images"] == {"train": 2, "val": 2, "test": 2}
    assert row["public_images"] == 1
    assert row["level"] == "note_disjoint"
    assert summary["negatives"]["images"] == {"train": 1, "val": 0, "test": 0}


def test_failed_clip_blocks_manifest(data_dir, capsys):
    setup_three_sessions(data_dir)
    data_dir.write(clips=[clip_row("S001/missing.mov", "S001", "EMPTY")])
    assert frames.main(["--data-dir", str(data_dir.root)]) == 1
    assert "clip not found" in capsys.readouterr().err
    assert not (data_dir.root / "manifest.csv").exists()


def test_same_name_in_one_session_is_rejected(data_dir):
    setup_three_sessions(data_dir)
    for suffix in ("jpg", "png"):
        path = data_dir.root / "raw" / "S001" / f"IMG_1.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), scene(1))
    data_dir.write(
        clips=[
            clip_row("S001/IMG_1.jpg", "S001", "EMPTY"),
            clip_row("S001/IMG_1.png", "S001", "EMPTY"),
        ]
    )
    _, failures = build_manifest(data_dir.root)
    assert failures == ["S001/IMG_1.png: same name as S001/IMG_1.jpg in S001"]
