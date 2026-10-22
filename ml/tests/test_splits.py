import json

import pytest
from conftest import SPLIT_CONFIG, distinct_hash, image, negative, note, session

from rupiahvision.data import splits
from rupiahvision.data.inventory import load_inventory, load_manifest
from rupiahvision.data.splits import (
    Paths,
    SplitConfig,
    assign_splits,
    check_leakage,
    load_roles,
    plan_roles,
    separation_level,
    target_roles,
)
from rupiahvision.labels import load_schema

CONFIG = SplitConfig(**{**SPLIT_CONFIG, "open_set_object_types": ()})


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, []),
        (1, ["any"]),
        (2, ["test", "trainval"]),
        (3, ["test", "val", "train"]),
        (5, ["test", "val", "train", "train", "train"]),
        (10, ["test"] * 2 + ["val"] * 2 + ["train"] * 6),
    ],
)
def test_target_roles(n, expected):
    assert target_roles(n, CONFIG) == expected


def test_every_class_size_keeps_a_training_subject():
    big_holdout = SplitConfig(**{**SPLIT_CONFIG, "val_fraction": 0.5, "test_fraction": 0.5})
    for n in range(3, 20):
        assert "train" in target_roles(n, big_holdout)


def test_plan_covers_each_class_and_is_deterministic(data_dir):
    data_dir.write(
        notes=[note(f"N00{i}") for i in range(1, 4)] + [note("N004", 1000)],
        negatives=[negative("O001"), negative("O002")],
    )
    inventory = load_inventory(data_dir.root)
    roles, warnings = plan_roles(inventory, {}, CONFIG)
    assert sorted(roles[n] for n in ("N001", "N002", "N003")) == ["test", "train", "val"]
    assert roles["N004"] == "any"
    assert sorted([roles["O001"], roles["O002"]]) == ["test", "trainval"]
    assert warnings == []
    assert plan_roles(inventory, {}, CONFIG)[0] == roles


def test_plan_keeps_existing_roles_and_fills_gaps(data_dir):
    data_dir.write(notes=[note("N001"), note("N002"), note("N003")])
    inventory = load_inventory(data_dir.root)
    roles, _ = plan_roles(inventory, {"N001": "test", "N002": "train"}, CONFIG)
    assert roles == {"N001": "test", "N002": "train", "N003": "val"}


def test_plan_warns_when_only_note_gets_company(data_dir):
    data_dir.write(notes=[note("N001"), note("N002")])
    inventory = load_inventory(data_dir.root)
    roles, warnings = plan_roles(inventory, {"N001": "any"}, CONFIG)
    assert roles == {"N001": "any", "N002": "test"}
    assert len(warnings) == 1 and "N001" in warnings[0] and "trainval" in warnings[0]


def test_open_set_object_types_are_test_only(data_dir):
    data_dir.write(negatives=[negative("O001", "foreign_note"), negative("O002", "foreign_note")])
    config = SplitConfig(**{**SPLIT_CONFIG, "open_set_object_types": ("foreign_note",)})
    roles, _ = plan_roles(load_inventory(data_dir.root), {}, config)
    assert set(roles.values()) == {"open_set"}


def three_split_setup(data_dir):
    """Class 50000_TE2022 with one note per split, sessions S001-S003 = train/val/test."""
    data_dir.write(
        notes=[note("N001"), note("N002"), note("N003")],
        sessions=[session("S001", "train"), session("S002", "val", "kitchen"),
                  session("S003", "test", "cafe")],
    )  # fmt: skip
    return {"N001": "train", "N002": "val", "N003": "test"}


def test_assign_uses_session_split_and_public_goes_to_train(data_dir):
    roles = three_split_setup(data_dir)
    data_dir.write(
        manifest=[
            image("own/1.jpg", "50000_TE2022_front", "N001", "S001", distinct_hash(1)),
            image("own/2.jpg", "50000_TE2022_back", "N003", "S003", distinct_hash(2), "back"),
            image("own/3.jpg", "unknown", "EMPTY", "S002", distinct_hash(3)),
            image("pub/4.jpg", "50000_TE2022_front", "", "", distinct_hash(4), source="public:x"),
        ]
    )
    inventory = load_inventory(data_dir.root)
    result, problems = assign_splits(
        inventory, load_manifest(data_dir.root / "manifest.csv"), roles
    )
    assert problems == []
    assert result == {"own/1.jpg": "train", "own/2.jpg": "test", "own/3.jpg": "val",
                      "pub/4.jpg": "train"}  # fmt: skip


def test_assign_reports_role_label_and_session_problems(data_dir):
    roles = three_split_setup(data_dir)
    data_dir.write(
        manifest=[
            image("a.jpg", "50000_TE2022_front", "N003", "S001", distinct_hash(1)),  # test note
            image("b.jpg", "1000_TE2016_front", "N001", "S001", distinct_hash(2)),  # wrong label
            image("c.jpg", "50000_TE2022_front", "N001", "S404", distinct_hash(3)),  # no session
            image("d.jpg", "50000_TE2022_front", "N404", "S001", distinct_hash(4)),  # no subject
        ]
    )
    inventory = load_inventory(data_dir.root)
    _, problems = assign_splits(inventory, load_manifest(data_dir.root / "manifest.csv"), roles)
    text = "\n".join(problems)
    assert "N003 has role 'test' (only test) but session S001 is a train session" in text
    assert "label '1000_TE2016_front' should be '50000_TE2022_front'" in text
    assert "session 'S404'" in text
    assert "subject 'N404'" in text


@pytest.mark.parametrize(
    ("subject_splits", "level"),
    [
        ({}, "no_test_data"),
        ({"A": {"train"}}, "no_test_data"),
        ({"A": {"train"}, "B": {"val"}, "C": {"test"}}, "note_disjoint"),
        ({"A": {"train", "val"}, "C": {"test"}}, "val_by_session"),
        ({"A": {"train", "val", "test"}}, "session_only"),
        ({"A": {"train"}, "C": {"test", "val"}}, "session_only"),
    ],
)
def test_separation_level(subject_splits, level):
    assert separation_level(subject_splits) == level


def test_leakage_finds_near_duplicates_and_reports_levels(data_dir):
    roles = three_split_setup(data_dir)
    data_dir.write(
        manifest=[
            image("tr.jpg", "50000_TE2022_front", "N001", "S001", "0000000000000000"),
            image("va.jpg", "50000_TE2022_front", "N002", "S002", distinct_hash(9)),
            image("te.jpg", "50000_TE2022_front", "N003", "S003", "0000000000000007"),
        ]
    )
    schema = load_schema(data_dir.root / "labels.json")
    inventory = load_inventory(data_dir.root)
    manifest = load_manifest(data_dir.root / "manifest.csv")
    split_of, _ = assign_splits(inventory, manifest, roles)
    report = check_leakage(schema, inventory, manifest, split_of, CONFIG)
    assert "1 near-duplicate image pair(s) between train and test" in report.errors[0]
    assert "tr.jpg ~ te.jpg (distance 3)" in report.errors[1]
    assert report.levels["50000_TE2022"] == "note_disjoint"
    assert report.levels["1000_TE2016"] == "no_notes"
    assert report.warnings == []


def test_leakage_flags_session_in_two_splits_and_shared_location(data_dir):
    data_dir.write(
        notes=[note("N001")],
        sessions=[session("S001", "train"), session("S002", "test")],  # same location
        manifest=[
            image("a.jpg", "50000_TE2022_front", "N001", "S001", distinct_hash(1)),
            image("b.jpg", "50000_TE2022_front", "N001", "S001", distinct_hash(2)),
            image("c.jpg", "50000_TE2022_front", "N001", "S002", distinct_hash(3)),
        ],
    )
    schema = load_schema(data_dir.root / "labels.json")
    inventory = load_inventory(data_dir.root)
    manifest = load_manifest(data_dir.root / "manifest.csv")
    tampered = {"a.jpg": "train", "b.jpg": "val", "c.jpg": "test"}
    report = check_leakage(schema, inventory, manifest, tampered, CONFIG)
    assert report.errors == ["session S001 has images in ['train', 'val']"]
    assert report.levels["50000_TE2022"] == "session_only"
    assert report.warnings == ["location 'home_desk' is used by both train and test sessions"]


def test_plan_make_check_end_to_end(data_dir, tmp_path, capsys):
    data_dir.write(
        notes=[note("N001"), note("N002"), note("N003")],
        sessions=[session("S001", "train"), session("S002", "val", "kitchen"),
                  session("S003", "test", "cafe")],
    )  # fmt: skip
    args = ["--data-dir", str(data_dir.root), "--results-dir", str(tmp_path / "results")]
    assert splits.main(["plan", *args]) == 0
    assert "Rp  50000 TE2022: 3" in capsys.readouterr().out

    roles = load_roles(data_dir.root / "subject_roles.csv")
    session_of = {"train": "S001", "val": "S002", "test": "S003"}
    data_dir.write(
        manifest=[
            image(f"{sid}.jpg", "50000_TE2022_front", sid, session_of[role], distinct_hash(i))
            for i, (sid, role) in enumerate(roles.items())
        ]
    )
    assert splits.main(["check", *args]) == 1  # nothing made yet
    assert splits.main(["make", *args]) == 0
    assert splits.main(["check", *args]) == 0

    summary = json.loads((tmp_path / "results" / "dataset_summary.json").read_text())
    row = next(
        c for c in summary["classes"] if c["denomination"] == 50000 and c["series"] == "TE2022"
    )
    assert row["notes"] == 3
    assert row["level"] == "note_disjoint"
    assert row["own_images"] == {"train": 1, "val": 1, "test": 1}
    assert summary["total_images"] == {"train": 1, "val": 1, "test": 1}

    # Adding an image without re-running make leaves the committed splits stale.
    data_dir.write(
        manifest=[
            *load_manifest_rows(data_dir),
            image("new.jpg", "unknown", "EMPTY", "S001", distinct_hash(99)),
        ]
    )
    assert splits.main(["check", *args]) == 1
    assert "splits.csv is out of date" in capsys.readouterr().err


def test_make_refuses_subjects_without_roles(data_dir, tmp_path, capsys):
    data_dir.write(notes=[note("N001")])
    paths = Paths(data_dir.root, tmp_path / "results")
    assert splits.cmd_make(paths) == 1
    assert "N001 has no role yet" in capsys.readouterr().err


def test_committed_splits_are_up_to_date_and_leak_free():
    assert splits.main(["check"]) == 0


def load_manifest_rows(data_dir):
    return [vars(row) for row in load_manifest(data_dir.root / "manifest.csv")]
