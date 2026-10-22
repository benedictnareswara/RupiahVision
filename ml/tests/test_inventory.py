import pytest
from conftest import distinct_hash, image, negative, note, session

from rupiahvision.data.inventory import (
    DATA_DIR,
    InventoryError,
    load_clips,
    load_inventory,
    load_manifest,
)


def clip(file, session_id, subject_id, side="front", state="flat"):
    return dict(
        file=file, session_id=session_id, subject_id=subject_id, side=side, state=state,
        lighting="daylight", background="plain",
    )  # fmt: skip


def test_loads_valid_inventory(data_dir):
    data_dir.write(
        notes=[note("N001"), note("N002", 1000, "TE2016", "worn")],
        negatives=[negative("O001")],
        sessions=[session("S001", "train")],
    )
    inventory = load_inventory(data_dir.root)
    assert inventory.notes["N002"].denomination == 1000
    assert inventory.group_of("N001") == "50000_TE2022"
    assert inventory.group_of("O001") == "negative:receipt"
    assert inventory.group_of("EMPTY") is None
    assert inventory.groups() == {
        "1000_TE2016": ["N002"],
        "50000_TE2022": ["N001"],
        "negative:receipt": ["O001"],
    }


def test_reports_every_problem_at_once(data_dir):
    bad_session = session("S1", "holdout")
    bad_session["date"] = "24/10/2026"
    data_dir.write(
        notes=[note("N001", 75000), note("N001", series="TE2020"), note("X9", wear="mint")],
        sessions=[bad_session],
    )
    with pytest.raises(InventoryError) as error:
        load_inventory(data_dir.root)
    text = "\n".join(error.value.problems)
    for expected in [
        "duplicate note_id 'N001'",
        "denomination 75000",
        "series 'TE2020'",
        "note_id 'X9'",
        "wear 'mint'",
        "session_id 'S1'",
        "split 'holdout'",
        "date '24/10/2026'",
    ]:
        assert expected in text


def test_rejects_wrong_header(data_dir):
    (data_dir.root / "notes.csv").write_text("id,denomination\nN001,1000\n")
    with pytest.raises(InventoryError, match="header must be"):
        load_inventory(data_dir.root)


def test_skips_blank_rows(data_dir):
    data_dir.write(notes=[note("N001")])
    with (data_dir.root / "notes.csv").open("a") as f:
        f.write(",,,,,\n\n")
    assert list(load_inventory(data_dir.root).notes) == ["N001"]


def test_clip_validation(data_dir):
    data_dir.write(
        notes=[note("N001")],
        negatives=[negative("O001")],
        sessions=[session("S001", "train")],
        clips=[
            clip("S001/a.mov", "S001", "N001"),
            clip("S001/b.mov", "S001", "O001", side="-", state="-"),
            clip("S001/c.mov", "S001", "EMPTY", side="-", state="-"),
        ],
    )
    inventory = load_inventory(data_dir.root)
    assert len(load_clips(inventory, data_dir.root)) == 3

    data_dir.write(
        clips=[
            clip("S001/a.mov", "S009", "N001", side="top"),
            clip("S001/b.mov", "S001", "O001"),
            clip("S001/c.mov", "S001", "N404"),
        ]
    )
    with pytest.raises(InventoryError) as error:
        load_clips(inventory, data_dir.root)
    text = "\n".join(error.value.problems)
    assert "session 'S009'" in text
    assert "side 'top'" in text
    assert "side must be '-' for non-banknote O001" in text
    assert "subject 'N404'" in text


def test_manifest_is_optional_but_validated(data_dir):
    assert load_manifest(data_dir.root / "missing.csv") == []
    data_dir.write(
        manifest=[
            image("a.jpg", "unknown", "EMPTY", "S001", distinct_hash(1)),
            image("b.jpg", "unknown", "EMPTY", "S001", "not-a-hash"),
        ]
    )
    with pytest.raises(InventoryError, match="dhash 'not-a-hash'"):
        load_manifest(data_dir.root / "manifest.csv")


def test_committed_inventory_is_valid():
    inventory = load_inventory(DATA_DIR)
    load_clips(inventory, DATA_DIR)
    load_manifest(DATA_DIR / "manifest.csv")
