import shutil

import pytest

from rupiahvision import REPO_ROOT
from rupiahvision.reporting import readme_tables
from rupiahvision.reporting.readme_tables import main, markdown_table, render

BLOCK = "<!-- BEGIN GENERATED: {0} -->\n{1}<!-- END GENERATED: {0} -->"


def test_markdown_table():
    assert markdown_table(["a", "b"], [(1, 2)]) == "| a | b |\n|---|---|\n| 1 | 2 |\n"


def test_render_replaces_only_block_bodies():
    readme = "intro\n" + BLOCK.format("t", "old\n") + "\noutro\n"
    assert render(readme, {"t": "new\n"}) == "intro\n" + BLOCK.format("t", "new\n") + "\noutro\n"


def test_render_is_idempotent():
    readme = BLOCK.format("t", "x\n")
    once = render(readme, {"t": "y\n"})
    assert render(once, {"t": "y\n"}) == once


@pytest.mark.parametrize(
    ("readme", "tables", "message"),
    [
        (BLOCK.format("t", ""), {}, "no generator"),
        ("no blocks here", {"t": ""}, "missing"),
        (BLOCK.format("t", "") + BLOCK.format("t", ""), {"t": ""}, "duplicate"),
    ],
)
def test_render_rejects_mismatched_blocks(readme, tables, message):
    with pytest.raises(ValueError, match=message):
        render(readme, tables)


@pytest.fixture
def repo_copy(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    shutil.copy(REPO_ROOT / "data" / "labels.json", tmp_path / "data" / "labels.json")
    monkeypatch.setattr(readme_tables, "TABLES", {"classes": readme_tables.classes_table})
    return tmp_path


def test_check_fails_on_stale_readme_then_passes_after_write(repo_copy):
    (repo_copy / "README.md").write_text(BLOCK.format("classes", "stale\n"))
    assert main(["--check", "--repo-root", str(repo_copy)]) == 1
    assert main(["--repo-root", str(repo_copy)]) == 0
    assert main(["--check", "--repo-root", str(repo_copy)]) == 0
    assert "Seratus ribu rupiah" in (repo_copy / "README.md").read_text()


def test_committed_readme_is_up_to_date():
    assert main(["--check"]) == 0
