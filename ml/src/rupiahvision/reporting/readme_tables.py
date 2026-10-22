"""Generate the README's tables from files in the repo, and check they are up to date.

Every table in README.md sits between a pair of markers:

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

Only this script writes between them. CI runs it with --check, which fails if the
README differs from what the source files (labels.json, results/*.json) say.

Usage:
    uv run python -m rupiahvision.reporting.readme_tables          # rewrite README.md
    uv run python -m rupiahvision.reporting.readme_tables --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rupiahvision import REPO_ROOT
from rupiahvision.labels import load_schema

BLOCK_RE = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[\w-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def classes_table(repo_root: Path) -> str:
    schema = load_schema(repo_root / "data" / "labels.json")
    rows = [
        (
            f"Rp{d:,}".replace(",", "."),
            ", ".join(schema.series),
            schema.speech["id-ID"][str(d)],
            schema.speech["en-US"][str(d)],
        )
        for d in schema.denominations
    ]
    rows.append(
        (
            "(no note / not confident)",
            "-",
            schema.speech["id-ID"]["unknown"],
            schema.speech["en-US"]["unknown"],
        )
    )
    return markdown_table(["Denomination", "Series", "Spoken (id-ID)", "Spoken (en-US)"], rows)


TABLES: dict[str, Callable[[Path], str]] = {
    "classes": classes_table,
}


def render(readme: str, tables: dict[str, str]) -> str:
    """Replace each generated block's body; every block and every table must match up."""
    found = [m.group("name") for m in BLOCK_RE.finditer(readme)]
    if len(found) != len(set(found)):
        raise ValueError(f"duplicate generated blocks in README: {found}")
    unknown = set(found) - set(tables)
    if unknown:
        raise ValueError(f"README has blocks with no generator: {sorted(unknown)}")
    missing = set(tables) - set(found)
    if missing:
        raise ValueError(f"README is missing generated blocks: {sorted(missing)}")
    return BLOCK_RE.sub(lambda m: m.group(1) + tables[m.group("name")] + m.group(4), readme)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is stale")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    readme_path = args.repo_root / "README.md"
    current = readme_path.read_text(encoding="utf-8")
    updated = render(current, {name: fn(args.repo_root) for name, fn in TABLES.items()})

    if args.check:
        if updated != current:
            print(
                "README.md is out of date. Run: "
                "uv run python -m rupiahvision.reporting.readme_tables",
                file=sys.stderr,
            )
            return 1
        print("README.md tables are up to date.")
        return 0

    if updated != current:
        readme_path.write_text(updated, encoding="utf-8")
        print(f"Updated {readme_path}")
    else:
        print("README.md already up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
