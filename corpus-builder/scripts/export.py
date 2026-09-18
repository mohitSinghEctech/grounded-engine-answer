"""Dump the corpus out of SQLite into files you can read.

SQLite is the working store because UNIQUE (act, section_number) catches
parser bugs that JSON would swallow. But a store you cannot eyeball is a
store you do not check, so this exports the same data in readable form.

    python corpus-builder/scripts/export.py --format jsonl
    python corpus-builder/scripts/export.py --format text --act ITA-2025
    python corpus-builder/scripts/export.py --format index

Output lands in data/export/. Generated, so gitignored - regenerate rather
than commit.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLUMNS = (
    "act, section_number, section_title, text, source_file, "
    "page_start, page_end, maps_to_1961"
)


def rows(db: Path, act: str | None):
    query = f"SELECT {COLUMNS} FROM sections"
    params: tuple = ()

    if act:
        query += " WHERE act = ?"
        params = (act,)

    query += " ORDER BY act, id"

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(query, params)]


def as_jsonl(records: list[dict], out: Path) -> None:
    """One JSON object per line - greppable, diffable, streamable."""
    with out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def as_text(records: list[dict], out: Path) -> None:
    """Plain text with headers - for reading a section end to end."""
    with out.open("w") as handle:
        for record in records:
            handle.write(
                f"{'=' * 78}\n"
                f"{record['act']}  section {record['section_number']}\n"
                f"{record['section_title']}\n"
                f"pages {record['page_start']}-{record['page_end']}\n"
                f"{'=' * 78}\n\n{record['text']}\n\n"
            )


def as_index(records: list[dict], out: Path) -> None:
    """One line per section - the table of contents, for scanning."""
    with out.open("w") as handle:
        for record in records:
            handle.write(
                f"{record['act']:<12} {record['section_number']:<10} "
                f"p{record['page_start']:<5} "
                f"{len(record['text']):>7} chars  {record['section_title']}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    parser.add_argument("--out", type=Path, default=Path("data/export"))
    parser.add_argument("--act", help="ITA-1961 or ITA-2025; omit for both")
    parser.add_argument("--format", choices=("jsonl", "text", "index"), default="index")
    args = parser.parse_args()

    records = rows(args.db, args.act)

    if not records:
        parser.error(f"no sections found{f' for {args.act}' if args.act else ''}")

    args.out.mkdir(parents=True, exist_ok=True)

    stem = args.act.lower() if args.act else "corpus"
    suffix = {"jsonl": "jsonl", "text": "txt", "index": "index.txt"}[args.format]
    path = args.out / f"{stem}.{suffix}"

    {"jsonl": as_jsonl, "text": as_text, "index": as_index}[args.format](records, path)

    print(f"  {len(records)} sections -> {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
