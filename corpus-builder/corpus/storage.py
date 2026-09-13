"""SQLite persistence for parsed sections.

SQLite rather than JSONL because this corpus gets interrogated constantly -
"show me every section in this chapter shorter than 200 characters" is one
SQL line and a throwaway script otherwise. It is stdlib, one file, and Qdrant
runs on it underneath anyway.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from corpus.models import Document, Section

SCHEMA = """
CREATE TABLE IF NOT EXISTS sections (
    id             INTEGER PRIMARY KEY,
    act            TEXT NOT NULL,
    section_number TEXT NOT NULL,
    section_title  TEXT,
    text           TEXT NOT NULL,
    source_file    TEXT NOT NULL,
    char_start     INTEGER NOT NULL,
    char_end       INTEGER NOT NULL,
    page_start     INTEGER NOT NULL,
    page_end       INTEGER NOT NULL,
    maps_to_1961   TEXT,
    -- Catches a detector bug immediately: a duplicate fails the insert
    -- rather than silently doubling a chunk that later pollutes retrieval.
    UNIQUE (act, section_number)
);

CREATE INDEX IF NOT EXISTS idx_sections_act ON sections(act);

CREATE TABLE IF NOT EXISTS footnotes (
    id   INTEGER PRIMARY KEY,
    act  TEXT NOT NULL,
    text TEXT NOT NULL
);
"""


def write(db_path: Path, doc: Document, sections: list[Section]) -> None:
    """Replace one act's sections and footnotes.

    Scoped by act so re-parsing 2025 does not wipe 1961, and idempotent so
    the parser can be re-run freely while its rules are being tuned.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    act = sections[0].act if sections else ""

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)

        conn.execute("DELETE FROM sections WHERE act = ?", (act,))
        conn.execute("DELETE FROM footnotes WHERE act = ?", (act,))

        conn.executemany(
            """
            INSERT INTO sections (
                act, section_number, section_title, text, source_file,
                char_start, char_end, page_start, page_end, maps_to_1961
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.act,
                    s.number,
                    s.title,
                    s.text,
                    s.source_file,
                    s.char_start,
                    s.char_end,
                    s.page_start,
                    s.page_end,
                    s.maps_to_1961,
                )
                for s in sections
            ],
        )

        conn.executemany(
            "INSERT INTO footnotes (act, text) VALUES (?, ?)",
            [(act, note) for note in doc.footnotes],
        )


def read(db_path: Path, act: str) -> list[tuple]:
    """Rows for one act, in document order.

    Returns tuples rather than Sections because the indexer only needs the
    handful of columns that become chunk metadata.
    """
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            """
            SELECT section_number, section_title, text, page_start, page_end,
                   maps_to_1961
            FROM sections WHERE act = ?
            ORDER BY id
            """,
            (act,),
        ).fetchall()
