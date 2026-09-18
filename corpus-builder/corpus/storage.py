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
    -- The counterpart section in the OTHER Act, one column per direction.
    -- maps_to_1961 sits on ITA-2025 rows and comes from the 2025 Act's own
    -- headings plus scripts/map_sections.py; maps_to_2025 sits on ITA-1961
    -- rows and is the inverse of it.
    --
    -- Both exist because the lookup is asked in both directions and a
    -- single column served only one. Every section-mapping question in the
    -- eval set asks 1961 -> 2025 ("section 80C corresponds to which
    -- section of the 2025 Act?"), which the forward column cannot answer:
    -- the mapping is recorded on 2025 s.123, not on 1961 s.80C. Filling
    -- 319 forward mappings changed nothing for those questions until the
    -- inverse was written too.
    maps_to_1961   TEXT,
    maps_to_2025   TEXT,
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


def migrate(conn: sqlite3.Connection) -> None:
    """Add columns a DB built by an older schema is missing.

    CREATE TABLE IF NOT EXISTS silently does nothing to an existing table,
    so a new column has to be added explicitly or every read of it fails
    on a corpus that was parsed before it existed. Re-parsing instead
    would mean re-reading two PDFs to add one nullable column.
    """
    have = {row[1] for row in conn.execute("PRAGMA table_info(sections)")}

    if have and "maps_to_2025" not in have:
        conn.execute("ALTER TABLE sections ADD COLUMN maps_to_2025 TEXT")


def write(db_path: Path, doc: Document, sections: list[Section]) -> None:
    """Replace one act's sections and footnotes.

    Scoped by act so re-parsing 2025 does not wipe 1961, and idempotent so
    the parser can be re-run freely while its rules are being tuned.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    act = sections[0].act if sections else ""

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        migrate(conn)

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
                   -- Whichever direction this Act's rows carry. A section
                   -- belongs to exactly one Act, so its counterpart is
                   -- always in the other one and the target Act is implied.
                   COALESCE(maps_to_1961, maps_to_2025) AS maps_to
            FROM sections WHERE act = ?
            ORDER BY id
            """,
            (act,),
        ).fetchall()
