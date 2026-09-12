"""Parse an Income-tax Act PDF into sections stored in SQLite.

Local batch script. Never imported by the service.

    python services/tax-agent/scripts/parse.py \
        --pdf data/raw/ita-2025.pdf --act ITA-2025 --report
    python services/tax-agent/scripts/parse.py \
        --pdf data/raw/ita-2025.pdf --act ITA-2025 --db data/corpus.db
"""

from __future__ import annotations

import argparse
import bisect
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pypdf

logging.disable(logging.WARNING)


# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_SOFT_HYPHEN = "­"
_ODD_SPACE = re.compile(r"[   \t]")
_MULTI_SPACE = re.compile(r" {2,}")
_CLAUSE_MARKER = re.compile(r"\(\s+([a-zA-Z]{1,4}|\d{1,3})\)")
_HYPHEN_BREAK = re.compile(r"(\w) *- *\n([a-z])")

# The rupee sign lives in a Symbol font and extracts as a backtick.
_RUPEE = re.compile(r"`\s*(?=[\d,])")
_FOOTNOTE = re.compile(r"^\d{1,2}\.\s+(Inserted|Substituted|Omitted|Renumbered)\b")
_SECTION_START = re.compile(r"^(\d{1,5})\.\s+(\S.*)$")
_MAX_SECTION = 536
_SCHEDULE = re.compile(r"^SCHEDULE\s+([IVXL]+)$")

# "18[Deduction in respect of..." - an amendment marker prefixing a title.
_TITLE_LEAD = re.compile(r"^\d{1,3}\[")

# "Meaning of international transaction. [S. 92B of the 1961 Act]"
_TITLE_MAPPING = re.compile(r"\[S+\.?\s*([^\]]+?)\s+of the 1961 Act\]\s*$")

_QUOTE_CHARS = "“‘\"'"

# Section-number lines closer together than this are table rows, not sections.
_TABLE_GAP = 4


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    text: str
    page_starts: list[int]
    footnotes: list[str]

    def page_for_offset(self, offset: int) -> int:
        """1-based page number containing this character offset."""
        return bisect.bisect_right(self.page_starts, offset)


@dataclass(frozen=True)
class Section:
    act: str
    number: str
    title: str
    text: str
    source_file: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    maps_to_1961: str | None


# --------------------------------------------------------------------------
# Stage 1 + 2: extract and normalise
# --------------------------------------------------------------------------


def normalise_page(text: str) -> str:
    text = text.replace(_SOFT_HYPHEN, "")
    text = _ODD_SPACE.sub(" ", text)

    lines = []
    for line in text.split("\n"):
        line = _MULTI_SPACE.sub(" ", line.strip())
        if line:
            lines.append(line)

    text = "\n".join(lines)

    # "( a)" is a kerning artifact; the printed Act reads "(a)".
    text = _CLAUSE_MARKER.sub(r"(\1)", text)

    # "es -\ntablished" -> "established"
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # "` 150000" -> "₹150000"
    text = _RUPEE.sub("\u20b9", text)

    return text


def load(pdf_path: Path) -> Document:
    """Extract and normalise, recording page offsets and setting footnotes aside."""
    reader = pypdf.PdfReader(pdf_path)

    parts: list[str] = []
    page_starts: list[int] = []
    footnotes: list[str] = []
    cursor = 0

    for page in reader.pages:
        clean = normalise_page(page.extract_text() or "")

        body_lines = []
        for line in clean.split("\n"):
            if _FOOTNOTE.match(line):
                footnotes.append(line)
            else:
                body_lines.append(line)

        body = "\n".join(body_lines)

        page_starts.append(cursor)
        parts.append(body)
        cursor += len(body) + 1  # +1 for the "\n" that join() inserts

    return Document(
        text="\n".join(parts),
        page_starts=page_starts,
        footnotes=footnotes,
    )


# --------------------------------------------------------------------------
# Stage 3: detect section boundaries
# --------------------------------------------------------------------------


def title_before(lines: list[str], index: int) -> tuple[str, int] | None:
    """Collect the marginal note preceding a section-number line.

    Titles wrap across lines, may be prefixed by an amendment marker
    ("18[..."), may open with a quotation mark, and may end with a
    cross-reference to the 1961 Act instead of a full stop.
    """
    if index == 0:
        return None

    parts: list[str] = []
    i = index - 1
    first = index

    while i >= 0 and len(parts) < 4:
        line = lines[i]

        if not line or line.startswith("(") or _SECTION_START.match(line):
            break

        parts.insert(0, line)
        first = i

        head = _TITLE_LEAD.sub("", line)

        if head and (head[0].isupper() or head[0] in _QUOTE_CHARS):
            break

        i -= 1

    if not parts:
        return None

    title = _TITLE_LEAD.sub("", " ".join(parts).strip())

    if not title or len(title) > 250:
        return None
    if not (title[0].isupper() or title[0] in _QUOTE_CHARS):
        return None

    return title, first


def split_mapping(title: str) -> tuple[str, str | None]:
    """Separate 'Title. [S. 92B of the 1961 Act]' into title and mapping."""
    match = _TITLE_MAPPING.search(title)

    if not match:
        return title, None

    return title[: match.start()].strip(), match.group(1)


def resolve_number(raw: str, last_number: int) -> int | None:
    """Turn a matched digit run into a section number.

    An amendment marker can fuse to the number: "5207." is marker 5 on
    section 207. Strip at most two leading digits, and only when the
    remainder has no leading zero and is at least three digits - which
    rejects the years and amounts that also start a line ("2026.", "25000.").
    """
    number = int(raw)

    if 1 <= number <= _MAX_SECTION:
        return number

    for cut in (1, 2):
        tail = raw[cut:]

        if len(tail) < 3 or tail.startswith("0"):
            continue

        candidate = int(tail)

        if 1 <= candidate <= _MAX_SECTION:
            return candidate

    return None


def detect(doc: Document, act: str, source: str) -> tuple[list[Section], list[int]]:
    """Return sections (including schedules) and the section numbers not found.

    Candidates are split into two tiers. A section-number line sitting within
    a few lines of another one is almost always a table row, so those go into
    tier B and are only used to fill genuine gaps left by tier A.
    """
    lines = doc.text.split("\n")
    number_lines = {i for i, line in enumerate(lines) if _SECTION_START.match(line)}

    line_offsets: list[int] = []
    running = 0
    for line in lines:
        line_offsets.append(running)
        running += len(line) + 1

    tier_a: list[tuple[int, int, str]] = []
    tier_b: list[tuple[int, int, str]] = []
    schedules_found: list[tuple[int, str]] = []

    offset = 0

    for index, line in enumerate(lines):
        schedule = _SCHEDULE.match(line)

        if schedule:
            schedules_found.append((offset, schedule.group(1)))

        if index in number_lines:
            found = title_before(lines, index)

            if found is not None:
                title, title_line = found
                raw = _SECTION_START.match(line).group(1)
                crowded = any(
                    j in number_lines for j in range(max(0, index - _TABLE_GAP), index)
                )
                titled = title.endswith(".") or title.endswith("]")
                bucket = tier_a if (titled and not crowded) else tier_b
                bucket.append((offset, raw, title, line_offsets[title_line]))

        offset += len(line) + 1

    chain: list[tuple[int, int, str]] = []
    last_number = 0

    for offset_a, raw, title, title_offset in tier_a:
        number = resolve_number(raw, last_number)

        if number is not None and number > last_number:
            chain.append((offset_a, number, title, title_offset))
            last_number = number

    # Fill gaps from tier B, but only where the offset falls between the
    # neighbouring accepted sections. A table row far from its gap is rejected.
    accepted = {n: o for o, n, _, _ in chain}

    for offset_b, raw, title, title_offset in tier_b:
        number = resolve_number(raw, 0)

        if number is None or number in accepted:
            continue

        lower = max((o for n, o in accepted.items() if n < number), default=-1)
        upper = min(
            (o for n, o in accepted.items() if n > number), default=len(doc.text) + 1
        )

        if lower < offset_b < upper:
            chain.append((offset_b, number, title, title_offset))
            accepted[number] = offset_b

    chain.sort()

    # The last section must stop at the first schedule, not run to EOF.
    corpus_end = schedules_found[0][0] if schedules_found else len(doc.text)

    boundaries: list[tuple[int, str, str, int]] = [
        (o, str(n), t, ts) for o, n, t, ts in chain if o < corpus_end
    ]
    boundaries += [(o, f"SCH-{r}", f"Schedule {r}.", o) for o, r in schedules_found]
    boundaries.sort()

    sections: list[Section] = []

    for i, (start, number, raw_title, _) in enumerate(boundaries):
        # End where the NEXT section's title begins, so a section never
        # carries the following section's heading in its body.
        end = boundaries[i + 1][3] if i + 1 < len(boundaries) else len(doc.text)
        title, mapping = split_mapping(raw_title)
        body = doc.text[start:end].strip()

        # "5207." is an amendment marker fused to the number; restore "207."
        if number.isdigit() and not body.startswith(f"{number}."):
            body = re.sub(r"^\d{1,5}\.", f"{number}.", body, count=1)

        sections.append(
            Section(
                act=act,
                number=number,
                title=title.rstrip("."),
                text=body,
                source_file=source,
                char_start=start,
                char_end=end,
                page_start=doc.page_for_offset(start),
                page_end=doc.page_for_offset(max(start, end - 1)),
                maps_to_1961=mapping,
            )
        )

    numeric = {int(s.number) for s in sections if s.number.isdigit()}
    missing = [n for n in range(1, max(numeric, default=0) + 1) if n not in numeric]

    return sections, missing


# --------------------------------------------------------------------------
# Stage 4: persist
# --------------------------------------------------------------------------

_SCHEMA = """
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
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)

        act = sections[0].act if sections else ""

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


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def report(doc: Document, sections: list[Section], missing: list[int]) -> None:
    lengths = sorted(len(s.text) for s in sections)
    schedules = [s for s in sections if s.number.startswith("SCH-")]
    mapped = [s for s in sections if s.maps_to_1961]

    print(f"pages       {len(doc.page_starts)}")
    print(f"characters  {len(doc.text):,}")
    print(f"footnotes   {len(doc.footnotes)}")
    print(f"sections    {len(sections) - len(schedules)}")
    print(f"schedules   {len(schedules)}")
    print(f"1961 maps   {len(mapped)}")

    if lengths:
        print(
            f"length      min {lengths[0]}  "
            f"median {lengths[len(lengths) // 2]}  "
            f"max {lengths[-1]:,}"
        )

    if missing:
        shown = ", ".join(str(n) for n in missing[:30])
        more = " ..." if len(missing) > 30 else ""
        print(f"missing     {len(missing)}: {shown}{more}")
    else:
        print("missing     none")

    suspicious = [s for s in sections if len(s.text) < 200]

    if suspicious:
        print(f"\nsuspiciously short ({len(suspicious)}):")
        for s in suspicious[:10]:
            print(f"  s.{s.number} p{s.page_start} {len(s.text):>5} chars  {s.title!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--act", required=True, help="e.g. ITA-2025")
    parser.add_argument("--db", type=Path, help="write to this SQLite file")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    doc = load(args.pdf)
    sections, missing = detect(doc, args.act, str(args.pdf))

    if args.report or not args.db:
        report(doc, sections, missing)

    if args.db:
        write(args.db, doc, sections)
        print(f"\nwrote {len(sections)} sections to {args.db}")


if __name__ == "__main__":
    main()
