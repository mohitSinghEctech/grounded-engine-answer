"""Stage 3: sections to chunks small enough to embed.

Two strategies, kept side by side on purpose. The naive one is the baseline
the section-aware one is measured against, and having both is what turns
"section chunking is better" from an assertion into a number.
"""

from __future__ import annotations

import re
from pathlib import Path

from corpus.models import Chunk
from corpus.storage import read

#: The 1961 Act was repealed on 1 April 2026 and the 2025 Act commenced the
#: same day. Years are the starting calendar year of the financial year, and
#: None means "open ended" at that end.
ACT_YEARS = {
    "ITA-1961": (None, 2025),
    "ITA-2025": (2026, None),
}

#: Sub-section markers at the start of a line. 1961 writes "(1)Where" with no
#: space, so the trailing whitespace is optional.
SUBSECTION = re.compile(r"(?m)^\((\d{1,2})\)\s*")

#: Shorter than this and a "section" is an omitted stub - "443. 9[***]" -
#: which carries no law and can only ever produce a wrong citation.
MIN_CHUNK_CHARS = 40

#: Upper bound for one chunk. Large enough to hold most whole sections,
#: small enough that a retrieved passage is readable in a prompt.
MAX_CHUNK_CHARS = 2400

#: Separators tried in order by the naive splitter, largest structure first.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Split on the largest natural separator that fits, then slide a window.

    The same idea as LangChain's RecursiveCharacterTextSplitter, without the
    dependency. It knows nothing about sections, which is exactly the point:
    this is the baseline.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    for separator in SEPARATORS:
        parts = text.split(separator)

        if len(parts) == 1:
            continue

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else current + separator + part

            if len(candidate) <= size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(part) > size:
                chunks.extend(split_recursive(part, size, overlap))
                current = ""
            else:
                current = part

        if current:
            chunks.append(current)

        if overlap and len(chunks) > 1:
            merged = [chunks[0]]

            for chunk in chunks[1:]:
                merged.append(merged[-1][-overlap:] + chunk)

            chunks = merged

        return [c for c in chunks if c.strip()]

    return [text[i : i + size] for i in range(0, len(text), size - overlap)]


def split_on_subsections(text: str, limit: int) -> list[str]:
    """Split an oversized section at sub-section boundaries.

    Legal text has natural joints - (1), (2), (3) - and cutting anywhere else
    severs a provision from its own conditions. Falls back to the naive
    splitter only when a section has no sub-sections at all.
    """
    if len(text) <= limit:
        return [text]

    marks = [m.start() for m in SUBSECTION.finditer(text) if m.start() > 0]

    if not marks:
        return split_recursive(text, limit, 0)

    bounds = [0] + marks + [len(text)]

    parts: list[str] = []
    current = ""

    for i in range(len(bounds) - 1):
        piece = text[bounds[i] : bounds[i + 1]]

        if len(current) + len(piece) <= limit:
            current += piece
            continue

        if current:
            parts.append(current)

        if len(piece) <= limit:
            current = piece
        else:
            current = ""
            parts.extend(split_recursive(piece, limit, 0))

    if current:
        parts.append(current)

    return [p for p in parts if p.strip()]


def from_sections(db_path: Path, act: str) -> list[Chunk]:
    """Section-aware chunks: every one knows which provision it came from."""
    year_from, year_to = ACT_YEARS.get(act, (None, None))

    chunks: list[Chunk] = []

    for number, title, text, page_start, page_end, maps_to in read(db_path, act):
        if len(text) < MIN_CHUNK_CHARS:
            continue

        parts = split_on_subsections(text, MAX_CHUNK_CHARS)

        for index, part in enumerate(parts):
            # Repeat the heading on every part, so a passage taken from the
            # middle of a long section still says what it belongs to.
            body = f"{act} section {number}. {title}.\n\n{part.strip()}"

            chunks.append(
                Chunk(
                    text=body,
                    payload={
                        "act": act,
                        "strategy": "sections",
                        "section_number": number,
                        "section_title": title,
                        "part": index,
                        "part_count": len(parts),
                        "page_start": page_start,
                        "page_end": page_end,
                        "tax_year_from": year_from,
                        "tax_year_to": year_to,
                        # Not "maps_to_1961": on an ITA-1961 chunk the
                        # counterpart is a 2025 section, and a key that
                        # names one direction cannot hold both.
                        "maps_to": maps_to,
                        "text": body,
                    },
                )
            )

    return chunks


def from_pdf(pdf_path: Path, act: str, size: int, overlap: int) -> list[Chunk]:
    """Naive chunks: fixed size, no structure, no citation possible.

    Kept as the baseline. Note the payload still carries the act's tax years,
    so the only difference between the two strategies is the chunking itself
    rather than whether filtering exists at all.
    """
    from corpus.pdf import load

    doc = load(pdf_path)
    year_from, year_to = ACT_YEARS.get(act, (None, None))

    chunks: list[Chunk] = []
    cursor = 0

    for index, body in enumerate(split_recursive(doc.text, size, overlap)):
        position = doc.text.find(body[:60], cursor)

        if position >= 0:
            cursor = position

        chunks.append(
            Chunk(
                text=body,
                payload={
                    "act": act,
                    "strategy": "naive",
                    "chunk_index": index,
                    "page": doc.page_for_offset(cursor),
                    "tax_year_from": year_from,
                    "tax_year_to": year_to,
                    "text": body,
                },
            )
        )

    return chunks
