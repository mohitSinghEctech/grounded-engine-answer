"""Chunk Income Tax Department guidance pages.

These are not statute. The department publishes them to explain which return
applies, what the slabs are, and which deductions exist - in the vocabulary
people actually use. They are indexed under a separate act label so an answer
can never present guidance as law.

Every page carries the department's own disclaimer, and it is repeated on
every chunk rather than kept once at the top: a passage retrieved from the
middle of a page must still say what it is.
"""

from __future__ import annotations

import re
from pathlib import Path

from corpus.models import Chunk

#: The label these chunks are indexed under. Deliberately not an Act.
#:
#: Was ITD-GUIDANCE, one letter from the ITA- prefix the statute uses, and
#: gpt-4.1-mini reliably wrote "ITA-GUIDANCE" instead - a citation for a
#: provision it was never given, which split_citations then discarded. Six
#: correct answers in the first baseline scored as refusals because of it.
#: DEPT- shares no prefix with ITA- and still has the hyphen the citation
#: regex requires.
ACT = "DEPT-GUIDANCE"

#: Separates the provenance header written by fetch_guidance.py.
HEADER_RULE = "-" * 72

#: "AY 2026-27" means the assessment year; the income year it assesses is the
#: one before it, and that is what tax_year filters on.
ASSESSMENT_YEAR = re.compile(r"\bAY\s*(20\d{2})\s*[-–]\s*\d{2}", re.IGNORECASE)

#: Portal navigation repeated on every page.
NAVIGATION = (
    "e-filing and Centralized Processing Center",
    "Tax Information Network",
    "Guidance to file Tax Return",
    "Return / Forms applicable to me",
    "Deductions on which I can get tax benefit",
    "Senior / Super Senior Citizen",
    "Hindu Undivided Family (HUF)",
    "Tax Professionals & Others",
    "| Income Tax Department",
)

#: A line reads as a heading when it is short and does not end like prose.
MAX_HEADING_CHARS = 90

#: Target chunk size. Smaller than the statute chunks because guidance is
#: denser - a slab table says more per character than a proviso does.
MAX_CHUNK_CHARS = 1400


def split_header(text: str) -> tuple[dict[str, str], str]:
    """Separate the provenance header from the page body."""
    if HEADER_RULE not in text:
        return {}, text

    head, _, body = text.partition(HEADER_RULE)

    meta = {}
    for line in head.split("\n"):
        key, sep, value = line.partition(": ")
        if sep:
            meta[key.strip().lower()] = value.strip()

    return meta, body.strip()


def tax_year_for(text: str) -> int | None:
    """Income year the page covers, from its assessment year.

    "AY 2026-27" assesses income earned in FY 2025-26, so the tax year is
    2025. Getting this backwards would file every page under the wrong Act.
    """
    match = ASSESSMENT_YEAR.search(text)

    return int(match.group(1)) - 1 if match else None


def is_navigation(line: str) -> bool:
    return any(fragment in line for fragment in NAVIGATION)


def is_heading(line: str) -> bool:
    """Whether a line introduces a block rather than being content.

    Slab tables are the trap here: "₹ 8,00,000 - ₹ 12,00,000  10%" is short
    and unpunctuated, so a naive length test calls it a heading and discards
    the rates - which are the single most valuable thing on these pages.
    A heading is a phrase; a row of figures is data.
    """
    stripped = line.strip()

    if not stripped or len(stripped) > MAX_HEADING_CHARS:
        return False

    if stripped.endswith((".", ",", ";", ":")):
        return False

    if "\u20b9" in stripped or "%" in stripped:
        return False

    digits = sum(character.isdigit() for character in stripped)

    return digits / len(stripped) <= 0.35


def from_guidance(path: Path) -> list[Chunk]:
    """Turn one guidance page into citable chunks.

    Chunks break at headings where possible and by size otherwise. Each
    carries the page it came from and the heading it sits under, so a
    citation names something a reader can actually find on the page.
    """
    meta, body = split_header(path.read_text(errors="ignore"))

    lines = [
        line.strip()
        for line in body.split("\n")
        if line.strip() and not is_navigation(line)
    ]

    if not lines:
        return []

    page_title = lines[0]
    year = tax_year_for(body)

    blocks: list[tuple[str, list[str]]] = []
    heading = page_title
    current: list[str] = []

    for line in lines[1:]:
        if is_heading(line) and current:
            blocks.append((heading, current))
            heading, current = line, []
        elif is_heading(line):
            heading = line
        else:
            current.append(line)

    if current:
        blocks.append((heading, current))

    chunks: list[Chunk] = []

    for block_index, (section_heading, paragraphs) in enumerate(blocks):
        text = "\n".join(paragraphs)

        if len(text) < 80:
            continue

        parts = [
            text[i : i + MAX_CHUNK_CHARS] for i in range(0, len(text), MAX_CHUNK_CHARS)
        ]

        for index, part in enumerate(parts):
            # The label and disclaimer ride on every chunk, so a passage can
            # never be quoted as if it were the Act.
            body_text = (
                f"Income Tax Department guidance - {page_title}\n"
                f"{section_heading}\n\n{part.strip()}\n\n"
                f"(Departmental guidance, not statute. "
                f"Source: {meta.get('source', 'incometax.gov.in')})"
            )

            chunks.append(
                Chunk(
                    text=body_text,
                    payload={
                        "act": ACT,
                        "strategy": "guidance",
                        "section_number": path.stem,
                        "section_title": section_heading[:200],
                        "block": block_index,
                        "part": index,
                        "part_count": len(parts),
                        "page_start": None,
                        "page_end": None,
                        "tax_year_from": year,
                        "tax_year_to": year,
                        "source_url": meta.get("source"),
                        "text": body_text,
                    },
                )
            )

    return chunks


def from_directory(directory: Path) -> list[Chunk]:
    """Every guidance page in a directory."""
    chunks: list[Chunk] = []

    for path in sorted(directory.glob("*.txt")):
        chunks.extend(from_guidance(path))

    return chunks
