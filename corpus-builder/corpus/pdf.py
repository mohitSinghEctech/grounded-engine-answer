"""Stage 1-2: PDF file to clean text.

Extraction is only half the job. A published Act arrives with kerning
artifacts, words split across line breaks, and glyphs that map to the wrong
character. Each fix below exists because it was found in a real document,
and each is applied per page so that character offsets stay valid.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pypdf

from corpus.models import Document

logging.disable(logging.WARNING)


# --- Cleanup patterns -------------------------------------------------------
# Every one of these was found by reading the extracted text, not guessed.

#: Invisible hyphen used for optional line breaks; carries no meaning.
SOFT_HYPHEN = "­"

#: Non-breaking and thin spaces that should behave like ordinary spaces.
ODD_SPACE = re.compile(r"[   \t]")

#: Justified text leaves runs of spaces mid-line.
MULTI_SPACE = re.compile(r" {2,}")

#: "( a)" is a kerning artifact; the printed Act reads "(a)". Removing the
#: space restores the source rather than altering it.
CLAUSE_MARKER = re.compile(r"\(\s+([a-zA-Z]{1,4}|\d{1,3})\)")

#: Words split across a line break: "es -\ntablished" -> "established".
#: The 2025 Act has 877 of these; requiring a lowercase continuation keeps
#: genuine dashes before a new clause intact.
HYPHEN_BREAK = re.compile(r"(\w) *- *\n([a-z])")

#: The rupee sign lives in a Symbol font and extracts as a backtick. All 228
#: occurrences in the 2025 Act are followed by a digit, so this is unambiguous.
RUPEE = re.compile(r"`\s*(?=[\d,])")

#: The 1961 PDF maps the em dash to U+20AC, 1006 times.
BAD_DASH = re.compile("€")

#: Amendment notes printed at the foot of a page. Kept, but separated from
#: the body so they are never embedded as if they were law.
FOOTNOTE = re.compile(r"^\d{1,2}\.\s+(Inserted|Substituted|Omitted|Renumbered)\b")


def clean_page(text: str) -> str:
    """Apply every cleanup to one page of extracted text.

    Line structure is preserved on purpose: the 2025 Act puts a section's
    title on the line *before* its number, and collapsing lines here would
    destroy that signal before detection can use it.
    """
    text = text.replace(SOFT_HYPHEN, "")
    text = ODD_SPACE.sub(" ", text)

    lines = [MULTI_SPACE.sub(" ", line.strip()) for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)

    text = CLAUSE_MARKER.sub(r"(\1)", text)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = RUPEE.sub("₹", text)
    text = BAD_DASH.sub("—", text)

    return text


def load(pdf_path: Path) -> Document:
    """Read a PDF into a single cleaned string.

    Cleaning happens per page, before the pages are joined, so that
    ``page_starts`` still points at the right place afterwards. Doing it the
    other way round shifts every offset and ``page_for_offset`` starts lying.
    """
    reader = pypdf.PdfReader(pdf_path)

    pages: list[str] = []
    page_starts: list[int] = []
    footnotes: list[str] = []
    cursor = 0

    for page in reader.pages:
        body_lines = []

        for line in clean_page(page.extract_text() or "").split("\n"):
            if FOOTNOTE.match(line):
                footnotes.append(line)
            else:
                body_lines.append(line)

        body = "\n".join(body_lines)

        page_starts.append(cursor)
        pages.append(body)
        cursor += len(body) + 1  # +1 for the "\n" that join() inserts

    return Document(
        text="\n".join(pages),
        page_starts=page_starts,
        footnotes=footnotes,
    )


def line_offsets(lines: list[str]) -> list[int]:
    """Character offset at which each line starts.

    Detection works line by line but records offsets into the whole document,
    so it needs this lookup.
    """
    offsets: list[int] = []
    running = 0

    for line in lines:
        offsets.append(running)
        running += len(line) + 1

    return offsets
