"""The three things that move through the pipeline.

All are frozen: each describes something that already happened, so nothing
downstream should be able to quietly rewrite it.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A PDF after extraction and cleanup.

    Sections routinely span page breaks, so the whole document is held as one
    string rather than a list of pages. ``page_starts`` is what lets a
    character offset be mapped back to a page for hand-checking.
    """

    text: str
    page_starts: list[int]
    footnotes: list[str]

    def page_for_offset(self, offset: int) -> int:
        """Return the 1-based page number containing ``offset``."""
        return bisect.bisect_right(self.page_starts, offset)


@dataclass(frozen=True)
class Section:
    """One numbered provision of an Act, or one schedule.

    ``number`` is text rather than an integer because the two Acts number
    differently: 2025 uses 1..536, while 1961 has 80C, 80-IA and 115JD.
    Schedules use SCH-I (2025) or SCH-1 (1961).
    """

    act: str
    number: str
    title: str
    text: str
    source_file: str
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    maps_to_1961: str | None = None


@dataclass
class Chunk:
    """A piece of text small enough to embed, plus the metadata that travels
    with it into Qdrant.

    The payload is what makes an answer citable. A chunk without a section
    number can be retrieved but never cited, which is the whole difference
    between the two chunking strategies.
    """

    text: str
    payload: dict = field(default_factory=dict)
