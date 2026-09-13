"""Section detection, one module per Act.

The two Acts are structurally different documents. A single parser with
flags would hide that; separate modules make each set of rules readable on
its own and let one change without risking the other.

    ITA-2025   title on the line BEFORE the number, integers 1..536
    ITA-1961   title on the SAME line, alphanumeric 80C / 80-IA / 115JD
"""

from __future__ import annotations

from collections.abc import Callable

from corpus.dialects import ita1961, ita2025
from corpus.models import Document, Section

#: Which detector handles which act.
DIALECTS: dict[str, Callable[[Document, str, str], tuple[list[Section], list[str]]]] = {
    "ITA-2025": ita2025.detect,
    "ITA-1961": ita1961.detect,
}


def detect(doc: Document, act: str, source: str) -> tuple[list[Section], list[str]]:
    """Run the detector for ``act``.

    Returns the sections found and a list of gaps worth investigating -
    missing section numbers for 2025, large jumps in numbering for 1961.
    """
    if act not in DIALECTS:
        raise SystemExit(f"No dialect for {act!r}. Known: {sorted(DIALECTS)}")

    return DIALECTS[act](doc, act, source)
