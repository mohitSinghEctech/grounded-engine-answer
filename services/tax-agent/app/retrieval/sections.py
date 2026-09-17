"""Spotting a section number the question names outright.

Similarity search cannot do this job. Asked "what is section 139 about?" it
returned 139C, 139A and 139B - neighbours in meaning, wrong in law - and
never s.139 itself, because a section number carries almost no semantic
signal. A number the user typed is an exact-match problem, so it gets an
exact lookup.

Pure functions, no dependencies, so the rules are cheap to test.
"""

from __future__ import annotations

import re

#: The cue word, then one section number or a list of them. The 1-3 digit
#: limit is load-bearing: it stops a year being read as a section, so
#: "section 139 and the 2025 Act" yields 139 alone.
_NAMED_SECTION = re.compile(
    r"\b(?:sections?|u/s|s\.)\s*"
    r"([0-9]{1,3}[A-Z]{0,4}"
    r"(?:\s*(?:,|and)\s*[0-9]{1,3}[A-Z]{0,4})*)",
    re.IGNORECASE,
)

#: Splits "80C, 80D and 80G" into its parts.
_LIST_SEPARATOR = re.compile(r"\s*(?:,|and)\s*", re.IGNORECASE)


def named_sections(question: str) -> list[str]:
    """Section numbers the question names outright, upper-cased and sorted.

    Silence is the right answer for a question that only *describes* a
    provision: a false positive drags an irrelevant section to the top of
    the window, where it displaces something the question actually needs.
    """
    found: set[str] = set()

    for match in _NAMED_SECTION.finditer(question):
        for part in _LIST_SEPARATOR.split(match.group(1)):
            if part:
                found.add(part.upper())

    return sorted(found)
