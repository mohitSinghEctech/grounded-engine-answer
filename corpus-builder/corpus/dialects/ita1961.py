"""Section detection for the Income-tax Act, 1961.

Shape of the document::

    115F. Capital gains on transfer of foreign exchange assets not to be
    charged in certain cases.
    (1)Where, in the case of an assessee being a non-resident Indian...

The title shares the number's line, wraps onto the next few, and is
terminated either by a dash before the body or by an amendment bracket.
Numbering is alphanumeric and has genuine gaps, so "the next number" is no
guide; ordering is enforced with a sort key instead.

The schedules carry no extractable heading at all - they announce themselves
by restarting the numbering at 1.
"""

from __future__ import annotations

import re

from corpus.models import Document, Section
from corpus.pdf import line_offsets

#: "115F.", "80C.", "80-IA." followed by the marginal note.
SECTION_START = re.compile(r"^(\d{1,3}[A-Z]{0,4}(?:-[A-Z]{1,3})?)\.\s*(\S.*)$")

#: The note ends at a dash ("Title. - Body") or at an amendment bracket
#: ("Title. [Substituted by Act 10 of 2000...").
TITLE_THEN_BODY = re.compile(r"^(.{6,240}?\.)\s*(?:[-–—]\s|\[)")

#: An amendment bracket can also open the note: "10A. [ Special provision...".
LEADING_BRACKET = re.compile(r"^\d{0,3}\[\s*")

#: Openings that mean "this is body text", not a marginal note.
NOT_A_TITLE = re.compile(
    r"^(Explanation|Provided|For the purposes|In this|Where|Notwithstanding|"
    r"Omitted|Substituted|Inserted)\b"
)

#: Below this the numbering has restarted, which means a schedule has begun.
SCHEDULE_RESTART = 3


def sort_key(number: str) -> tuple[int, str]:
    """Order 1961 numbers: 80 < 80A < 80AA < 80C < 80-IA < 81.

    The numeric part dominates and the letter suffix breaks ties, with
    hyphens ignored so 80-IA sorts with 80IA.
    """
    match = re.match(r"(\d+)(.*)", number)

    if not match:
        return (0, number)

    return (int(match.group(1)), match.group(2).replace("-", ""))


def looks_like_marginal_note(title: str) -> bool:
    """Whether a candidate reads like a section heading rather than prose."""
    if not title or len(title) > 250:
        return False

    if not title[0].isupper():
        return False

    if not title.rstrip().endswith("."):
        return False

    if NOT_A_TITLE.match(title):
        return False

    # A heading is a phrase, not a table row full of figures.
    digits = sum(character.isdigit() for character in title)

    return digits / len(title) < 0.2


def marginal_note(lines: list[str], index: int, rest: str) -> str | None:
    """Extract the marginal note from the text following a section number.

    The note may finish on the same line, wrap onto the next few, and may be
    prefixed by an amendment bracket.
    """
    joined = LEADING_BRACKET.sub("", rest.strip())

    for step in range(1, 4):
        if TITLE_THEN_BODY.match(joined) or joined.rstrip().endswith("."):
            break

        if index + step >= len(lines):
            break

        following = lines[index + step]

        if not following or SECTION_START.match(following):
            break

        joined = f"{joined} {following.strip()}"

    match = TITLE_THEN_BODY.match(joined)

    if match:
        candidate = match.group(1)
    elif joined.rstrip().endswith("."):
        candidate = joined.rstrip()
    else:
        return None

    return candidate if looks_like_marginal_note(candidate) else None


def find_schedule_starts(
    lines: list[str], offsets: list[int], after: int, last_numeric: int
) -> list[int]:
    """Offsets where each schedule begins.

    There is no heading to match, so the signal is the numbering resetting:
    a low-numbered marginal note appearing after a high-numbered one means a
    new schedule has started.
    """
    starts: list[int] = []
    previous = last_numeric

    for index, line in enumerate(lines):
        if offsets[index] <= after:
            continue

        match = SECTION_START.match(line)

        if not match or marginal_note(lines, index, match.group(2)) is None:
            continue

        numeric = sort_key(match.group(1))[0]

        if numeric <= SCHEDULE_RESTART < previous:
            starts.append(offsets[index])

        previous = numeric

    return starts


def detect(doc: Document, act: str, source: str) -> tuple[list[Section], list[str]]:
    """Find every section and schedule in the 1961 Act.

    Explanations and table rows reuse earlier numbers, so a strictly
    increasing sort key rejects them without needing to know what they are.
    """
    lines = doc.text.split("\n")
    offsets = line_offsets(lines)

    chain: list[tuple[int, str, str]] = []
    last_key: tuple[int, str] = (0, "")

    for index, line in enumerate(lines):
        match = SECTION_START.match(line)

        if not match:
            continue

        title = marginal_note(lines, index, match.group(2))

        if title is None:
            continue

        key = sort_key(match.group(1))

        if key <= last_key:
            continue

        chain.append((offsets[index], match.group(1), title))
        last_key = key

    last_offset = chain[-1][0] if chain else 0
    last_numeric = sort_key(chain[-1][1])[0] if chain else 0

    schedule_starts = find_schedule_starts(lines, offsets, last_offset, last_numeric)
    corpus_end = schedule_starts[0] if schedule_starts else len(doc.text)

    blocks = list(chain)
    blocks += [
        (offset, f"SCH-{ordinal}", f"Schedule {ordinal}")
        for ordinal, offset in enumerate(schedule_starts, start=1)
    ]
    blocks.sort()

    sections: list[Section] = []

    for i, (start, number, title) in enumerate(blocks):
        if number[0].isdigit() and start >= corpus_end:
            continue

        end = blocks[i + 1][0] if i + 1 < len(blocks) else len(doc.text)

        sections.append(
            Section(
                act=act,
                number=number,
                title=title.rstrip("."),
                text=doc.text[start:end].strip(),
                source_file=source,
                char_start=start,
                char_end=end,
                page_start=doc.page_for_offset(start),
                page_end=doc.page_for_offset(max(start, end - 1)),
            )
        )

    # 1961 numbering has real gaps, so report large jumps rather than a list
    # of every absent number.
    numbers = [s.number for s in sections if s.number[0].isdigit()]
    gaps = [
        f"{numbers[i]}->{numbers[i + 1]}"
        for i in range(len(numbers) - 1)
        if sort_key(numbers[i + 1])[0] - sort_key(numbers[i])[0] > 3
    ]

    return sections, gaps
