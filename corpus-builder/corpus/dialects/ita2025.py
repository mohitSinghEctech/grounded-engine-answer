"""Section detection for the Income-tax Act, 2025.

Shape of the document::

    Income deemed to accrue or arise in India.      <- marginal note
    9.  (1) The income referred to in sub-sections  <- number, then body

The title sits on the line before the number, wraps across up to four lines,
and the Act is numbered 1..536 with no gaps. Schedules announce themselves
with a bare ``SCHEDULE I`` heading.
"""

from __future__ import annotations

import re

from corpus.models import Document, Section
from corpus.pdf import line_offsets

#: A line that opens a section: a number, a full stop, then text. Allows up
#: to five digits because an amendment marker can fuse to the number
#: ("5207." is marker 5 on section 207).
SECTION_START = re.compile(r"^(\d{1,5})\.\s+(\S.*)$")

#: The Act's highest section number, used to spot fused markers.
MAX_SECTION = 536

#: A bare "SCHEDULE I" heading.
SCHEDULE = re.compile(r"^SCHEDULE\s+([IVXL]+)$")

#: An amendment marker prefixing a title: "18[Deduction in respect of...".
TITLE_LEAD = re.compile(r"^\d{1,3}\[")

#: Some titles end with a pointer to the old Act, which is worth keeping.
TITLE_MAPPING = re.compile(r"\[S+\.?\s*([^\]]+?)\s+of the 1961 Act\]\s*$")

#: Titles may open with a quotation mark: '"Transfer" ... defined.'
QUOTE_CHARS = "“‘\"'"

#: Section-number lines closer together than this are table rows, not
#: sections. Real sections are hundreds of lines apart.
TABLE_GAP = 4


def resolve_number(raw: str) -> int | None:
    """Turn a matched digit run into a section number, or None.

    An amendment marker can fuse to the number, so "5207." means section 207.
    Strip at most two leading digits, and only when what remains has no
    leading zero and is at least three digits - which rejects the years and
    amounts that also begin a line ("2026.", "25000.").
    """
    number = int(raw)

    if 1 <= number <= MAX_SECTION:
        return number

    for cut in (1, 2):
        tail = raw[cut:]

        if len(tail) < 3 or tail.startswith("0"):
            continue

        if 1 <= int(tail) <= MAX_SECTION:
            return int(tail)

    return None


def title_before(lines: list[str], index: int) -> tuple[str, int] | None:
    """Collect the marginal note preceding a section-number line.

    Returns the note and the index of its first line, so the previous
    section can be made to end where this one's title begins.

    Titles wrap across lines, may be prefixed by an amendment marker, may
    open with a quotation mark, and may end with a cross-reference to the
    1961 Act instead of a full stop.
    """
    if index == 0:
        return None

    parts: list[str] = []
    i = index - 1
    first = index

    while i >= 0 and len(parts) < 4:
        line = lines[i]

        if not line or line.startswith("(") or SECTION_START.match(line):
            break

        parts.insert(0, line)
        first = i

        head = TITLE_LEAD.sub("", line)

        if head and (head[0].isupper() or head[0] in QUOTE_CHARS):
            break

        i -= 1

    if not parts:
        return None

    title = TITLE_LEAD.sub("", " ".join(parts).strip())

    if not title or len(title) > 250:
        return None

    if not (title[0].isupper() or title[0] in QUOTE_CHARS):
        return None

    return title, first


def split_mapping(title: str) -> tuple[str, str | None]:
    """Separate "Title. [S. 92B of the 1961 Act]" into title and mapping."""
    match = TITLE_MAPPING.search(title)

    if not match:
        return title, None

    return title[: match.start()].strip(), match.group(1)


def detect(doc: Document, act: str, source: str) -> tuple[list[Section], list[int]]:
    """Find every section and schedule in the 2025 Act.

    Candidates are sorted into two tiers, because no single test separates
    sections from table rows:

        tier A   has a title, and no other section-number line within
                 TABLE_GAP lines above it
        tier B   everything else with a title - used only to fill gaps left
                 by tier A, and only where the offset falls between the
                 accepted neighbours

    Tier B exists because some genuine sections do follow closely, and some
    genuine titles lack a trailing full stop. Rejecting them outright loses
    real sections; accepting them freely lets table rows in.
    """
    lines = doc.text.split("\n")
    offsets = line_offsets(lines)
    number_lines = {i for i, line in enumerate(lines) if SECTION_START.match(line)}

    tier_a: list[tuple[int, str, str, int]] = []
    tier_b: list[tuple[int, str, str, int]] = []
    schedules: list[tuple[int, str]] = []

    for index, line in enumerate(lines):
        schedule = SCHEDULE.match(line)

        if schedule:
            schedules.append((offsets[index], schedule.group(1)))

        if index not in number_lines:
            continue

        found = title_before(lines, index)

        if found is None:
            continue

        title, title_line = found
        raw = SECTION_START.match(line).group(1)

        crowded = any(
            j in number_lines for j in range(max(0, index - TABLE_GAP), index)
        )
        complete = title.endswith(".") or title.endswith("]")

        bucket = tier_a if (complete and not crowded) else tier_b
        bucket.append((offsets[index], raw, title, offsets[title_line]))

    chain: list[tuple[int, int, str, int]] = []
    last_number = 0

    for offset, raw, title, title_offset in tier_a:
        number = resolve_number(raw)

        if number is not None and number > last_number:
            chain.append((offset, number, title, title_offset))
            last_number = number

    accepted = {number: offset for offset, number, _, _ in chain}

    for offset, raw, title, title_offset in tier_b:
        number = resolve_number(raw)

        if number is None or number in accepted:
            continue

        lower = max((o for n, o in accepted.items() if n < number), default=-1)
        upper = min(
            (o for n, o in accepted.items() if n > number), default=len(doc.text) + 1
        )

        if lower < offset < upper:
            chain.append((offset, number, title, title_offset))
            accepted[number] = offset

    chain.sort()

    # The last section must stop at the first schedule, not run to EOF.
    corpus_end = schedules[0][0] if schedules else len(doc.text)

    boundaries: list[tuple[int, str, str, int]] = [
        (o, str(n), t, ts) for o, n, t, ts in chain if o < corpus_end
    ]
    boundaries += [(o, f"SCH-{r}", f"Schedule {r}.", o) for o, r in schedules]
    boundaries.sort()

    sections: list[Section] = []

    for i, (start, number, raw_title, _) in enumerate(boundaries):
        # End where the NEXT section's title begins, so a section never
        # carries the following section's heading in its body.
        end = boundaries[i + 1][3] if i + 1 < len(boundaries) else len(doc.text)

        title, mapping = split_mapping(raw_title)
        body = doc.text[start:end].strip()

        # Restore "207." from a fused "5207.".
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
