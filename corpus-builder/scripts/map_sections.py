"""Populate maps_to_1961 by matching section titles across the two Acts.

    python corpus-builder/scripts/map_sections.py --db data/corpus.db
    python corpus-builder/scripts/map_sections.py --db data/corpus.db --write

The 2025 Act prints its 1961 counterpart in the heading itself - "Title.
[S. 92B of the 1961 Act]" - and the parser reads it, but the source PDF
carries that bracket for only **4 of 552** sections. So SM-01 and SM-02
("section 80C corresponds to which section of the 2025 Act?") failed every
eval run since the first baseline: the mapping the question asks for was
simply not in the index.

The mapping is written in BOTH directions - maps_to_1961 on the 2025 rows
and maps_to_2025 on the 1961 rows. The first version wrote only the forward
one and was inert: every section-mapping question asks 1961 -> 2025, so the
agent's map_section tool looked up 1961 s.80C, found nothing, and fell back
to searching. Populating 319 mappings changed nothing until the inverse
existed.

Titles are the way in, because Parliament reused them. Three strategies,
each requiring a UNIQUE answer - a wrong mapping is worse than none, since
the agent would then cite the wrong section with full confidence:

  exact   normalised titles identical                     273
  prefix  one title is the start of the other             15
  fuzzy   >= 0.90 similar and only one candidate          30

Normalisation strips "in respect of" against "for", and "etc", because
those are drafting boilerplate rather than meaning. That single change is
what connects 2025 s.123 to 1961 s.80C, whose titles differ only in that
phrase and in the extra instruments 1961 lists after it.

Left unmapped on purpose: 15 where a 1961 title is reused by several
sections, 214 with no candidate at all, and 5 whose titles are too short
to carry a signal ("Definitions", "Interpretation"). Those are honest
gaps - the agent's map_section tool returns found=false for them, and its
description then tells the model to search the other Act by subject
instead of guessing a number.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import storage  # noqa: E402  - after the path insert above

#: Words that differ between the two Acts without changing the provision.
#: "in respect of" vs "for" is the big one; "etc" and "certain cases" are
#: tails that one Act prints and the other does not.
BOILERPLATE = re.compile(
    r"\b(the|of|in|for|and|to|a|an|or|any|on|respect|etc|certain|cases)\b"
)

#: A title this short carries no signal - "Definitions" appears in both
#: Acts many times over, and matching on it would be worse than a gap.
MIN_TITLE = 8

#: Prefix matching needs enough text to mean something. "Deduction" is the
#: start of dozens of headings; "Deduction life insurance premia" is not.
MIN_PREFIX = 25

#: difflib ratio. Deliberately high: below this, near-misses start pairing
#: provisions that merely sound alike.
FUZZY = 0.90


def normalise(title: str | None) -> str:
    """Lowercase, drop punctuation and bracketed refs, remove boilerplate."""
    text = (title or "").lower()
    text = re.sub(r"\[.*?\]", " ", text)  # the "[S. 92B of the 1961 Act]" hint
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = BOILERPLATE.sub(" ", text)

    return " ".join(text.split())


def load(connection: sqlite3.Connection, act: str) -> list[tuple[str, str]]:
    return [
        (number, title or "")
        for number, title in connection.execute(
            "SELECT section_number, section_title FROM sections WHERE act = ?",
            (act,),
        )
    ]


class Matcher:
    """Finds the 1961 section a 2025 title refers to, or nothing."""

    def __init__(self, sections_1961: list[tuple[str, str]]):
        self.by_title: dict[str, list[str]] = defaultdict(list)

        for number, title in sections_1961:
            key = normalise(title)

            if key:
                self.by_title[key].append(number)

        self.keys = list(self.by_title)

    def find(self, title: str) -> tuple[str | None, str]:
        key = normalise(title)

        if len(key) < MIN_TITLE:
            return None, "too-short"

        if key in self.by_title:
            candidates = self.by_title[key]

            # A title several 1961 sections share cannot identify one of
            # them. Reported rather than guessed at.
            if len(candidates) > 1:
                return None, "ambiguous"

            return candidates[0], "exact"

        if len(key) >= MIN_PREFIX:
            # One heading is the start of the other: the same provision,
            # where one Act enumerates more instruments than the other.
            near = [
                other
                for other in self.keys
                if len(other) >= MIN_PREFIX
                and (other.startswith(key) or key.startswith(other))
            ]
            numbers = {n for other in near for n in self.by_title[other]}

            if len(numbers) == 1:
                return numbers.pop(), "prefix"

        close = difflib.get_close_matches(key, self.keys, n=2, cutoff=FUZZY)

        if len(close) == 1 and len(self.by_title[close[0]]) == 1:
            return self.by_title[close[0]][0], "fuzzy"

        return None, "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    parser.add_argument(
        "--write",
        action="store_true",
        help="apply the mappings; without it, only report what would change",
    )
    parser.add_argument("--show", type=int, default=8, help="sample this many mappings")
    args = parser.parse_args()

    connection = sqlite3.connect(args.db)

    sections_2025 = load(connection, "ITA-2025")
    matcher = Matcher(load(connection, "ITA-1961"))

    before = connection.execute(
        "SELECT COUNT(*) FROM sections "
        "WHERE act = 'ITA-2025' AND maps_to_1961 IS NOT NULL"
    ).fetchone()[0]

    how: Counter[str] = Counter()
    found: list[tuple[str, str, str]] = []

    for number, title in sections_2025:
        counterpart, strategy = matcher.find(title)
        how[strategy] += 1

        if counterpart:
            found.append((number, counterpart, strategy))

    print(f"ITA-2025 sections           {len(sections_2025)}")
    print(f"already mapped by the PDF   {before}")
    print(f"mapped by title             {len(found)}\n")

    for strategy in ("exact", "prefix", "fuzzy", "ambiguous", "none", "too-short"):
        if how[strategy]:
            print(f"  {strategy:<12} {how[strategy]:>4}")

    print(f"\nsample of {args.show}:")
    for number, counterpart, strategy in found[: args.show]:
        print(f"  2025 s.{number:<7} -> 1961 s.{counterpart:<9} [{strategy}]")

    if not args.write:
        print("\nreport only. pass --write to apply.")
        return

    storage.migrate(connection)

    # The heading's own bracketed reference is the Act speaking for itself,
    # so it always wins over an inferred title match.
    connection.executemany(
        "UPDATE sections SET maps_to_1961 = ? "
        "WHERE act = 'ITA-2025' AND section_number = ? AND maps_to_1961 IS NULL",
        [(counterpart, number) for number, counterpart, _ in found],
    )

    # And the inverse, onto the 1961 rows. Without this the mapping is
    # only traversable 2025 -> 1961, and every section-mapping question in
    # the eval set asks the other way: "section 80C corresponds to which
    # section of the 2025 Act?" reaches for 1961 s.80C, where nothing was
    # written. A smoke test caught it - map_section returned found=false
    # on the one question the forward mapping existed to fix.
    #
    # Read back from the table rather than from `found`, so the 4 mappings
    # the PDF supplied are inverted too.
    pairs = connection.execute(
        "SELECT section_number, maps_to_1961 FROM sections "
        "WHERE act = 'ITA-2025' AND maps_to_1961 IS NOT NULL"
    ).fetchall()

    # A 1961 section two 2025 sections both claim cannot name one of them,
    # so it is left null - the same rule the forward direction uses for an
    # ambiguous title, and for the same reason.
    claimed: dict[str, list[str]] = {}

    for number_2025, number_1961 in pairs:
        claimed.setdefault(number_1961, []).append(number_2025)

    unique = {k: v[0] for k, v in claimed.items() if len(v) == 1}
    contested = {k: v for k, v in claimed.items() if len(v) > 1}

    connection.executemany(
        "UPDATE sections SET maps_to_2025 = ? "
        "WHERE act = 'ITA-1961' AND section_number = ?",
        [(number_2025, number_1961) for number_1961, number_2025 in unique.items()],
    )
    connection.commit()

    after = connection.execute(
        "SELECT COUNT(*) FROM sections "
        "WHERE act = 'ITA-2025' AND maps_to_1961 IS NOT NULL"
    ).fetchone()[0]

    reverse = connection.execute(
        "SELECT COUNT(*) FROM sections "
        "WHERE act = 'ITA-1961' AND maps_to_2025 IS NOT NULL"
    ).fetchone()[0]

    print(f"\nwritten. maps_to_1961 populated for {after} of {len(sections_2025)}")
    print(f"         maps_to_2025 populated for {reverse} sections of the 1961 Act")

    if contested:
        print(f"         {len(contested)} left null - two 2025 sections claim them:")

        for number_1961, numbers in list(contested.items())[:5]:
            print(f"           1961 s.{number_1961} <- {', '.join(numbers)}")

    print("\nre-index to carry it into the payload: make reindex (server)")
    print("                                       make index-standalone (on-disk)")


if __name__ == "__main__":
    main()
