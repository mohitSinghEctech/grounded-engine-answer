"""Parse a published Act into sections stored in SQLite.

    python corpus-builder/scripts/parse.py \
        --pdf data/raw/ita-2025.pdf --act ITA-2025 --report
    python corpus-builder/scripts/parse.py \
        --pdf data/raw/ita-1961.PDF --act ITA-1961 --db data/corpus.db

Run from the repository root. --report prints statistics without writing,
which is how the detection rules were tuned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import storage  # noqa: E402
from corpus.dialects import detect  # noqa: E402
from corpus.models import Document, Section  # noqa: E402
from corpus.pdf import load  # noqa: E402


def report(doc: Document, sections: list[Section], gaps: list) -> None:
    """Print what was found, and what looks wrong.

    The two useful signals are gaps in the numbering and sections that came
    out implausibly short - both point straight at a detection rule.
    """
    lengths = sorted(len(s.text) for s in sections)
    schedules = [s for s in sections if s.number.startswith("SCH-")]
    mapped = [s for s in sections if s.maps_to_1961]

    print(f"pages       {len(doc.page_starts)}")
    print(f"characters  {len(doc.text):,}")
    print(f"footnotes   {len(doc.footnotes)}")
    print(f"sections    {len(sections) - len(schedules)}")
    print(f"schedules   {len(schedules)}")

    if mapped:
        print(f"1961 maps   {len(mapped)}")

    if lengths:
        median = lengths[len(lengths) // 2]
        print(f"length      min {lengths[0]}  median {median}  max {lengths[-1]:,}")

    if gaps:
        shown = ", ".join(str(g) for g in gaps[:20])
        print(f"gaps        {len(gaps)}: {shown}{' ...' if len(gaps) > 20 else ''}")
    else:
        print("gaps        none")

    short = [s for s in sections if len(s.text) < 200]

    if short:
        print(f"\nsuspiciously short ({len(short)}):")

        for s in short[:10]:
            print(f"  s.{s.number} p{s.page_start} {len(s.text):>5} chars  {s.title!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--act", required=True, help="ITA-2025 or ITA-1961")
    parser.add_argument("--db", type=Path, help="write to this SQLite file")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    doc = load(args.pdf)
    sections, gaps = detect(doc, args.act, str(args.pdf))

    if args.report or not args.db:
        report(doc, sections, gaps)

    if args.db:
        storage.write(args.db, doc, sections)
        print(f"\nwrote {len(sections)} sections to {args.db}")


if __name__ == "__main__":
    main()
