"""Compare runs, and tell a real change from run-to-run noise.

    python eval/compare.py eval/runs/t1-exact-r*.csv
    python eval/compare.py --label before eval/runs/a*.csv \
                           --label after  eval/runs/b*.csv

A single 40-question run cannot settle a one- or two-question difference.
Seven runs of near-identical configurations produced failure counts from 4
to 8, and the questions that moved - OC-05, YS-07, SY-06, SM-04 - flickered
between runs rather than responding to any change. Meanwhile SM-01, SM-02
and SM-06 failed in all seven. The first group is noise; the second is the
signal, and only repetition separates them.

So: run a configuration two or three times, point this at the CSVs, and read
the per-signal spread. Where two groups are given, a difference matters only
if the ranges do not overlap.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

SIGNALS = (
    ("retrieval hit", "retrieval_hit"),
    ("act correct", "act_correct"),
    ("refused correctly", "refused_correctly"),
    ("grounded", "grounded"),
    ("answer contains", "contains_expected"),
    ("no fabrication", "no_fabrication"),
    ("answer correct", "answer_correct"),
    # The path, not the output. A CSV written before trajectory scoring
    # existed has no such column, and `proportion` returns None for a
    # signal it cannot find - so old runs still compare cleanly against
    # new ones, on the signals both of them carry.
    ("agent used", "agent_used"),
    ("path correct", "path_correct"),
    ("no repeat calls", "no_redundant_calls"),
)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if not row.get("error")]


def proportion(rows: list[dict[str, str]], column: str) -> tuple[int, int] | None:
    """Passes and applicable count, or None when the signal never applies."""
    values = [rows[i][column] for i in range(len(rows)) if rows[i].get(column)]
    values = [v for v in values if v in ("True", "False")]

    if not values:
        return None

    return sum(1 for v in values if v == "True"), len(values)


def failures(rows: list[dict[str, str]]) -> set[str]:
    return {
        row["id"]
        for row in rows
        if row.get("retrieval_hit") == "False"
        or row.get("refused_correctly") == "False"
    }


def describe(label: str, paths: list[Path]) -> dict[str, list[float]]:
    runs = [read(p) for p in paths]

    print(f"\n{'=' * 66}")
    print(f"{label}   {len(runs)} run(s)")
    print("=" * 66)

    rates: dict[str, list[float]] = {}

    for name, column in SIGNALS:
        counted = [proportion(rows, column) for rows in runs]
        counted = [c for c in counted if c]

        if not counted:
            continue

        percentages = [passed / total for passed, total in counted]
        rates[column] = percentages

        spread = (
            f"{min(percentages):.0%}-{max(percentages):.0%}"
            if len(set(percentages)) > 1
            else "stable"
        )

        print(
            f"  {name:<18} {statistics.mean(percentages):>5.0%}"
            f"   {spread:>10}"
            f"   {' '.join(f'{p}/{t}' for p, t in counted)}"
        )

    # Which questions fail every time, and which come and go. The first kind
    # is worth engineering against; the second will move on its own.
    tally = Counter()

    for rows in runs:
        tally.update(failures(rows))

    if tally:
        always = [q for q, n in sorted(tally.items()) if n == len(runs)]
        sometimes = [(q, n) for q, n in sorted(tally.items()) if n < len(runs)]

        if always:
            print(f"\n  fails every run ({len(always)}):  {', '.join(always)}")
        if sometimes:
            flaky = ", ".join(f"{q} {n}/{len(runs)}" for q, n in sometimes)
            print(f"  intermittent:  {flaky}")

    fabricated = Counter()

    for rows in runs:
        for row in rows:
            if row.get("unsupported"):
                fabricated[row["id"]] += 1

    if fabricated:
        invented = ", ".join(
            f"{q} {n}/{len(runs)}" for q, n in sorted(fabricated.items())
        )
        print(f"  invented provisions:  {invented}")

    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="*", type=Path)
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="name a group; repeat with a following list of CSVs to compare two",
    )
    args, extra = parser.parse_known_args()

    groups: list[tuple[str, list[Path]]] = []

    if args.label:
        # --label before a.csv b.csv --label after c.csv d.csv
        current: list[Path] | None = None

        for token in sys.argv[1:]:
            if token == "--label":
                current = None
                continue
            if current is None and token in args.label:
                groups.append((token, []))
                current = groups[-1][1]
                continue
            if current is not None:
                current.append(Path(token))
    else:
        paths = [p for p in args.csvs + [Path(e) for e in extra] if p.suffix == ".csv"]

        if not paths:
            sys.exit("give at least one CSV")

        groups = [("all runs", paths)]

    missing = [p for _, ps in groups for p in ps if not p.exists()]

    if missing:
        sys.exit(f"no such file: {missing[0]}")

    measured = [(label, describe(label, paths)) for label, paths in groups if paths]

    if len(measured) == 2:
        (left_name, left), (right_name, right) = measured

        print(f"\n{'=' * 66}")
        print(f"{left_name} vs {right_name}")
        print("=" * 66)

        for name, column in SIGNALS:
            if column not in left or column not in right:
                continue

            before, after = left[column], right[column]
            change = statistics.mean(after) - statistics.mean(before)

            # Overlapping ranges mean the runs cannot tell the two apart.
            overlap = min(before) <= max(after) and min(after) <= max(before)
            verdict = "within noise" if overlap else "REAL"

            print(f"  {name:<18} {change:>+5.0%}   {verdict}")


if __name__ == "__main__":
    main()
