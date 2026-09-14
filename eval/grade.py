"""Grade the one signal a harness cannot: whether the answer is right.

    python eval/grade.py                      # grade eval/runs/baseline.csv
    python eval/grade.py --csv path/to.csv
    python eval/grade.py --id PR-02           # one question
    python eval/grade.py --summary            # rates only, grade nothing
    python eval/grade.py --regrade            # revisit rows already graded

Walks the ungraded rows one at a time, shows what was asked, what was
expected, what came back, and writes your verdict into `answer_correct`.
Progress is saved when you quit, so 40 rows can be graded in several
sittings.

WHAT YOU ARE JUDGING

Groundedness and correctness *relative to the corpus* - not whether the
answer matches Indian tax law as you know it. That distinction is the whole
discipline: an answer that is true in the world but unsupported by the cited
provision is a failure of this system, and an answer that faithfully reports
a provision is a pass even if the corpus itself is out of date. Grading
against your own knowledge instead makes the score unreproducible, which
defeats the point of having one.

Retrieval is already scored separately, so do not fail an answer for citing
too little. Judge what it did with what it had.

  PASS   every claim traces to a cited provision; the substance answers the
         question; figures are quoted correctly; and if the provisions only
         answer part of it, the answer says which part is missing (rule 4)

  FAIL   asserts anything the cited provisions do not support; misattributes
         a figure or a section; silently drops a material part of the
         question; answers a different question; or advises what to do or
         computes a liability (rule 3)

For a question that should be refused, you are judging the refusal: did it
decline for the right reason and say what it could not cover, rather than
declining a question the corpus actually answers.
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path
from typing import Any

import yaml

GRADE_COLUMN = "answer_correct"
NOTE_COLUMN = "grader_note"

VERDICTS = {
    "y": "True",
    "n": "False",
}

WIDTH = 78


def load_expectations(path: Path) -> dict[str, dict[str, Any]]:
    """The question set, keyed by id, for the expectations the CSV omits."""
    questions = yaml.safe_load(path.read_text())["questions"]

    return {q["id"]: q for q in questions}


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # The grader's note has nowhere to live until the first grading pass.
    if NOTE_COLUMN not in fieldnames:
        fieldnames.append(NOTE_COLUMN)

        for row in rows:
            row.setdefault(NOTE_COLUMN, "")

    return rows, fieldnames


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Rewrite in place, via a temporary file so an interrupt cannot truncate."""
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def rule(character: str = "-") -> str:
    return character * WIDTH


def show(row: dict[str, str], question: dict[str, Any] | None, position: str) -> None:
    print(f"\n{rule('=')}")
    print(f"{position}  {row['id']}   {row['category']}")
    print(rule("="))

    print("\nQUESTION")
    print(
        textwrap.fill(
            row["question"], WIDTH, initial_indent="  ", subsequent_indent="  "
        )
    )

    if question:
        expected_sections = question.get("expected_sections") or []
        expected_acts = question.get("expected_acts") or []

        print("\nEXPECTED")
        print(f"  should refuse   {bool(question.get('should_refuse'))}")

        if expected_acts:
            print(f"  acts            {', '.join(expected_acts)}")
        if expected_sections:
            print(f"  sections        {', '.join(expected_sections)}")
        if question.get("answer_contains"):
            print(f"  must contain    {', '.join(question['answer_contains'])}")
        if question.get("tax_year") is not None:
            print(f"  tax year        {question['tax_year']}")

        if question.get("note"):
            print("\n  note:")
            print(
                textwrap.fill(
                    str(question["note"]).strip(),
                    WIDTH,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )

    print("\nWHAT CAME BACK")
    print(f"  cited           {row['cited'] or '(nothing)'}")
    print(f"  cited acts      {row['cited_acts'] or '(none)'}")
    print(f"  refused         {row['refused']}  {row.get('refusal_reason') or ''}")
    print(f"  retrieved       {row['retrieved']} passages")

    scored = [
        (label, row.get(key, ""))
        for label, key in (
            ("retrieval_hit", "retrieval_hit"),
            ("act_correct", "act_correct"),
            ("contains", "contains_expected"),
        )
        if row.get(key)
    ]

    if scored:
        print("  auto-scored     " + "   ".join(f"{k}={v}" for k, v in scored))

    print("\nANSWER")

    answer = row["answer"].strip() or "(empty)"

    for line in textwrap.wrap(answer, WIDTH - 2) or ["(empty)"]:
        print(f"  {line}")


def prompt_verdict(row: dict[str, str]) -> str | None:
    """Ask for a verdict. Returns None when the grader wants to stop."""
    while True:
        try:
            reply = (
                input(
                    "\n  [y] correct   [n] incorrect   [s] skip   "
                    "[q] save and quit\n  > "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if reply in ("q", "quit"):
            return None

        if reply in ("s", "skip", ""):
            return "skip"

        if reply in VERDICTS:
            row[GRADE_COLUMN] = VERDICTS[reply]

            note = input("  note (optional, enter to skip)\n  > ").strip()

            if note:
                row[NOTE_COLUMN] = note

            return "graded"

        print("  Unrecognised. Use y, n, s or q.")


def summarise(rows: list[dict[str, str]]) -> None:
    graded = [r for r in rows if r.get(GRADE_COLUMN) in ("True", "False")]
    passed = [r for r in graded if r[GRADE_COLUMN] == "True"]

    print(f"\n{rule('=')}")
    print(f"answer_correct   {len(passed)}/{len(graded)} graded", end="")

    if graded:
        print(f"   {len(passed) / len(graded):.0%}", end="")

    print(f"   ({len(rows) - len(graded)} ungraded)")
    print(rule("="))

    if not graded:
        return

    categories = sorted({r["category"] for r in graded})

    for category in categories:
        subset = [r for r in graded if r["category"] == category]
        ok = sum(1 for r in subset if r[GRADE_COLUMN] == "True")

        print(f"  {category:<16} {ok:>2}/{len(subset):<2}")

    failures = [r for r in graded if r[GRADE_COLUMN] == "False"]

    if failures:
        print(f"\n  graded incorrect ({len(failures)}):")

        for row in failures:
            note = row.get(NOTE_COLUMN) or ""
            print(f"    {row['id']:<7} {note[:56]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=Path("eval/runs/baseline.csv"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.yaml"))
    parser.add_argument("--id", help="grade only this question")
    parser.add_argument("--category", help="grade only this category")
    parser.add_argument(
        "--regrade",
        action="store_true",
        help="include rows that already carry a verdict",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print rates and exit without grading",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"no such CSV: {args.csv}. Run `make eval-baseline` first.")

    rows, fieldnames = read_rows(args.csv)
    expectations = load_expectations(args.questions)

    if args.summary:
        summarise(rows)
        return

    queue = rows

    if args.id:
        queue = [r for r in queue if r["id"] == args.id]
    if args.category:
        queue = [r for r in queue if r["category"] == args.category]
    if not args.regrade:
        queue = [r for r in queue if r.get(GRADE_COLUMN) not in ("True", "False")]

    # An errored row has no answer to judge; grading it would record a
    # verdict on the harness rather than on the system.
    queue = [r for r in queue if not r.get("error")]

    if not queue:
        print("Nothing left to grade.")
        summarise(rows)
        return

    print(__doc__.split("WHAT YOU ARE JUDGING")[1].strip())
    print(f"\n{len(queue)} row(s) to grade in {args.csv}")

    graded_count = 0

    for index, row in enumerate(queue, 1):
        show(row, expectations.get(row["id"]), f"[{index}/{len(queue)}]")

        outcome = prompt_verdict(row)

        if outcome is None:
            break

        if outcome == "graded":
            graded_count += 1
            # Saved after every verdict, so nothing is lost to a crash or a
            # closed terminal partway through forty rows.
            write_rows(args.csv, rows, fieldnames)

    write_rows(args.csv, rows, fieldnames)

    print(f"\nGraded {graded_count} row(s). Wrote {args.csv}")
    summarise(rows)


if __name__ == "__main__":
    main()
