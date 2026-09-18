"""Render the /ask/stream event stream as readable lines.

    make ask-stream Q="what is section 139 about?"

Exists because Server-Sent Events are unpleasant to read raw, and because
seeing the timings is the quickest way to understand where a request's
time actually goes: embedding is a network call, the vector search is
local and nearly free, and the model is most of the wall clock.
"""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
import time

#: Steps that mark the beginning of something slow, so the elapsed time
#: printed against the matching end step is the cost of that one thing.
QUIET = {"resolving", "verifying"}


def _print_wrapped(text: str, indent: str = " " * 11) -> None:
    """The whole answer, wrapped, paragraph breaks preserved."""
    width = max(40, min(shutil.get_terminal_size((100, 24)).columns, 100))

    for paragraph in text.split("\n"):
        if not paragraph.strip():
            print()
            continue

        for line in textwrap.wrap(paragraph, width - len(indent)):
            print(f"{indent}{line}")


def main() -> None:
    started = time.perf_counter()
    event = "message"

    for raw in sys.stdin:
        line = raw.rstrip("\n")

        if line.startswith("event: "):
            event = line[len("event: ") :]
            continue

        if not line.startswith("data: "):
            continue

        payload = json.loads(line[len("data: ") :])
        step = payload.pop("step", "?")
        elapsed = f"{time.perf_counter() - started:6.2f}s"

        if event == "result":
            cites = ", ".join(
                f"{c['act']} s.{c['section_number']}" for c in payload["citations"]
            )
            invented = payload.get("unsupported_citations") or []

            print(
                f"{elapsed}  ANSWER   refused={payload['refused']} "
                f"reason={payload['refusal_reason']} "
                f"finish={payload.get('finish_reason')}"
            )
            print(f"{'':9} cites    {cites or '(none)'}")

            if invented:
                print(f"{'':9} INVENTED {', '.join(invented)}")

            # Printed in full and wrapped. An earlier version sliced this to
            # 300 characters, which read exactly like a truncated answer -
            # the opposite of what a viewer is for.
            print(f"{'':9} text")
            _print_wrapped(payload["answer"])
            continue

        if event == "error":
            print(f"{elapsed}  ERROR    {payload}")
            continue

        if step in QUIET and not payload:
            continue

        # "sections" carries a list of dicts on the retrieved step; print it
        # as one line per provision rather than as a wall of JSON.
        sections = payload.pop("sections", None)
        detail = " ".join(f"{k}={v}" for k, v in payload.items())

        print(f"{elapsed}  {step:<14} {detail}")

        if isinstance(sections, list):
            for entry in sections:
                if isinstance(entry, dict):
                    print(
                        f"{'':9}   {entry['act']:<14} s.{str(entry['section']):<12}"
                        f" {str(entry['title'])[:46]:<46} {entry['score']}"
                    )
                else:
                    print(f"{'':9}   {entry}")


if __name__ == "__main__":
    main()
