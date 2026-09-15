"""Run the evaluation set against a running tax-agent and score the results.

    python eval/run.py                          # against the default collection
    python eval/run.py --limit 5                # smoke test
    python eval/run.py --category out-of-corpus # one slice
    python eval/run.py --delay 0                # no pacing, if quota allows
    python eval/run.py --out eval/runs/baseline.csv

Scores four objective signals per question. Answer correctness is not scored
here - it needs judgment, and an LLM judge would only substitute its own
error rate. Grade the CSV's `answer` column by hand on the first run.

Paced, and retries while the service reports a rate limit. A 429 means the
question was never asked: recording it as a result would score the quota
rather than the system, and a run with holes in it cannot be compared
against the next one.

The service must already be running; this does not start anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Statuses worth asking again, all of which mean the question never got an
# answer: 429 the upstream model rate limited us, 503 it was unreachable,
# 504 it timed out. A 502 or a 422 mean "not like that" and will not improve
# on a second try.
RETRY_STATUS = frozenset({429, 503, 504})

# However long the server asks us to wait, never stall a whole run on it.
MAX_BACKOFF_SECONDS = 60.0


@dataclass
class Result:
    """One question's outcome: what was asked, what came back, how it scored."""

    id: str
    category: str
    question: str

    answer: str = ""
    refused: bool = False
    refusal_reason: str = ""
    cited: list[str] = field(default_factory=list)
    cited_acts: list[str] = field(default_factory=list)
    # Provisions the model cited but was never given.
    unsupported: list[str] = field(default_factory=list)
    retrieved: int = 0

    latency_ms: int = 0
    retrieval_ms: int = 0
    llm_ms: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    error: str | None = None

    # How many times this question had to be asked again. A run full of
    # retries is a quota-bound run, and its latencies mean nothing.
    retries: int = 0

    # Scores. None means "not applicable to this question".
    retrieval_hit: bool | None = None
    act_correct: bool | None = None
    refused_correctly: bool | None = None
    grounded: bool | None = None
    contains_expected: bool | None = None
    # False when the answer invented a provision reference. Scored, not
    # just logged: a fabricated citation reads exactly like a real one.
    no_fabrication: bool | None = None


def ask(url: str, question: str, tax_year: int | None, timeout: float) -> dict:
    """POST one question to /ask and return the parsed response."""
    body = {"question": question, "max_tokens": 1200}

    if tax_year is not None:
        body["tax_year"] = tax_year

    request = urllib.request.Request(
        f"{url}/ask",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def retry_after(exc: urllib.error.HTTPError) -> float | None:
    """The server's own backoff advice, when it sends any.

    The tax-agent forwards the upstream's Retry-After on a 429, so prefer it
    over guessing; it is the only number here that knows the real quota.
    """
    header = exc.headers.get("Retry-After") if exc.headers else None

    try:
        return min(float(header), MAX_BACKOFF_SECONDS)
    except (TypeError, ValueError):
        return None


def ask_with_retry(
    url: str,
    question: str,
    tax_year: int | None,
    timeout: float,
    max_retries: int,
    base_delay: float,
) -> tuple[dict, int]:
    """POST one question, asking again while the service says "not now".

    Returns the payload and the number of retries it took, so a run can
    report whether it was quota-bound.
    """
    for attempt in range(max_retries + 1):
        try:
            return ask(url, question, tax_year, timeout), attempt
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt >= max_retries:
                raise

            wait = retry_after(exc) or min(
                max(base_delay, 1.0) * 2**attempt, MAX_BACKOFF_SECONDS
            )

            print(
                f"          HTTP {exc.code}, waiting {wait:.0f}s "
                f"(retry {attempt + 1}/{max_retries})",
                flush=True,
            )
            time.sleep(wait)

    # Unreachable: the final attempt either returns or re-raises above.
    raise RuntimeError("retry loop exhausted without result")


def score(question: dict, payload: dict) -> Result:
    """Compare one response against the question's expectations.

    Every check is objective. A signal that does not apply - act_correct on
    a question that must refuse, for instance - stays None rather than
    counting as a pass, so the summary is not quietly inflated.
    """
    citations = payload.get("citations") or []

    result = Result(
        id=question["id"],
        category=question["category"],
        question=question["question"],
        answer=payload.get("answer", ""),
        refused=bool(payload.get("refused")),
        refusal_reason=payload.get("refusal_reason") or "",
        cited=[c.get("section_number") for c in citations if c.get("section_number")],
        cited_acts=sorted({c.get("act") for c in citations if c.get("act")}),
        unsupported=payload.get("unsupported_citations") or [],
        retrieved=payload.get("retrieved", 0),
        latency_ms=payload.get("latency_ms", 0),
        retrieval_ms=payload.get("retrieval_ms", 0),
        llm_ms=payload.get("llm_ms"),
        total_tokens=payload.get("total_tokens"),
        finish_reason=payload.get("finish_reason"),
    )

    # Applies to every question, refusals included: inventing a source
    # while declining is still inventing a source.
    result.no_fabrication = not result.unsupported

    should_refuse = bool(question.get("should_refuse"))
    result.refused_correctly = result.refused == should_refuse

    if should_refuse:
        # Nothing else is meaningful: a correct refusal cites nothing.
        return result

    result.grounded = len(citations) > 0

    expected_sections = question.get("expected_sections") or []

    if expected_sections:
        # A hit means the right provision was cited, not that every expected
        # one was: a partial answer still found the law.
        result.retrieval_hit = any(s in result.cited for s in expected_sections)

    expected_acts = question.get("expected_acts") or []

    if expected_acts and result.cited_acts:
        result.act_correct = all(a in expected_acts for a in result.cited_acts)

    phrases = question.get("answer_contains") or []

    if phrases:
        # Compared with thousands separators stripped. The Acts print the
        # same figure two ways - "₹150000" in the 2025 Act, "1,25,000" in the
        # guidance - and models reformat freely, so a literal match measures
        # formatting luck rather than whether the right value was stated.
        lowered = _normalise_digits(result.answer.lower())
        result.contains_expected = all(
            _normalise_digits(p.lower()) in lowered for p in phrases
        )

    return result


def _normalise_digits(text: str) -> str:
    """Drop separators that sit between two digits, and nothing else."""
    return re.sub(r"(?<=\d)[,\s](?=\d)", "", text)


def rate(results: list[Result], attribute: str) -> tuple[int, int]:
    """Passes and applicable count for one signal."""
    applicable = [getattr(r, attribute) for r in results]
    applicable = [v for v in applicable if v is not None]

    return sum(1 for v in applicable if v), len(applicable)


def summarise(results: list[Result]) -> None:
    ok = [r for r in results if not r.error]

    print(f"\n{'=' * 62}")
    print(f"{len(ok)} answered, {len(results) - len(ok)} errored")
    print(f"{'=' * 62}")

    for label, attribute in (
        ("retrieval hit   ", "retrieval_hit"),
        ("act correct     ", "act_correct"),
        ("refused correctly", "refused_correctly"),
        ("grounded        ", "grounded"),
        ("answer contains ", "contains_expected"),
        ("no fabrication  ", "no_fabrication"),
    ):
        passed, total = rate(ok, attribute)

        if total:
            print(f"  {label} {passed:>3}/{total:<3} {passed / total:>6.0%}")

    print()

    for category in sorted({r.category for r in ok}):
        rows = [r for r in ok if r.category == category]
        hit, hit_total = rate(rows, "retrieval_hit")
        ref, ref_total = rate(rows, "refused_correctly")

        parts = [f"  {category:<16} n={len(rows):<3}"]

        if hit_total:
            parts.append(f"retrieval {hit}/{hit_total}")
        if ref_total:
            parts.append(f"refusal {ref}/{ref_total}")

        print("  ".join(parts))

    # Why answers were withheld. A refusal that should have happened and one
    # that should not look identical in the rate above; they do not here.
    reasons = Counter(r.refusal_reason for r in ok if r.refused and r.refusal_reason)

    if reasons:
        print("\n  refusals by reason:")

        for reason, count in reasons.most_common():
            print(f"    {reason:<20} {count:>3}")

    latencies = [r.latency_ms for r in ok if r.latency_ms]
    tokens = [r.total_tokens for r in ok if r.total_tokens]

    if latencies:
        print(
            f"\n  latency  median {statistics.median(latencies):>6.0f} ms   "
            f"max {max(latencies):>6} ms"
        )

        # Where that time went, so a slow run points at a layer instead of
        # prompting a round of guessing.
        retrievals = [r.retrieval_ms for r in ok if r.retrieval_ms]
        generations = [r.llm_ms for r in ok if r.llm_ms]

        if retrievals:
            print(
                f"    retrieval  median {statistics.median(retrievals):>6.0f} ms   "
                f"max {max(retrievals):>6} ms"
            )

        if generations:
            print(
                f"    llm        median {statistics.median(generations):>6.0f} ms   "
                f"max {max(generations):>6} ms"
            )

    if tokens:
        print(
            f"  tokens   total {sum(tokens):>7,}   "
            f"median {statistics.median(tokens):>5.0f}"
        )

    retried = [r for r in results if r.retries]

    if retried:
        # Latency includes no waiting - the server timed itself - but a run
        # that had to wait this often was shaped by quota, not by the system.
        print(
            f"  retries  {sum(r.retries for r in retried)} across "
            f"{len(retried)} question(s): "
            f"{', '.join(r.id for r in retried)}"
        )

    truncated = [r.id for r in ok if r.finish_reason == "length"]

    if truncated:
        print(f"\n  ⚠ truncated answers: {', '.join(truncated)}")

    fabricated = [r for r in ok if r.unsupported]

    if fabricated:
        print(f"\n  ⚠ invented provisions ({len(fabricated)}):")

        for r in fabricated:
            print(f"    {r.id:<7} {', '.join(r.unsupported)}")

    failures = [
        r for r in ok if r.retrieval_hit is False or r.refused_correctly is False
    ]

    if failures:
        print(f"\n  failures ({len(failures)}):")

        for r in failures:
            reason = []
            if r.retrieval_hit is False:
                reason.append(f"cited {r.cited or 'nothing'}")
            if r.refused_correctly is False:
                reason.append("refused" if r.refused else "did not refuse")

            print(f"    {r.id:<7} {r.category:<16} {'; '.join(reason)}")


def write_csv(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "category",
                "question",
                "refused",
                "refusal_reason",
                "retrieved",
                "cited",
                "cited_acts",
                "unsupported",
                "retrieval_hit",
                "act_correct",
                "refused_correctly",
                "grounded",
                "contains_expected",
                "no_fabrication",
                "latency_ms",
                "retrieval_ms",
                "llm_ms",
                "total_tokens",
                "finish_reason",
                "retries",
                "error",
                "answer_correct",  # left blank, for grading by hand
                "answer",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    r.id,
                    r.category,
                    r.question,
                    r.refused,
                    r.refusal_reason,
                    r.retrieved,
                    " ".join(r.cited),
                    " ".join(r.cited_acts),
                    " ".join(r.unsupported),
                    r.retrieval_hit,
                    r.act_correct,
                    r.refused_correctly,
                    r.grounded,
                    r.contains_expected,
                    r.no_fabrication,
                    r.latency_ms,
                    r.retrieval_ms,
                    r.llm_ms,
                    r.total_tokens,
                    r.finish_reason,
                    r.retries,
                    r.error,
                    "",
                    " ".join(r.answer.split()),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.yaml"))
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--out", type=Path, default=Path("eval/runs/latest.csv"))
    parser.add_argument("--category", help="run only this category")
    parser.add_argument("--limit", type=int, help="first N questions only")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="seconds to wait between questions, to stay under the quota",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="times to re-ask a question the service rate limited",
    )
    args = parser.parse_args()

    questions = yaml.safe_load(args.questions.read_text())["questions"]

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]

    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        sys.exit("no questions selected")

    print(
        f"running {len(questions)} questions against {args.url}\n"
        f"  pacing {args.delay:.0f}s between questions, "
        f"up to {args.max_retries} retries on a rate limit\n"
    )

    results: list[Result] = []

    for index, question in enumerate(questions, 1):
        if index > 1 and args.delay > 0:
            time.sleep(args.delay)

        started = time.perf_counter()
        retries = 0

        try:
            payload, retries = ask_with_retry(
                args.url,
                question["question"],
                question.get("tax_year"),
                args.timeout,
                args.max_retries,
                args.delay,
            )
            result = score(question, payload)
        except urllib.error.HTTPError as exc:
            # A retryable status only escapes once every retry is spent.
            retries = args.max_retries if exc.code in RETRY_STATUS else 0
            result = Result(
                id=question["id"],
                category=question["category"],
                question=question["question"],
                error=f"HTTP {exc.code}",
            )
        except Exception as exc:
            result = Result(
                id=question["id"],
                category=question["category"],
                question=question["question"],
                error=type(exc).__name__,
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        result.retries = retries
        results.append(result)

        mark = "!" if result.error else ("." if result.refused_correctly else "F")
        note = result.error or (result.refusal_reason if result.refused else "")
        # Flushed: a paced run takes minutes, and buffered progress that
        # appears only at the end is no progress at all.
        print(
            f"  [{index:>2}/{len(questions)}] {result.id:<7} {mark}  "
            f"{elapsed:>6}ms  cited={result.cited or '-'}"
            f"{'  ' + note if note else ''}",
            flush=True,
        )

    summarise(results)
    write_csv(results, args.out)

    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
