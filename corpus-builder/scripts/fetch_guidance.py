"""Download official Income Tax Department guidance pages as text.

The Acts state the law but never explain which return a salaried individual
files, or what the slabs are. The department publishes that on its own
e-filing portal, and those pages are the vocabulary users actually ask in.

    python corpus-builder/scripts/fetch_guidance.py
    python corpus-builder/scripts/fetch_guidance.py --list

Output goes to ``data/raw/guidance/*.txt``, one file per page, each with a
header recording the source URL and retrieval date.

IMPORTANT: this is departmental *guidance*, not statute. The pages carry
their own disclaimer - "only to give an overview and general guidance and is
not exhaustive" - and it is preserved in every file. When indexed, guidance
is labelled separately from the Acts so an answer never presents it as law.

Only official government sources are fetched. Commercial tax sites are
copyrighted, unversioned, and cannot be cited to a section, which would
defeat the point of a grounded system.
"""

from __future__ import annotations

import argparse
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://www.incometax.gov.in"

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

#: Be a good citizen: one request at a time, with a pause between.
DELAY_SECONDS = 1.5

#: Curated seeds. The portal's how-to-file pages are video walkthroughs
#: with no usable text, so only the return-applicable pages are listed.
#: Curated seeds. Each covers one taxpayer category's returns, deductions
#: and slabs - the three things the Acts do not spell out.
PAGES: dict[str, str] = {
    # Returns, deductions and slabs per taxpayer category. Verified against
    # the portal's own help index; paths without a category prefix 404.
    # The -0/-1 suffixes are different assessment years; each page states
    # which AY it covers in its own heading.
    "individual-ay1": "/iec/foportal/help/individual/return-applicable",
    "individual-ay2": "/iec/foportal/help/individual/return-applicable-0",
    "individual-ay3": "/iec/foportal/help/individual/return-applicable-1",
    "individual-business": "/iec/foportal/help/individual-business-profession",
    "company-ay1": "/iec/foportal/help/company/return-applicable",
    "company-ay2": "/iec/foportal/help/company/return-applicable-0",
    "dos-and-donts": "/iec/foportal/help/all-topics/dosndont",
}

#: Leftovers that survive tag stripping and carry no meaning.
NOISE = re.compile(r"[{};]|cls-\d|fill:|prefix__|^\s*\d+\s*$")

#: Navigation and footer text repeated on every page.
BOILERPLATE = (
    "e-Filing of Income Tax Return or Forms",
    "Queries related to PAN & TAN",
    "Queries related to AIS, TIS, SFT",
    "Central & State Government Department",
    "Ask a Question",
    "Toggle navigation",
)


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})

    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="ignore")


def to_text(html: str) -> str:
    """Strip a portal page down to its readable content.

    BeautifulSoup with the stdlib parser - no lxml, which would add ~10 MB
    for no benefit at this scale.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()

    lines: list[str] = []
    seen: set[str] = set()

    for raw in soup.get_text("\n").split("\n"):
        line = " ".join(raw.split())

        if len(line) < 25 or NOISE.search(line):
            continue

        if any(line.startswith(b) for b in BOILERPLATE):
            continue

        # The portal repeats headings in menus and breadcrumbs.
        if line in seen:
            continue

        seen.add(line)
        lines.append(line)

    return "\n".join(lines)


def save(name: str, url: str, body: str, out_dir: Path) -> Path:
    """Write one page with a provenance header.

    The header is part of the file, not a side-car, so the source can never
    be separated from the text by a later processing step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.txt"

    header = (
        f"SOURCE: {url}\n"
        f"RETRIEVED: {datetime.now(timezone.utc).astimezone().isoformat()}\n"
        f"PUBLISHER: Income Tax Department, Government of India\n"
        f"TYPE: departmental guidance, not statute\n"
        f"{'-' * 72}\n\n"
    )

    path.write_text(header + body + "\n")

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/guidance"))
    parser.add_argument("--list", action="store_true", help="show the seeds and exit")
    args = parser.parse_args()

    if args.list:
        for name, path in PAGES.items():
            print(f"  {name:<24} {BASE}{path}")
        return

    for index, (name, path) in enumerate(PAGES.items()):
        url = f"{BASE}{path}"

        try:
            body = to_text(fetch(url))
        except Exception as exc:
            print(f"  {name:<24} FAILED: {exc}")
            continue

        if len(body) < 500:
            print(f"  {name:<24} thin ({len(body)} chars) - probably JS-rendered")
            continue

        saved = save(name, url, body, args.out)
        print(f"  {name:<24} {len(body):>6} chars -> {saved}")

        if index < len(PAGES) - 1:
            time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    main()
