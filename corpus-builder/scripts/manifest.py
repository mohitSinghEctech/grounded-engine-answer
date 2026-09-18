"""Record where every source document came from, and detect when it changes.

The corpus is only trustworthy if its provenance is. This writes
``data/manifest.json``: one entry per file with its source URL, a SHA-256 of
the bytes, and when it was retrieved.

    python corpus-builder/scripts/manifest.py --update
    python corpus-builder/scripts/manifest.py --check

``--check`` re-downloads each known URL and compares hashes. A changed hash
means the government republished the document and the index is now stale.
Exit code 1 says "a human should look".

Commit ``manifest.json``; the PDFs themselves are large and rebuildable
from it, so they stay out of git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pypdf

logging.disable(logging.WARNING)

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

#: Known source for each file. Government sites block default Python user
#: agents, and incometaxindia.gov.in blocks non-Indian IPs outright, so some
#: files can only be fetched by hand - recorded here as None.
SOURCES: dict[str, str | None] = {
    "finance-bill-2026.pdf": "https://www.indiabudget.gov.in/doc/Finance_Bill.pdf",
    "finance-bill-2026-memorandum.pdf": "https://www.indiabudget.gov.in/doc/memo.pdf",
    "budget-highlights-2026.pdf": "https://www.indiabudget.gov.in/doc/bh1.pdf",
    # Downloaded by hand from incometaxindia.gov.in, which refuses automated
    # and non-Indian requests.
    "ita-2025.pdf": None,
    "ita-1961.pdf": None,
}


def relative_path(path: Path) -> str:
    """Path as recorded in the manifest, e.g. "raw/guidance/company-ay1.txt"."""
    parts = path.parts

    return "/".join(parts[parts.index("raw") :]) if "raw" in parts else path.name


def digest(path: Path) -> str:
    """SHA-256 of a file's bytes.

    This is what turns "the corpus changed" from a guess into a fact.
    """
    sha = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)

    return sha.hexdigest()


def describe(path: Path) -> dict:
    """One manifest entry: identity, provenance, and a usability check."""
    entry = {
        "path": relative_path(path),
        "source_url": SOURCES.get(path.name)
        or (guidance_source(path) if path.suffix.lower() == ".txt" else None),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "retrieved_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }

    if path.suffix.lower() == ".txt":
        entry["has_text_layer"] = True
        entry["kind"] = "guidance"
        return entry

    entry["kind"] = "statute"

    try:
        reader = pypdf.PdfReader(path)
        middle = reader.pages[len(reader.pages) // 2].extract_text() or ""

        entry["pages"] = len(reader.pages)
        # A scanned PDF extracts almost nothing, and no amount of parsing
        # rescues it. Recording this makes the failure obvious later.
        entry["has_text_layer"] = len(middle.strip()) > 300
    except Exception as exc:
        entry["error"] = str(exc)

    return entry


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})

    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def guidance_source(path: Path) -> str | None:
    """Guidance files record their own source URL in a header line."""
    for line in path.read_text(errors="ignore").split("\n")[:5]:
        if line.startswith("SOURCE: "):
            return line[len("SOURCE: ") :].strip()

    return None


def update(raw_dir: Path, manifest_path: Path) -> None:
    """Rewrite the manifest from whatever is currently in data/raw."""
    files = sorted(
        p
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "files": [describe(p) for p in files],
    }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {manifest_path} - {len(files)} files")

    for entry in manifest["files"]:
        source = "auto" if entry["source_url"] else "manual"
        text = "text" if entry.get("has_text_layer") else "NO TEXT"

        print(
            f"  {Path(entry['path']).name:<34} "
            f"{entry.get('pages', '?'):>4}pp  {text:<7} {source:<6} "
            f"{entry['sha256'][:16]}"
        )


def check(raw_dir: Path, manifest_path: Path) -> int:
    """Re-download every known URL and compare hashes.

    Returns 1 if anything moved, so this can be a scheduled job whose exit
    code means "the law changed, review it".
    """
    if not manifest_path.exists():
        print("no manifest yet - run --update first")
        return 1

    manifest = json.loads(manifest_path.read_text())
    changed = 0
    skipped = 0

    for entry in manifest["files"]:
        name = Path(entry["path"]).name
        url = entry.get("source_url")

        if not url:
            print(f"  {name:<34} skipped (no source URL - fetched by hand)")
            skipped += 1
            continue

        try:
            remote = hashlib.sha256(download(url)).hexdigest()
        except Exception as exc:
            print(f"  {name:<34} FETCH FAILED: {exc}")
            changed += 1
            continue

        if remote == entry["sha256"]:
            print(f"  {name:<34} unchanged")
        else:
            print(f"  {name:<34} CHANGED  {entry['sha256'][:12]} -> {remote[:12]}")
            changed += 1

    print(f"\n{changed} changed, {skipped} unverifiable")

    return 1 if changed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--update", action="store_true", help="rewrite the manifest")
    parser.add_argument("--check", action="store_true", help="re-fetch and compare")
    args = parser.parse_args()

    if args.check:
        sys.exit(check(args.raw, args.manifest))

    update(args.raw, args.manifest)


if __name__ == "__main__":
    main()
