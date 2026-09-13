"""Query a collection built by index.py.

    python corpus-builder/scripts/search.py "deduction for life insurance premium" \
        --qdrant http://localhost:6333 --collection ita_sections --tax-year 2027

Free of the model, instant, and the tool you reach for constantly: it
answers "was the right section even retrieved?" before you ask why an answer
was wrong. The same logic lives behind the service's /search endpoint.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from corpus import embedding, vectorstore  # noqa: E402


def citation(payload: dict) -> str:
    """One line naming where a hit came from, or saying that it cannot."""
    if payload.get("strategy") == "guidance":
        return (
            f"{payload.get('act')} {payload.get('section_number')} - "
            f"{(payload.get('section_title') or '')[:60]}"
        )

    if payload.get("strategy") != "sections":
        return (
            f"{payload.get('act')} chunk {payload.get('chunk_index')} "
            f"[p{payload.get('page')}] - NO SECTION"
        )

    total = payload.get("part_count", 1)
    part = f" (part {payload.get('part', 0) + 1}/{total})" if total > 1 else ""

    return (
        f"{payload.get('act')} s.{payload.get('section_number')} - "
        f"{payload.get('section_title')} [p{payload.get('page_start')}]{part}"
    )


def show(qdrant, collection: str, vector, args) -> None:
    hits = vectorstore.search(
        qdrant,
        collection,
        vector,
        limit=args.top_k,
        act=args.act,
        tax_year=args.tax_year,
    )

    print(f"\n=== {collection} - {len(hits)} hits ===")

    if not hits:
        print("  (nothing matched the filter)")
        return

    for rank, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        snippet = " ".join((payload.get("text") or "").split())

        print(f"\n{rank}. score {hit.score:.4f}  {citation(payload)}")
        print(
            textwrap.fill(
                snippet[:340], width=88, initial_indent="   ", subsequent_indent="   "
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--qdrant", default="data/index")
    parser.add_argument("--collection", default="ita_sections")
    parser.add_argument(
        "--compare", action="store_true", help="query the naive and sections indexes"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--act", help="ITA-1961 or ITA-2025")
    parser.add_argument("--tax-year", type=int, help="starting year of the FY")
    args = parser.parse_args()

    qdrant = vectorstore.connect(args.qdrant)
    existing = {c.name for c in qdrant.get_collections().collections}

    targets = ["ita_naive", "ita_sections"] if args.compare else [args.collection]
    targets = [t for t in targets if t in existing]

    if not targets:
        parser.error(f"no such collection. available: {sorted(existing) or 'none'}")

    print(f"query: {args.query!r}")

    if args.act or args.tax_year:
        print(f"filter: act={args.act} tax_year={args.tax_year}")

    vector = embedding.embed_one(args.query)

    for collection in targets:
        show(qdrant, collection, vector, args)


if __name__ == "__main__":
    main()
