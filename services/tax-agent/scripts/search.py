"""Query a Qdrant collection built by index.py.

Local batch script. The logic here becomes the /search endpoint in step 4.

    python services/tax-agent/scripts/search.py "deduction for life insurance premium"
    python services/tax-agent/scripts/search.py "..." --compare
    python services/tax-agent/scripts/search.py "..." --tax-year 2027
"""

from __future__ import annotations

import argparse
import logging
import os
import textwrap

from openai import OpenAI
from qdrant_client import QdrantClient, models

logging.disable(logging.WARNING)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def embedding_client() -> OpenAI:
    """Build the embedding client from the same env vars the service uses,
    so an index can never be built with a different model than it is queried
    with. OPENAI_API_KEY is accepted as a fallback.
    """
    key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not key:
        raise SystemExit("set EMBEDDING_API_KEY (or OPENAI_API_KEY)")

    return OpenAI(api_key=key, base_url=os.getenv("EMBEDDING_BASE_URL") or None)


def connect(target: str) -> QdrantClient:
    if target.startswith("http"):
        return QdrantClient(url=target)
    return QdrantClient(path=target)


def embed(text: str) -> list[float]:
    """Embed the query with the same model that built the index.

    A mismatch here is the quiet failure mode: different models produce
    vectors of the same shape but a meaningless geometry.
    """
    client = embedding_client()
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def build_filter(act: str | None, tax_year: int | None) -> models.Filter | None:
    must: list[models.Condition] = []

    if act:
        must.append(
            models.FieldCondition(key="act", match=models.MatchValue(value=act))
        )

    if tax_year is not None:
        # An act governs a year when it started on or before it and either
        # has not been repealed or was repealed after it.
        must.append(
            models.Filter(
                should=[
                    models.IsNullCondition(
                        is_null=models.PayloadField(key="tax_year_from")
                    ),
                    models.FieldCondition(
                        key="tax_year_from", range=models.Range(lte=tax_year)
                    ),
                ]
            )
        )
        must.append(
            models.Filter(
                should=[
                    models.IsNullCondition(
                        is_null=models.PayloadField(key="tax_year_to")
                    ),
                    models.FieldCondition(
                        key="tax_year_to", range=models.Range(gte=tax_year)
                    ),
                ]
            )
        )

    return models.Filter(must=must) if must else None


def citation(payload: dict) -> str:
    if payload.get("strategy") == "sections":
        number = payload.get("section_number")
        title = payload.get("section_title", "")
        pages = payload.get("page_start")
        part = payload.get("part", 0)
        total = payload.get("part_count", 1)
        suffix = f" (part {part + 1}/{total})" if total > 1 else ""
        return f"{payload.get('act')} s.{number} — {title} [p{pages}]{suffix}"
    return (
        f"{payload.get('act')} chunk {payload.get('chunk_index')} "
        f"[p{payload.get('page')}] — NO SECTION"
    )


def run(qdrant: QdrantClient, collection: str, vector, args) -> None:
    hits = qdrant.query_points(
        collection_name=collection,
        query=vector,
        limit=args.top_k,
        query_filter=build_filter(args.act, args.tax_year),
        with_payload=True,
    ).points

    print(f"\n=== {collection} — {len(hits)} hits ===")

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
        "--compare", action="store_true", help="query naive and sections"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--act")
    parser.add_argument("--tax-year", type=int)
    args = parser.parse_args()

    qdrant = connect(args.qdrant)
    existing = {c.name for c in qdrant.get_collections().collections}

    targets = ["ita_naive", "ita_sections"] if args.compare else [args.collection]
    targets = [t for t in targets if t in existing]

    if not targets:
        parser.error(f"no such collection. available: {sorted(existing) or 'none'}")

    print(f"query: {args.query!r}")
    if args.act or args.tax_year:
        print(f"filter: act={args.act} tax_year={args.tax_year}")

    vector = embed(args.query)

    for collection in targets:
        run(qdrant, collection, vector, args)


if __name__ == "__main__":
    main()
