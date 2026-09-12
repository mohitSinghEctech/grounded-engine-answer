"""Build a Qdrant index from the Act, two ways.

Local batch script. Never imported by the service.

    --strategy naive     chunk the raw PDF text by character count.
                         No section metadata, so answers cannot cite.
    --strategy sections  chunk on section boundaries from corpus.db.
                         Every chunk carries act, section and page.

    python services/tax-agent/scripts/index.py --strategy naive \
        --pdf data/raw/ita-2025.pdf --qdrant data/index
    python services/tax-agent/scripts/index.py --strategy sections \
        --db data/corpus.db --qdrant data/index
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient, models

logging.disable(logging.WARNING)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIMS = int(os.getenv("EMBEDDING_DIMS", "1536"))
BATCH = 128

# The 1961 Act was repealed on 1 April 2026; the 2025 Act commenced the
# same day. Tax years are stored as the starting calendar year.
ACT_YEARS = {
    "ITA-1961": (None, 2025),
    "ITA-2025": (2026, None),
}


@dataclass
class Chunk:
    text: str
    payload: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Strategy A: naive character chunking
# --------------------------------------------------------------------------

_SPLITTERS = ["\n\n", "\n", ". ", " "]


def split_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Split on the largest natural separator that fits, then slide a window.

    The same idea as LangChain's RecursiveCharacterTextSplitter, without the
    dependency. It knows nothing about sections, which is the point.
    """
    if len(text) <= size:
        return [text] if text.strip() else []

    for separator in _SPLITTERS:
        parts = text.split(separator)

        if len(parts) == 1:
            continue

        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else current + separator + part

            if len(candidate) <= size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(part) > size:
                chunks.extend(split_recursive(part, size, overlap))
                current = ""
            else:
                current = part

        if current:
            chunks.append(current)

        if overlap and len(chunks) > 1:
            merged = [chunks[0]]
            for chunk in chunks[1:]:
                merged.append(merged[-1][-overlap:] + chunk)
            chunks = merged

        return [c for c in chunks if c.strip()]

    return [text[i : i + size] for i in range(0, len(text), size - overlap)]


def chunks_naive(pdf: Path, act: str, size: int, overlap: int) -> list[Chunk]:
    import pypdf

    reader = pypdf.PdfReader(pdf)
    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    text = "\n".join(t for _, t in pages)

    starts, cursor = [], 0
    for number, page_text in pages:
        starts.append((cursor, number))
        cursor += len(page_text) + 1

    def page_of(offset: int) -> int:
        found = 1
        for start, number in starts:
            if start <= offset:
                found = number
            else:
                break
        return found

    out: list[Chunk] = []
    cursor = 0

    for index, body in enumerate(split_recursive(text, size, overlap)):
        position = text.find(body[:60], cursor)
        if position >= 0:
            cursor = position

        out.append(
            Chunk(
                text=body,
                payload={
                    "act": act,
                    "strategy": "naive",
                    "chunk_index": index,
                    "page": page_of(cursor),
                    "text": body,
                },
            )
        )

    return out


# --------------------------------------------------------------------------
# Strategy B: section-aware chunking
# --------------------------------------------------------------------------

# "(1) ", "(2) " at the start of a line: sub-section boundaries.
_SUBSECTION = re.compile(r"(?m)^\((\d{1,2})\)\s")

MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 2400


def split_on_subsections(text: str, limit: int) -> list[str]:
    """Split an oversized section at sub-section boundaries, never mid-provision."""
    if len(text) <= limit:
        return [text]

    marks = [m.start() for m in _SUBSECTION.finditer(text)]

    if not marks:
        return split_recursive(text, limit, 0)

    bounds = [0] + [m for m in marks if m > 0] + [len(text)]

    parts: list[str] = []
    current = ""

    for i in range(len(bounds) - 1):
        piece = text[bounds[i] : bounds[i + 1]]

        if len(current) + len(piece) <= limit:
            current += piece
        else:
            if current:
                parts.append(current)
            current = piece if len(piece) <= limit else ""
            if not current:
                parts.extend(split_recursive(piece, limit, 0))

    if current:
        parts.append(current)

    return [p for p in parts if p.strip()]


def chunks_sections(db: Path, act: str) -> list[Chunk]:
    year_from, year_to = ACT_YEARS.get(act, (None, None))

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            """
            SELECT section_number, section_title, text, page_start, page_end,
                   maps_to_1961
            FROM sections WHERE act = ?
            ORDER BY id
            """,
            (act,),
        ).fetchall()

    out: list[Chunk] = []

    for number, title, text, page_start, page_end, maps_to in rows:
        # Omitted sections ("443. 9[***]") carry no law. Indexing them adds
        # noise that can only ever produce a wrong citation.
        if len(text) < MIN_CHUNK_CHARS:
            continue

        parts = split_on_subsections(text, MAX_CHUNK_CHARS)

        for index, part in enumerate(parts):
            # Every chunk repeats its heading, so a passage retrieved from
            # the middle of a long section still says what it belongs to.
            body = f"{act} section {number}. {title}.\n\n{part.strip()}"

            out.append(
                Chunk(
                    text=body,
                    payload={
                        "act": act,
                        "strategy": "sections",
                        "section_number": number,
                        "section_title": title,
                        "part": index,
                        "part_count": len(parts),
                        "page_start": page_start,
                        "page_end": page_end,
                        "tax_year_from": year_from,
                        "tax_year_to": year_to,
                        "maps_to_1961": maps_to,
                        "text": body,
                    },
                )
            )

    return out


# --------------------------------------------------------------------------
# Embedding and upload
# --------------------------------------------------------------------------


def embedding_client() -> OpenAI:
    """Build the embedding client from the same env vars the service uses,
    so an index can never be built with a different model than it is queried
    with. OPENAI_API_KEY is accepted as a fallback.
    """
    key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not key:
        raise SystemExit("set EMBEDDING_API_KEY (or OPENAI_API_KEY)")

    return OpenAI(api_key=key, base_url=os.getenv("EMBEDDING_BASE_URL") or None)


def embed_all(client: OpenAI, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
        print(f"  embedded {min(start + BATCH, len(texts)):>5} / {len(texts)}")

    return vectors


def connect(target: str) -> QdrantClient:
    if target.startswith("http"):
        return QdrantClient(url=target)
    return QdrantClient(path=target)


def upload(qdrant: QdrantClient, collection: str, chunks: list[Chunk], vectors) -> None:
    if qdrant.collection_exists(collection):
        qdrant.delete_collection(collection)

    qdrant.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=EMBED_DIMS, distance=models.Distance.COSINE
        ),
    )

    # Without these the year filter still works - by scanning. With them
    # Qdrant narrows before the vector search.
    for field_name in ("act", "strategy", "section_number"):
        qdrant.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    for field_name in ("tax_year_from", "tax_year_to"):
        qdrant.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    stamp = date.today().isoformat()

    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                **chunk.payload,
                "embedding_model": EMBED_MODEL,
                "corpus_date": stamp,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    for start in range(0, len(points), 256):
        qdrant.upsert(collection_name=collection, points=points[start : start + 256])

    print(f"  uploaded {len(points)} points to '{collection}'")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("naive", "sections"), required=True)
    parser.add_argument("--act", default="ITA-2025")
    parser.add_argument("--pdf", type=Path, help="naive strategy input")
    parser.add_argument("--db", type=Path, help="sections strategy input")
    parser.add_argument("--qdrant", default="data/index", help="path or http URL")
    parser.add_argument("--collection")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument(
        "--dry-run", action="store_true", help="chunk only, no API calls"
    )
    args = parser.parse_args()

    if args.strategy == "naive":
        if not args.pdf:
            parser.error("--pdf is required for the naive strategy")
        chunks = chunks_naive(args.pdf, args.act, args.size, args.overlap)
    else:
        if not args.db:
            parser.error("--db is required for the sections strategy")
        chunks = chunks_sections(args.db, args.act)

    lengths = sorted(len(c.text) for c in chunks)

    print(f"strategy    {args.strategy}")
    print(f"chunks      {len(chunks)}")
    print(
        f"length      min {lengths[0]}  "
        f"median {lengths[len(lengths) // 2]}  max {lengths[-1]}"
    )

    if args.strategy == "sections":
        cited = len({c.payload["section_number"] for c in chunks})
        print(f"sections    {cited} distinct, citable")
    else:
        print("sections    0 - naive chunks cannot cite a section")

    if args.dry_run:
        print("\nsample chunk:\n")
        print(chunks[len(chunks) // 2].text[:600])
        return

    collection = args.collection or f"ita_{args.strategy}"

    print(f"\nembedding with {EMBED_MODEL}...")
    vectors = embed_all(embedding_client(), [c.text for c in chunks])

    print(f"uploading to {args.qdrant}...")
    upload(connect(args.qdrant), collection, chunks, vectors)


if __name__ == "__main__":
    main()
