"""Stage 5: vectors into Qdrant, and search over them.

Two connection modes, same code either way:

    http://localhost:6333   server - dev, because it has a dashboard
    data/index              embedded - production, index baked into the image

One collection holds both Acts. Which one answers a question is decided by
the tax-year filter, not by which collection is queried.
"""

from __future__ import annotations

import uuid
from datetime import date

from qdrant_client import QdrantClient, models

from corpus.embedding import DIMENSIONS, MODEL
from corpus.models import Chunk

#: Fields that must be indexed, or filtering degrades to a full scan.
KEYWORD_FIELDS = ("act", "strategy", "section_number")
INTEGER_FIELDS = ("tax_year_from", "tax_year_to")

#: Upsert batch size. Large enough to be fast, small enough that a failure
#: does not lose much work.
UPLOAD_BATCH = 256


def connect(target: str) -> QdrantClient:
    """Open a server URL or an embedded on-disk index."""
    if target.startswith("http"):
        return QdrantClient(url=target)

    return QdrantClient(path=target)


def point_id(chunk: Chunk) -> str:
    """A stable id derived from act, strategy, section and part.

    Deterministic so that re-indexing an act replaces its points instead of
    duplicating them.
    """
    payload = chunk.payload
    unit = payload.get("section_number", payload.get("chunk_index"))

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{payload.get('act')}/{payload.get('strategy')}/"
            f"{unit}/{payload.get('part', 0)}",
        )
    )


def ensure_collection(qdrant: QdrantClient, name: str) -> None:
    """Create the collection, or check that an existing one still fits.

    Refusing a dimension mismatch here is the difference between a loud
    failure and an index that silently returns meaningless neighbours.
    """
    if not qdrant.collection_exists(name):
        qdrant.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=DIMENSIONS, distance=models.Distance.COSINE
            ),
        )

        for field in KEYWORD_FIELDS:
            qdrant.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        for field in INTEGER_FIELDS:
            qdrant.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        return

    existing = qdrant.get_collection(name).config.params.vectors.size

    if existing != DIMENSIONS:
        raise SystemExit(
            f"{name!r} holds {existing}-dimension vectors but this run produces "
            f"{DIMENSIONS}. Use a different collection."
        )


def replace_act(qdrant: QdrantClient, name: str, act: str) -> None:
    """Delete one act's points, leaving the rest of the collection intact."""
    qdrant.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="act", match=models.MatchValue(value=act))
                ]
            )
        ),
    )


def upload(
    qdrant: QdrantClient, name: str, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    """Write chunks and their vectors, replacing only this act's points."""
    act = chunks[0].payload.get("act") if chunks else None

    ensure_collection(qdrant, name)

    if act:
        replace_act(qdrant, name, act)

    stamp = date.today().isoformat()

    points = [
        models.PointStruct(
            id=point_id(chunk),
            vector=vector,
            payload={
                **chunk.payload,
                "embedding_model": MODEL,
                "corpus_date": stamp,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    for start in range(0, len(points), UPLOAD_BATCH):
        qdrant.upsert(collection_name=name, points=points[start : start + UPLOAD_BATCH])


def year_filter(act: str | None, tax_year: int | None) -> models.Filter | None:
    """Restrict a search to the act that governs ``tax_year``.

    A null bound means "open ended", so it has to match rather than be
    excluded - IsNullCondition alongside the range. Getting that backwards
    fails silently by returning fewer results, never by raising.
    """
    must: list[models.Condition] = []

    if act:
        must.append(
            models.FieldCondition(key="act", match=models.MatchValue(value=act))
        )

    if tax_year is not None:

        def bounded(field: str, condition: models.Range) -> models.Filter:
            return models.Filter(
                should=[
                    models.IsNullCondition(is_null=models.PayloadField(key=field)),
                    models.FieldCondition(key=field, range=condition),
                ]
            )

        must.append(bounded("tax_year_from", models.Range(lte=tax_year)))
        must.append(bounded("tax_year_to", models.Range(gte=tax_year)))

    return models.Filter(must=must) if must else None


def search(
    qdrant: QdrantClient,
    name: str,
    vector: list[float],
    limit: int,
    act: str | None = None,
    tax_year: int | None = None,
):
    """Nearest neighbours, narrowed by act and tax year before ranking."""
    return qdrant.query_points(
        collection_name=name,
        query=vector,
        limit=limit,
        query_filter=year_filter(act, tax_year),
        with_payload=True,
    ).points
