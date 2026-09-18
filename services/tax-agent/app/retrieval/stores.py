"""Vector stores, and the registry that picks one.

`QdrantStore` is what production uses. `InMemoryStore` exists to prove the
socket is real: it shares no code with Qdrant, needs no server, and the
retriever cannot tell them apart. If the interface were leaking Qdrant
concepts, writing the second one would have been impossible - which is
exactly why it is worth having.

Each store owns the translation from `VectorQuery` into its own filter
language, and returns raw payloads. Neither knows what a tax year means.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

import httpx
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.errors import RetrieverUnavailable
from app.retrieval.base import Hit, VectorQuery, VectorStore

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant, filtered on the payload the corpus builder writes."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        label: str = "qdrant",
    ):
        self._client = client
        self._collection = collection
        self._label = label

    @property
    def name(self) -> str:
        return f"{self._label}:{self._collection}"

    def _filter(self, query: VectorQuery) -> models.Filter | None:
        conditions: list[models.Condition] = []

        if query.tax_year is not None:
            # An Act governs a year if it commenced on or before it and was
            # either never repealed or repealed after it. A null bound means
            # "open ended", so it must match rather than be excluded.
            def bounded(field: str, rng: models.Range) -> models.Filter:
                return models.Filter(
                    should=[
                        models.IsNullCondition(is_null=models.PayloadField(key=field)),
                        models.FieldCondition(key=field, range=rng),
                    ]
                )

            conditions.append(
                bounded("tax_year_from", models.Range(lte=query.tax_year))
            )
            conditions.append(bounded("tax_year_to", models.Range(gte=query.tax_year)))

        if query.section_numbers:
            conditions.append(
                models.FieldCondition(
                    key="section_number",
                    match=models.MatchAny(any=list(query.section_numbers)),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    async def search(self, query: VectorQuery) -> list[Hit]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query.vector,
                limit=query.limit,
                query_filter=self._filter(query),
                with_payload=True,
            )
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.error(
                "Retrieval failed | error=RETRIEVER_UNAVAILABLE | reason=%s",
                type(exc).__name__,
            )
            raise RetrieverUnavailable() from exc

        return [Hit(score=p.score, payload=p.payload or {}) for p in response.points]

    async def verify(self, expected_dimensions: int) -> None:
        if not await self._client.collection_exists(self._collection):
            raise RuntimeError(
                f"Qdrant collection {self._collection!r} does not exist. "
                f"Run scripts/index.py first."
            )

        info = await self._client.get_collection(self._collection)
        actual = info.config.params.vectors.size

        if actual != expected_dimensions:
            raise RuntimeError(
                f"Collection {self._collection!r} holds {actual}-dimension "
                f"vectors but the embedder produces {expected_dimensions}. "
                f"The index was built with a different model; rebuild it."
            )

        logger.info(
            "Vector store ready | store=%s | points=%s | dims=%s",
            self.name,
            info.points_count,
            actual,
        )

    async def close(self) -> None:
        await self._client.close()


class InMemoryStore:
    """A list of vectors and a cosine loop. No server, no network.

    Used by the tests, and as the proof that `VectorStore` is a real
    boundary rather than Qdrant with extra steps. Linear scan, so it suits
    tens of passages, not thousands.
    """

    def __init__(self, entries: list[tuple[list[float], dict]] | None = None):
        self._entries = entries or []

    @property
    def name(self) -> str:
        return f"memory:{len(self._entries)}"

    def add(self, vector: list[float], payload: dict) -> None:
        self._entries.append((vector, payload))

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))

        if not left_norm or not right_norm:
            return 0.0

        return dot / (left_norm * right_norm)

    @staticmethod
    def _governs(payload: dict, tax_year: int) -> bool:
        start, end = payload.get("tax_year_from"), payload.get("tax_year_to")

        # Null means open ended at that end, matching QdrantStore.
        return (start is None or start <= tax_year) and (end is None or end >= tax_year)

    async def search(self, query: VectorQuery) -> list[Hit]:
        hits = [
            Hit(score=self._cosine(query.vector, vector), payload=payload)
            for vector, payload in self._entries
            if (query.tax_year is None or self._governs(payload, query.tax_year))
            and (
                not query.section_numbers
                or payload.get("section_number") in query.section_numbers
            )
        ]

        hits.sort(key=lambda hit: hit.score, reverse=True)

        return hits[: query.limit]

    async def verify(self, expected_dimensions: int) -> None:
        for vector, payload in self._entries:
            if len(vector) != expected_dimensions:
                raise RuntimeError(
                    f"In-memory entry {payload.get('section_number')!r} holds a "
                    f"{len(vector)}-dimension vector but the embedder produces "
                    f"{expected_dimensions}."
                )

        logger.info(
            "Vector store ready | store=%s | points=%s | dims=%s",
            self.name,
            len(self._entries),
            expected_dimensions,
        )

    async def close(self) -> None:
        return None


def _qdrant(settings: Settings) -> QdrantStore:
    """Qdrant as a server, reached over HTTP."""
    if not settings.qdrant_url:
        raise ValueError(
            "VECTOR_STORE_PROVIDER=qdrant needs QDRANT_URL. For an on-disk "
            "index with no server, use VECTOR_STORE_PROVIDER=qdrant_embedded."
        )

    return QdrantStore(
        client=AsyncQdrantClient(url=settings.qdrant_url),
        collection=settings.qdrant_collection,
    )


def _qdrant_embedded(settings: Settings) -> QdrantStore:
    """Qdrant as a directory, opened in-process.

    The same class - only the client differs - because a store's job is
    unchanged by where it keeps its bytes. This is the deployable shape: no
    second container, no service discovery, nothing to wait for at startup.

    Two consequences worth knowing. The directory is opened with an
    exclusive lock, so exactly one process may hold it; that suits one
    container and rules out scaling by running several. And the index is
    read-only in practice - it is built by the corpus builder and shipped,
    not written to at request time.
    """
    path = Path(settings.qdrant_path)

    if not path.exists():
        raise RuntimeError(
            f"No index at {path}. Build one with `make index-standalone`, "
            f"or set QDRANT_PATH to where it lives."
        )

    return QdrantStore(
        client=AsyncQdrantClient(path=str(path)),
        collection=settings.qdrant_collection,
        label="embedded",
    )


def _memory(settings: Settings) -> InMemoryStore:
    # Starts empty, so it is only useful where something seeds it - tests,
    # or a future script that loads a small corpus for a demo.
    return InMemoryStore()


#: name -> factory. A Chroma, pgvector or FAISS store is one class
#: implementing `search`, `verify` and `close`, plus one entry here.
STORES: dict[str, Callable[[Settings], VectorStore]] = {
    "qdrant": _qdrant,
    "qdrant_embedded": _qdrant_embedded,
    "memory": _memory,
}


def create(settings: Settings) -> VectorStore:
    """Build the configured store, failing loudly on an unknown name."""
    try:
        factory = STORES[settings.vector_store_provider]
    except KeyError:
        known = ", ".join(sorted(STORES))
        raise ValueError(
            f"Unknown VECTOR_STORE_PROVIDER {settings.vector_store_provider!r}. "
            f"Known stores: {known}."
        ) from None

    store = factory(settings)

    logger.info(
        "Vector store resolved | provider=%s | store=%s",
        settings.vector_store_provider,
        store.name,
    )

    return store
