import logging

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from qdrant_client import AsyncQdrantClient, models

from app.config import Settings
from app.errors import EmbeddingFailed, RetrieverUnavailable
from app.services.base import Passage

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(
        self,
        qdrant: AsyncQdrantClient,
        embedder: AsyncOpenAI,
        settings: Settings,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.settings = settings

    async def verify(self) -> None:
        """Refuse to start if the index disagrees with our embedding config.

        Querying a 1536-dimension collection with 768-dimension vectors does
        not raise anywhere useful - it returns plausible-looking nonsense. The
        only safe place to catch it is before serving a single request.
        """
        collection = self.settings.qdrant_collection

        if not await self.qdrant.collection_exists(collection):
            raise RuntimeError(
                f"Qdrant collection {collection!r} does not exist. "
                f"Run scripts/index.py first."
            )

        info = await self.qdrant.get_collection(collection)
        actual = info.config.params.vectors.size

        if actual != self.settings.embedding_dims:
            raise RuntimeError(
                f"Collection {collection!r} has {actual}-dimension vectors but "
                f"EMBEDDING_DIMS is {self.settings.embedding_dims}. The index was "
                f"built with a different model."
            )

        logger.info(
            "Retriever ready | collection=%s | points=%s | dims=%s",
            collection,
            info.points_count,
            actual,
        )

    async def _embed(self, text: str) -> list[float]:
        try:
            response = await self.embedder.embeddings.create(
                model=self.settings.embedding_model,
                input=[text],
            )
        except APITimeoutError as exc:
            logger.error("Embedding failed | error=EMBEDDING_FAILED | reason=timeout")
            raise EmbeddingFailed("The embedding request timed out.") from exc
        except (APIConnectionError, APIStatusError) as exc:
            logger.error(
                "Embedding failed | error=EMBEDDING_FAILED | reason=%s",
                type(exc).__name__,
            )
            raise EmbeddingFailed() from exc

        return response.data[0].embedding

    def _year_filter(self, tax_year: int | None) -> models.Filter | None:
        if tax_year is None:
            return None

        # An act governs a year if it commenced on or before it, and either
        # has not been repealed or was repealed after it. A null bound means
        # "open ended", so it must match rather than be excluded.
        def bounded(field: str, condition: models.Range) -> models.Filter:
            return models.Filter(
                should=[
                    models.IsNullCondition(is_null=models.PayloadField(key=field)),
                    models.FieldCondition(key=field, range=condition),
                ]
            )

        return models.Filter(
            must=[
                bounded("tax_year_from", models.Range(lte=tax_year)),
                bounded("tax_year_to", models.Range(gte=tax_year)),
            ]
        )

    @staticmethod
    def _to_passage(point) -> Passage:
        payload = point.payload or {}

        return Passage(
            text=payload.get("text", ""),
            score=point.score,
            act=payload.get("act", "unknown"),
            section_number=payload.get("section_number"),
            section_title=payload.get("section_title"),
            page_start=payload.get("page_start") or payload.get("page"),
            corpus_date=payload.get("corpus_date", "unknown"),
        )

    async def search(
        self,
        question: str,
        top_k: int,
        tax_year: int | None = None,
    ) -> list[Passage]:
        vector = await self._embed(question)

        try:
            response = await self.qdrant.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                limit=top_k,
                query_filter=self._year_filter(tax_year),
                with_payload=True,
            )
        except (httpx.HTTPError, OSError, ValueError) as exc:
            logger.error(
                "Retrieval failed | error=RETRIEVER_UNAVAILABLE | reason=%s",
                type(exc).__name__,
            )
            raise RetrieverUnavailable() from exc

        floor = self.settings.retrieval_min_score
        passages = [
            self._to_passage(point) for point in response.points if point.score >= floor
        ]

        logger.info(
            "Retrieval completed | hits=%s | kept=%s | min_score=%.2f | tax_year=%s",
            len(response.points),
            len(passages),
            floor,
            tax_year,
        )

        return passages
