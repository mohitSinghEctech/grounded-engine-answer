"""Embedders, and the registry that picks one.

Same shape as `llm-gateway/app/vendors.py`: a dictionary from a name to a
factory, so adding a provider is adding an entry rather than editing the
service.

The index and the embedder must agree. Text embedded by one model and
searched with another returns confident nonsense rather than an error, so
changing `EMBEDDER_PROVIDER` or `EMBEDDING_MODEL` means rebuilding the
index - and `VectorStore.verify` refuses to start when the dimensions no
longer match.
"""

from __future__ import annotations

import logging
from typing import Callable

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.config import Settings
from app.errors import EmbeddingFailed
from app.retrieval.base import Embedder

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedder:
    """Any embedding endpoint that speaks the OpenAI API.

    Which is most of them: OpenAI itself, Together, a local vLLM or
    text-embeddings-inference server, or an Ollama instance. Point
    EMBEDDING_BASE_URL at it and set the model name; no code changes.
    """

    def __init__(self, client: AsyncOpenAI, model: str, dimensions: int):
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return self._model

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(
                model=self._model,
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

    async def close(self) -> None:
        await self._client.close()


def _openai(settings: Settings) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(
        client=AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            # Retries belong to the caller, which knows the request budget.
            max_retries=0,
        ),
        model=settings.embedding_model,
        dimensions=settings.embedding_dims,
    )


#: name -> factory. Add a provider here and it becomes selectable.
#:
#: A sentence-transformers or Hugging Face embedder would be the next entry:
#: wrap `SentenceTransformer(model).encode` in a class with `dimensions`,
#: `name` and an async `embed`, and run it in a thread so it does not block
#: the event loop. It needs no network at request time, which makes it the
#: cheapest way to run this whole service offline.
EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {
    "openai": _openai,
}


def create(settings: Settings) -> Embedder:
    """Build the configured embedder, failing loudly on an unknown name."""
    try:
        factory = EMBEDDERS[settings.embedder_provider]
    except KeyError:
        known = ", ".join(sorted(EMBEDDERS))
        raise ValueError(
            f"Unknown EMBEDDER_PROVIDER {settings.embedder_provider!r}. "
            f"Known embedders: {known}."
        ) from None

    embedder = factory(settings)

    logger.info(
        "Embedder resolved | provider=%s | model=%s | dims=%s",
        settings.embedder_provider,
        embedder.name,
        embedder.dimensions,
    )

    return embedder
