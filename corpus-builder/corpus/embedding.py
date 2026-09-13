"""Stage 4: text to vectors.

The model and its dimensions come from the same environment variables the
service reads, so an index can never be built with one model and queried
with another. That mismatch does not raise - it returns plausible nonsense -
so the only defence is to make both sides read one source.
"""

from __future__ import annotations

import os

from openai import OpenAI

#: Must match the service's EMBEDDING_MODEL.
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

#: Must match the collection's configured vector size.
DIMENSIONS = int(os.getenv("EMBEDDING_DIMS", "1536"))

#: The embeddings endpoint accepts batches; 128 keeps requests well inside
#: the payload limit while cutting round trips by two orders of magnitude.
BATCH = 128


def client() -> OpenAI:
    """Build the embedding client from the environment.

    Accepts OPENAI_API_KEY as a fallback so the scripts work with whichever
    variable happens to be set, while the service uses the explicit one.
    """
    key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not key:
        raise SystemExit("set EMBEDDING_API_KEY (or OPENAI_API_KEY)")

    return OpenAI(api_key=key, base_url=os.getenv("EMBEDDING_BASE_URL") or None)


def embed_all(texts: list[str], progress: bool = True) -> list[list[float]]:
    """Embed every text, in batches."""
    openai = client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        response = openai.embeddings.create(model=MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)

        if progress:
            print(f"  embedded {min(start + BATCH, len(texts)):>5} / {len(texts)}")

    return vectors


def embed_one(text: str) -> list[float]:
    """Embed a single query with the same model that built the index."""
    return client().embeddings.create(model=MODEL, input=[text]).data[0].embedding
