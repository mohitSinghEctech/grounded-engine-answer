"""Retrieval: pluggable embedders and vector stores behind one protocol.

    from app.retrieval import GroundedRetriever, embedders, stores

    embedder = embedders.create(settings)
    store = stores.create(settings)
    retriever = GroundedRetriever(embedder, store, settings)

To add a provider, add one entry to `embedders.EMBEDDERS` or
`stores.STORES`. Nothing else in the service changes.
"""

from app.retrieval.base import Embedder, Hit, VectorQuery, VectorStore
from app.retrieval.retriever import GroundedRetriever
from app.retrieval.sections import named_sections

__all__ = [
    "Embedder",
    "GroundedRetriever",
    "Hit",
    "VectorQuery",
    "VectorStore",
    "named_sections",
]
