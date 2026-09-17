"""The two sockets retrieval plugs into: something that embeds, something
that stores.

The original `QdrantRetriever` did three jobs in one class - turn text into
numbers, search a database, and decide what to keep. That works until you
want a different database, at which point all three have to be rewritten
together. Splitting them means a new store is one small class and a
dictionary entry.

A `Protocol` is a list of methods with no code behind it. Anything with
matching methods satisfies it, with no inheritance and no registration, so
`GroundedRetriever` never learns which store it is talking to.

Where the boundary sits is the whole design:

  * `VectorQuery` describes *what* to look for - a vector, a limit, an
    optional tax year, optional section numbers. It says nothing about how
    to express that, because every store has its own filter language.
  * Each store translates the query into its own dialect. Qdrant builds a
    `models.Filter`; a SQL store would build a WHERE clause.
  * A store returns scores and raw payload dictionaries, not `Passage`
    objects. Payload *shape* is a fact about this corpus, not about the
    database, so exactly one place knows it - `GroundedRetriever`. A new
    store never has to learn what "tax_year_from" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class VectorQuery:
    """One search, described without reference to any particular database."""

    vector: list[float]
    limit: int

    #: Keep only provisions that govern this year. None means every year.
    tax_year: int | None = None

    #: Restrict to these section numbers. Used for the exact-lookup path,
    #: where the user named a section and similarity is not the question.
    section_numbers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hit:
    """One result: how well it matched, and whatever the store had stored."""

    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class Embedder(Protocol):
    """Turns text into a vector.

    `dimensions` is declared rather than discovered because it has to be
    checked against the index before the first request. Querying a
    1536-dimension index with 768-dimension vectors raises nothing
    anywhere useful - it returns plausible nonsense.
    """

    @property
    def dimensions(self) -> int: ...

    @property
    def name(self) -> str: ...

    async def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Holds embedded passages and finds the closest ones.

    To add a store - Chroma, pgvector, FAISS, Pinecone - implement these
    three methods and add one entry to `stores.STORES`. Nothing else in the
    service changes.
    """

    @property
    def name(self) -> str: ...

    async def search(self, query: VectorQuery) -> list[Hit]: ...

    async def verify(self, expected_dimensions: int) -> None:
        """Raise RuntimeError if the index cannot serve this embedder.

        Called once at startup. Failing here fails the container's health
        check, which is much easier to diagnose than wrong answers.
        """
        ...

    async def close(self) -> None: ...
