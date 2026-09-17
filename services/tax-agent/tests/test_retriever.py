"""Retrieval policy, tested without Qdrant, without OpenAI, without Docker.

That this is possible at all is the point of the refactor. These tests
drive the real `GroundedRetriever` through the same `Embedder` and
`VectorStore` protocols production uses, with a fake embedder and the
in-memory store. If either protocol were leaking Qdrant concepts, none of
this would run.

The behaviours pinned here were all found by the eval, not by inspection:
one section flooding the window, and named sections never being retrieved.
"""

from types import SimpleNamespace

import pytest

from app.retrieval.base import VectorQuery
from app.retrieval.retriever import GroundedRetriever
from app.retrieval.stores import InMemoryStore


class StubEmbedder:
    """Maps text to a vector by counting a few marker words.

    Deterministic and dependency-free. It does not need to be a good
    embedder - the tests are about what the retriever does with the
    ordering, not about how the ordering is produced.
    """

    def __init__(self, dimensions: int = 3):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return "stub"

    async def embed(self, text: str) -> list[float]:
        lowered = text.lower()

        return [
            float(lowered.count("interest")),
            float(lowered.count("house")),
            float(lowered.count("insurance")),
        ]


def settings(**overrides):
    base = dict(
        retrieval_top_k=6,
        retrieval_min_score=0.30,
        retrieval_max_per_section=2,
        retrieval_over_fetch=4,
    )
    base.update(overrides)

    return SimpleNamespace(**base)


def payload(act, section, title, text, **extra):
    row = {
        "act": act,
        "section_number": section,
        "section_title": title,
        "text": text,
        "corpus_date": "2026-09-14",
    }
    row.update(extra)

    return row


@pytest.fixture
def store():
    """Four chunks of s.234A against one each of two other sections.

    Shaped after the real measurement: s.234A held 4 of 6 slots on one
    query, so the answer saw one provision four times instead of four
    provisions once.
    """
    memory = InMemoryStore()

    for part in range(4):
        memory.add(
            [1.0, 0.0, 0.0],
            payload("ITA-1961", "234A", "Interest for defaults", f"interest {part}"),
        )

    memory.add([0.9, 0.0, 0.0], payload("ITA-2025", "423", "Interest", "interest"))
    memory.add([0.8, 0.0, 0.0], payload("ITA-1961", "234B", "Interest", "interest"))

    return memory


@pytest.mark.anyio
async def test_no_one_section_may_flood_the_window(store):
    retriever = GroundedRetriever(StubEmbedder(), store, settings())

    passages = await retriever.search("interest on a late return", top_k=6)

    counts = {}
    for p in passages:
        counts[p.section_number] = counts.get(p.section_number, 0) + 1

    assert counts["234A"] == 2, "the cap should hold 234A to two chunks"
    assert {"423", "234B"} <= set(counts), "freed slots go to other sections"


@pytest.mark.anyio
async def test_raising_the_cap_lets_one_section_take_more(store):
    retriever = GroundedRetriever(
        StubEmbedder(), store, settings(retrieval_max_per_section=4)
    )

    passages = await retriever.search("interest on a late return", top_k=6)

    assert sum(1 for p in passages if p.section_number == "234A") == 4


@pytest.mark.anyio
async def test_a_named_section_is_retrieved_even_when_similarity_misses_it():
    """The SM-06 failure, reproduced in miniature.

    s.139 is worded nothing like the question, so similarity ranks its
    neighbours above it. Naming it has to be enough.
    """
    memory = InMemoryStore()
    memory.add([0.0, 0.0, 0.0], payload("ITA-1961", "139", "Return of income", "text"))

    for neighbour in ("139A", "139B", "139C"):
        memory.add(
            [1.0, 1.0, 1.0], payload("ITA-1961", neighbour, "Something else", "text")
        )

    retriever = GroundedRetriever(StubEmbedder(), memory, settings())

    passages = await retriever.search("What is section 139 about?", top_k=3)

    assert passages[0].section_number == "139", "the named section comes first"


@pytest.mark.anyio
async def test_the_score_floor_does_not_apply_to_a_named_section():
    """A section the user named is relevant because they named it."""
    memory = InMemoryStore()
    memory.add([0.0, 0.0, 0.0], payload("ITA-1961", "80C", "Life insurance", "text"))

    retriever = GroundedRetriever(
        StubEmbedder(), memory, settings(retrieval_min_score=0.99)
    )

    passages = await retriever.search("what does section 80C say?", top_k=6)

    assert [p.section_number for p in passages] == ["80C"]


@pytest.mark.anyio
async def test_the_year_filter_excludes_an_act_that_does_not_govern():
    memory = InMemoryStore()
    memory.add(
        [0.0, 0.0, 1.0],
        payload(
            "ITA-1961",
            "80C",
            "Life insurance",
            "insurance",
            tax_year_from=None,
            tax_year_to=2025,
        ),
    )
    memory.add(
        [0.0, 0.0, 1.0],
        payload(
            "ITA-2025",
            "123",
            "Life insurance",
            "insurance",
            tax_year_from=2026,
            tax_year_to=None,
        ),
    )

    retriever = GroundedRetriever(StubEmbedder(), memory, settings())

    old = await retriever.search("insurance", top_k=6, tax_year=2024)
    new = await retriever.search("insurance", top_k=6, tax_year=2027)

    assert [p.act for p in old] == ["ITA-1961"]
    assert [p.act for p in new] == ["ITA-2025"]


@pytest.mark.anyio
async def test_verify_rejects_an_index_built_by_a_different_model():
    """The failure that returns nonsense instead of raising, if unchecked."""
    memory = InMemoryStore()
    memory.add([0.0] * 768, payload("ITA-1961", "1", "Short title", "text"))

    retriever = GroundedRetriever(StubEmbedder(dimensions=1536), memory, settings())

    with pytest.raises(RuntimeError, match="1536"):
        await retriever.verify()


@pytest.mark.anyio
async def test_a_store_failure_on_the_exact_lookup_is_survivable():
    """Similarity results alone are still a usable answer."""

    class BrokenOnFilter(InMemoryStore):
        async def search(self, query: VectorQuery):
            if query.section_numbers:
                raise OSError("store went away")

            return await super().search(query)

    memory = BrokenOnFilter()
    memory.add([1.0, 0.0, 0.0], payload("ITA-1961", "234A", "Interest", "interest"))

    retriever = GroundedRetriever(StubEmbedder(), memory, settings())

    passages = await retriever.search("interest under section 234A", top_k=6)

    assert [p.section_number for p in passages] == ["234A"]
