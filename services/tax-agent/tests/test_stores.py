"""The store registry, and the two ways a Qdrant index can be reached.

The registry is what makes a single self-contained image possible: the
service talks to a server in development and to a directory in the
deployable image, and nothing above the store layer knows which.
"""

import pytest

from app.retrieval import stores
from app.retrieval.base import VectorQuery
from app.retrieval.stores import InMemoryStore


class FakeSettings:
    """Only the fields the store factories read."""

    def __init__(self, **overrides):
        self.vector_store_provider = "qdrant"
        self.qdrant_url = None
        self.qdrant_path = "data/index"
        self.qdrant_collection = "ita_sections"
        self.__dict__.update(overrides)


def test_unknown_provider_names_the_known_ones():
    with pytest.raises(ValueError, match="memory, qdrant, qdrant_embedded"):
        stores.create(FakeSettings(vector_store_provider="pinecone"))


def test_the_server_store_demands_a_url_and_says_what_to_use_instead():
    """A URL-less "qdrant" is the mistake someone makes reaching for the
    embedded one, so the error names the alternative."""
    with pytest.raises(ValueError, match="qdrant_embedded"):
        stores.create(FakeSettings(vector_store_provider="qdrant", qdrant_url=None))


def test_the_embedded_store_refuses_a_path_that_does_not_exist():
    """Fail at startup, where it fails the health check, rather than on the
    first question."""
    with pytest.raises(RuntimeError, match="index-standalone"):
        stores.create(
            FakeSettings(
                vector_store_provider="qdrant_embedded",
                qdrant_path="/nonexistent/index",
            )
        )


def test_the_memory_store_needs_no_configuration_at_all():
    store = stores.create(FakeSettings(vector_store_provider="memory"))

    assert store.name == "memory:0"


@pytest.mark.anyio
async def test_the_memory_store_honours_a_section_number_filter():
    """Every store must support the exact-lookup path, not just Qdrant."""
    memory = InMemoryStore()
    memory.add([1.0], {"act": "ITA-1961", "section_number": "139"})
    memory.add([1.0], {"act": "ITA-1961", "section_number": "139A"})

    hits = await memory.search(
        VectorQuery(vector=[1.0], limit=10, section_numbers=("139",))
    )

    assert [h.payload["section_number"] for h in hits] == ["139"]


@pytest.mark.anyio
async def test_verify_rejects_a_dimension_mismatch():
    memory = InMemoryStore()
    memory.add([0.0] * 768, {"act": "ITA-1961", "section_number": "1"})

    with pytest.raises(RuntimeError, match="1536"):
        await memory.verify(1536)
