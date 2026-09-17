"""Retrieval policy: what to ask for, and what to keep.

Everything provider-specific lives in `embedders.py` and `stores.py`. What
is left here is the part that is about *this corpus* rather than about any
database, and it is the part worth reading:

  1. Embed the question.
  2. If the question names a section, fetch those sections exactly. They are
     what was asked for; similarity is not the question.
  3. Search by similarity for everything else.
  4. Put the exact matches first, cap how much of the window any one section
     may occupy, and truncate to top_k.

Step 4 is why the cap exists: a section is stored as several chunks - 3,286
of them over 1,423 sections - and without a cap one section takes the whole
window. Measured: s.234A held 4 of 6 slots, and two chunks of s.24 crowded
out s.23 entirely, so an answer about house property silently lost a third
of the law.
"""

from __future__ import annotations

import logging
import time

from app.config import Settings
from app.progress import Progress
from app.retrieval.base import Embedder, Hit, VectorQuery, VectorStore
from app.retrieval.sections import named_sections
from app.services.base import Passage

logger = logging.getLogger(__name__)


class GroundedRetriever:
    """Composes an embedder and a store into the `Retriever` protocol.

    Holds no provider knowledge. Swap Qdrant for Chroma and this class does
    not change; that is the whole point of the split.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        settings: Settings,
    ):
        self.embedder = embedder
        self.store = store
        self.settings = settings

    async def verify(self) -> None:
        """Refuse to start if the index cannot serve this embedder."""
        await self.store.verify(self.embedder.dimensions)

    @staticmethod
    def _to_passage(hit: Hit) -> Passage:
        """Payload shape is a fact about this corpus, so it is known here.

        A store returns whatever it stored; only this method knows that
        "tax_year_from" and "section_number" mean anything.
        """
        payload = hit.payload

        return Passage(
            text=payload.get("text", ""),
            score=hit.score,
            act=payload.get("act", "unknown"),
            section_number=payload.get("section_number"),
            section_title=payload.get("section_title"),
            page_start=payload.get("page_start") or payload.get("page"),
            corpus_date=payload.get("corpus_date", "unknown"),
        )

    def _cap_per_section(self, passages: list[Passage]) -> list[Passage]:
        """Limit how much of the window any one section may occupy.

        Input order is score order and is preserved, so the best chunks of
        each section survive and the rest are dropped. Lower-ranked chunks
        of a *different* section move up into the freed slots, which is the
        point: breadth over repetition.

        The cap is 2 rather than 1 because a guidance page carries one
        section_number across many chunks; collapsing to one would throw
        away most of a page the procedural questions need.
        """
        limit = self.settings.retrieval_max_per_section
        seen: dict[tuple[str, str | None], int] = {}
        kept: list[Passage] = []

        for passage in passages:
            key = (passage.act, passage.section_number)

            if seen.get(key, 0) >= limit:
                continue

            seen[key] = seen.get(key, 0) + 1
            kept.append(passage)

        return kept

    async def _exact(
        self,
        vector: list[float],
        numbers: list[str],
        tax_year: int | None,
        limit: int,
    ) -> list[Passage]:
        """Chunks of the sections the question named, best-matching first.

        Still ordered by similarity *within* those sections, so a long
        section contributes the part that answers the question rather than
        whichever chunk happens to come first.

        No score floor: the user named this section, so its relevance is
        established by the question and not by cosine distance.
        """
        try:
            hits = await self.store.search(
                VectorQuery(
                    vector=vector,
                    limit=limit,
                    tax_year=tax_year,
                    section_numbers=tuple(numbers),
                )
            )
        except Exception as exc:
            # A failed exact lookup must not fail the request; similarity
            # search alone still produces a usable answer.
            logger.warning(
                "Named-section lookup failed, continuing | reason=%s",
                type(exc).__name__,
            )
            return []

        return [self._to_passage(hit) for hit in hits]

    async def search(
        self,
        question: str,
        top_k: int,
        tax_year: int | None = None,
        progress: Progress | None = None,
    ) -> list[Passage]:
        # A discarding emitter when nobody is listening, so there is one
        # code path rather than a streaming one and a plain one.
        report = progress or Progress.discarded()

        # Embedding is a network call to a third party; the store query is
        # usually local. Timed and reported apart, so a slow retrieval can
        # be blamed on the right one instead of on "retrieval".
        await report.emit("embedding", model=self.embedder.name)

        embed_started = time.perf_counter()
        vector = await self.embedder.embed(question)
        embed_ms = int((time.perf_counter() - embed_started) * 1000)

        await report.emit(
            "embedded", dimensions=self.embedder.dimensions, took_ms=embed_ms
        )

        search_started = time.perf_counter()

        # Ask for more than top_k, because capping discards some. Too low
        # and the window cannot be refilled after the cap.
        over_fetched = top_k * self.settings.retrieval_over_fetch

        await report.emit(
            "searching",
            store=self.store.name,
            limit=over_fetched,
            tax_year=tax_year,
        )

        hits = await self.store.search(
            VectorQuery(vector=vector, limit=over_fetched, tax_year=tax_year)
        )

        await report.emit("searched", candidates=len(hits))

        floor = self.settings.retrieval_min_score
        above_floor = [self._to_passage(h) for h in hits if h.score >= floor]

        await report.emit(
            "filtering",
            min_score=floor,
            kept=len(above_floor),
            dropped=len(hits) - len(above_floor),
        )

        numbers = named_sections(question)
        exact: list[Passage] = []

        if numbers:
            await report.emit("looking_up", sections=numbers)

            exact = await self._exact(vector, numbers, tax_year, over_fetched)

            await report.emit("looked_up", sections=numbers, found=len(exact))

        before_cap = exact + above_floor
        passages = self._cap_per_section(before_cap)[:top_k]

        await report.emit(
            "capping",
            max_per_section=self.settings.retrieval_max_per_section,
            before=len(before_cap),
            after=len(passages),
        )

        search_ms = int((time.perf_counter() - search_started) * 1000)

        # Facts, not a sentence: the wording is the UI's business.
        await report.emit(
            "retrieved",
            count=len(passages),
            acts=sorted({p.act for p in passages}),
            sections=[
                {
                    "act": p.act,
                    "section": p.section_number,
                    "title": p.section_title,
                    "score": round(p.score, 3),
                }
                for p in passages
            ],
            took_ms=embed_ms + search_ms,
        )

        logger.info(
            "Retrieval completed | store=%s | hits=%s | above_floor=%s | "
            "named=%s | exact=%s | kept=%s | min_score=%.2f | "
            "max_per_section=%s | tax_year=%s | embed_ms=%s | search_ms=%s",
            self.store.name,
            len(hits),
            len(above_floor),
            numbers or "-",
            len(exact),
            len(passages),
            floor,
            self.settings.retrieval_max_per_section,
            tax_year,
            embed_ms,
            search_ms,
        )

        return passages

    async def close(self) -> None:
        await self.store.close()
