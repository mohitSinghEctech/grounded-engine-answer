"""The functions the model may ask for, and the schemas that describe them.

Two halves that must stay in step:

  * SCHEMAS - what the model reads. This is prompt engineering, not
    documentation: the model picks a tool by reading the description, so
    "use this when / do NOT use this when" earns its space. The
    parameters constrain what it may send; the description persuades it
    to send the right thing.

  * ToolBox - what actually runs. Every handler returns plain JSON-able
    data, including its failures: a tool that raises kills the request,
    while a tool that returns {"found": false} lets the model try
    something else. That difference is most of what makes a loop an agent
    rather than a script.

Nothing here re-implements retrieval. `search_provisions` and
`get_section` both go through `GroundedRetriever.search`, which already
detects a named section in the question and takes the exact-lookup path -
so the tools inherit the score floor, the per-section cap and the parser
fixes that the eval harness measured.
"""

from __future__ import annotations

import logging
from typing import Any

from app.progress import Progress
from app.services.base import Passage, Retriever

logger = logging.getLogger(__name__)

#: The model's own way of stopping. Returned by `cannot_answer` so the loop
#: can tell "declared failure" from "a tool happened to return nothing".
GAVE_UP = "__gave_up__"

ACTS = ("ITA-1961", "ITA-2025", "DEPT-GUIDANCE")

SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "search_provisions",
            "description": (
                "Search Indian income-tax provisions by meaning. Use this "
                "for questions about what the law says or allows. Do NOT "
                "use it when the question already names a section number - "
                "use get_section for that, it is exact. Returns up to 6 "
                "provisions, each with act, section_number, section_title "
                "and text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for, in the words the statute "
                            "would use rather than the user's words."
                        ),
                    },
                    "act": {
                        "type": "string",
                        "enum": list(ACTS),
                        "description": (
                            "Restrict to one Act. Omit to search all of "
                            "them, which is usually right."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_section",
            "description": (
                "Fetch one provision by its exact section number, e.g. "
                "'80D' or '234A'. Use whenever the question names a "
                "section, and after map_section tells you the counterpart "
                "number. Returns the provision, or found=false."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_number": {
                        "type": "string",
                        "description": "Just the number, e.g. '80D'. No 's.' prefix.",
                    },
                    "act": {"type": "string", "enum": list(ACTS)},
                },
                "required": ["section_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_section",
            "description": (
                "Given a section of the 2025 Act, find the corresponding "
                "section of the 1961 Act. Use this when asked what changed "
                "between the two Acts: map first, then get_section on both "
                "numbers, then compare the texts. Returns found=false when "
                "the corpus has no mapping for that section - then say so "
                "rather than guessing a number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section_number": {"type": "string"},
                    "act": {
                        "type": "string",
                        "enum": list(ACTS),
                        "description": "The Act the section number belongs to.",
                    },
                },
                "required": ["section_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cannot_answer",
            "description": (
                "Call this instead of answering when the provisions you "
                "have found do not cover the question. Better than a vague "
                "answer. Say what you searched for, so the gap is on record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why the provisions do not answer it.",
                    },
                    "tried": {
                        "type": "string",
                        "description": "The searches and sections you looked at.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
)


class ToolBox:
    """Runs the tools, and remembers every provision they returned.

    That memory is the grounding invariant: the final answer's citations
    are checked against the union of everything any tool surfaced, so
    agency cannot become a side door around citation verification. If
    anything, it raises the stakes - there are now four places a
    fabrication could enter instead of one.
    """

    def __init__(self, retriever: Retriever, top_k: int, progress: Progress):
        self.retriever = retriever
        self.top_k = top_k
        self.progress = progress

        #: Everything seen this run, de-duplicated by (act, section, text).
        self.seen: list[Passage] = []
        self._keys: set[tuple[str, str | None, str]] = set()

    # ── dispatch ──────────────────────────────────────────────────────

    async def run(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one tool by the name the model chose.

        A dict lookup, never getattr - the name is model output, so it may
        only select from a fixed set. An unknown name is data, not an
        exception: the model misremembered and can correct itself.
        """
        handler = {
            "search_provisions": self._search,
            "get_section": self._get_section,
            "map_section": self._map_section,
            "cannot_answer": self._cannot_answer,
        }.get(name)

        if handler is None:
            logger.warning("Model asked for an unknown tool | name=%s", name)
            return {
                "error": f"No tool named {name!r}.",
                "available": [s["function"]["name"] for s in SCHEMAS],
            }

        try:
            return await handler(**arguments)
        except TypeError as exc:
            # Wrong or missing arguments - the model wrote the JSON, so
            # this is its mistake to fix, not a crash.
            return {"error": f"Bad arguments for {name}: {exc}"}

    # ── the tools ─────────────────────────────────────────────────────

    async def _search(self, query: str, act: str | None = None) -> Any:
        passages = await self._retrieve(query, act)

        if not passages:
            return {"found": False, "note": "Nothing matched. Try other wording."}

        return {"found": True, "provisions": [self._render(p) for p in passages]}

    async def _get_section(self, section_number: str, act: str | None = None) -> Any:
        # Phrased as a question so the retriever's named-section detector
        # fires and takes the exact path, score floor bypassed.
        passages = await self._retrieve(f"section {section_number}", act)

        exact = [
            p
            for p in passages
            if (p.section_number or "").lower() == section_number.lower()
        ]

        if not exact:
            return {
                "found": False,
                "note": (
                    f"No section {section_number} in the corpus"
                    + (f" for {act}" if act else "")
                    + "."
                ),
            }

        return {"found": True, "provisions": [self._render(p) for p in exact]}

    async def _map_section(self, section_number: str, act: str | None = None) -> Any:
        passages = await self._retrieve(f"section {section_number}", act)

        for passage in passages:
            if (passage.section_number or "").lower() != section_number.lower():
                continue

            if passage.maps_to:
                return {
                    "found": True,
                    "from": {"act": passage.act, "section": passage.section_number},
                    "to": {"act": "ITA-1961", "section": passage.maps_to},
                }

        return {
            "found": False,
            "note": (
                "The corpus has no cross-Act mapping for that section. Do "
                "not guess a number: search the other Act by the section's "
                "subject instead, or say the mapping is unavailable."
            ),
        }

    async def _cannot_answer(self, reason: str, tried: str = "") -> Any:
        await self.progress.emit("gave_up", reason=reason[:200])
        logger.info("Agent declared it cannot answer | reason=%s", reason[:200])

        return {"stop": GAVE_UP, "reason": reason, "tried": tried}

    # ── shared plumbing ───────────────────────────────────────────────

    async def _retrieve(self, query: str, act: str | None) -> list[Passage]:
        # Over-fetch when filtering by Act, because the filter happens
        # here rather than in the store: VectorQuery speaks tax_year, not
        # act, and translating one to the other would bake a calendar
        # assumption into a tool description. Post-filtering costs a few
        # wasted candidates and stays exact.
        limit = self.top_k * 3 if act else self.top_k

        passages = await self.retriever.search(question=query, top_k=limit)

        if act:
            passages = [p for p in passages if p.act == act]

        kept = passages[: self.top_k]
        self._remember(kept)

        return kept

    def _remember(self, passages: list[Passage]) -> None:
        for passage in passages:
            key = (passage.act, passage.section_number, passage.text[:80])

            if key not in self._keys:
                self._keys.add(key)
                self.seen.append(passage)

    @staticmethod
    def _render(passage: Passage) -> dict[str, Any]:
        """What the model sees. Trimmed, because every character is a token."""
        return {
            "act": passage.act,
            "section_number": passage.section_number,
            "section_title": passage.section_title,
            "text": passage.text[:1200],
        }
