import re

from app.services.base import Passage

REFUSAL = (
    "The provisions available to me do not cover this question. "
    "I can only answer from the text of the Act I have been given."
)

_SYSTEM = """You answer questions about Indian income tax law using ONLY the \
numbered provisions supplied below.

Rules:
1. Every factual claim must cite the provision it comes from, in the form \
(ITA-2025 s.123). Cite the section, never the passage number.
2. Use ONLY the supplied provisions. If they do not answer the question, say \
so plainly and stop. Do not answer from general knowledge.
3. State what the law says. Do not advise what the reader should do, and do \
not estimate anyone's tax.
4. If the provisions answer the question only partly, answer that part and \
say which part is not covered.
5. Quote figures exactly as they appear, including the rupee symbol.

Provisions:
{provisions}"""


def format_passage(index: int, passage: Passage) -> str:
    if passage.section_number:
        heading = f"[{index}] {passage.act} s.{passage.section_number}"
        if passage.section_title:
            heading += f" — {passage.section_title}"
    else:
        heading = (
            f"[{index}] {passage.act} (uncitable chunk, page {passage.page_start})"
        )

    return f"{heading}\n{passage.text.strip()}"


def build_prompt(question: str, passages: list[Passage]) -> str:
    provisions = "\n\n".join(
        format_passage(i, p) for i, p in enumerate(passages, start=1)
    )

    return f"{_SYSTEM.format(provisions=provisions)}\n\nQuestion: {question}\n\nAnswer:"


_CITATION = re.compile(r"\b([A-Z]{2,6}-\d{4})\s+s\.\s*([0-9A-Za-z][0-9A-Za-z\-]*)")


def extract_citations(answer: str) -> set[tuple[str, str]]:
    """Pull (act, section) pairs out of an answer, e.g. "(ITA-2025 s.123)"."""
    return {
        (match.group(1), match.group(2).rstrip(".,;)"))
        for match in _CITATION.finditer(answer)
    }


def split_citations(
    answer: str, passages: list[Passage]
) -> tuple[list[Passage], set[tuple[str, str]]]:
    """Separate cited passages from citations that were never supplied.

    A citation the model produced for a section it was not given is a
    fabrication - the most dangerous failure this system can have, because
    it looks exactly like a correct answer.
    """
    supplied = {(p.act, p.section_number): p for p in passages if p.section_number}

    cited: list[Passage] = []
    unsupported: set[tuple[str, str]] = set()

    for reference in extract_citations(answer):
        passage = supplied.get(reference)

        if passage is None:
            unsupported.add(reference)
        elif passage not in cited:
            cited.append(passage)

    return cited, unsupported
