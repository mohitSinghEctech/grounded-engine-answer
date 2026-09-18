import re

from app.services.base import Passage

REFUSAL = (
    "The provisions available to me do not cover this question. "
    "I can only answer from the text of the Act I have been given."
)

_SYSTEM = """You answer questions about Indian income tax law using ONLY the \
numbered provisions supplied below.

Rules:
1. Every factual claim must cite the provision it comes from, using the act \
code and section exactly as they appear in the provision's heading:
   (ITA-2025 s.123)          a section of the 2025 Act
   (ITA-1961 s.80C)          a section of the 1961 Act
   (DEPT-GUIDANCE s.individual-salaried)   a departmental guidance page
Copy the act code character for character, and always use the "s." form: \
write (ITA-2025 s.124), never "ITA-2025 section 124" or "section 124", \
which do not count as citations. Never write a bracketed passage number \
such as [1] or [2] - those number the list below and mean nothing to the \
reader.
2. Use ONLY the supplied provisions, and never cite a section number that \
does not appear in a heading below, even if you know it exists. Do not \
answer from general knowledge. If they do not answer the question, say so \
plainly and stop.
3. State what the law says. Do not advise what the reader should do, and do \
not estimate anyone's tax. If the question asks which option to choose, \
what is best or most beneficial, or what someone's tax comes to, make the \
FIRST line of your reply exactly:
   REFUSE: OUT_OF_SCOPE
and then say briefly what the provisions do cover. Emit that line only for \
those cases - not when the provisions simply fail to answer a legitimate \
question, which rule 2 already covers.
4. If the provisions answer the question only partly, answer that part and \
say which part is not covered.
5. Quote figures exactly as they appear, including the rupee symbol.
6. The Act rarely uses everyday names. "NPS" is the "pension scheme of \
the Central Government"; "HRA" is "house rent allowance". Answer when \
the provisions cover the substance of the question, and say which term \
the Act itself uses. Refuse only when the substance is absent, never \
because the wording differs.

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


def build_prompt(
    question: str, passages: list[Passage], note: str | None = None
) -> str:
    """The provisions, the rules, and the question.

    `note` is for a second attempt: it lands between the question and the
    answer, which is the last thing the model reads and so the strongest
    place to correct a specific mistake the first attempt made. Nothing
    uses it on a first attempt, and the prompt is byte-identical without it.
    """
    provisions = "\n\n".join(
        format_passage(i, p) for i, p in enumerate(passages, start=1)
    )

    correction = f"\n\n{note.strip()}" if note else ""

    return (
        f"{_SYSTEM.format(provisions=provisions)}"
        f"\n\nQuestion: {question}{correction}\n\nAnswer:"
    )


# Act codes end in a year (ITA-2025) or a word (DEPT-GUIDANCE), so the
# second half cannot assume digits. The hyphen is required, which is why
# the guidance label keeps one.
#: "s.124" is asked for, but "section 124" identifies the same supplied
#: provision just as unambiguously, and rejecting it turns a correct answer
#: into an ungrounded one. SM-04 answered "covered under ITA-2025 section
#: 124" across several runs and scored as a refusal every time.
_CITATION = re.compile(
    r"\b([A-Z][A-Z0-9]{1,5}-[A-Z0-9]{2,10})\s+"
    r"(?:s\.\s*|sections?\s+)"
    r"([0-9A-Za-z][0-9A-Za-z\-]*)"
)


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
