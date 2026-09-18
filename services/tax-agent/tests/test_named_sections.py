"""Which questions name a section outright, and which only describe one.

Similarity search is good at the second and useless at the first: asked
"what is section 139 about?" it returns 139C, 139A and 139B, because a
section number carries almost no semantic signal. So the extractor has to
fire on a named section and stay silent otherwise - a false positive drags
an irrelevant provision into the window at the top.
"""

import pytest

from app.retrieval.sections import named_sections


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is section 139 about?", ["139"]),
        ("Section 80C of the 1961 Act corresponds to which section?", ["80C"]),
        ("...was section 234A in the 1961 Act. What is it now?", ["234A"]),
        ("...found in the 2025 Act, previously section 80GG?", ["80GG"]),
        ("declare on presumptive basis u/s 44AD", ["44AD"]),
        ("the deduction under s.115BAC", ["115BAC"]),
        ("sections 80C and 80D both apply", ["80C", "80D"]),
    ],
)
def test_named_sections_are_found(question, expected):
    assert named_sections(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        # A year is four digits and never follows the word "section".
        "Which deduction applies to life insurance premium for FY 2026-27?",
        "What does the Income-tax Act 2025 define as a tax year?",
        "What is the penalty for under-reporting of income?",
        "Which ITR form should a salaried individual file?",
    ],
)
def test_questions_that_name_no_section_return_nothing(question):
    assert named_sections(question) == []


def test_the_same_section_named_twice_appears_once():
    assert named_sections("section 80C ... section 80c again") == ["80C"]


def test_a_reference_written_in_words_is_still_a_citation():
    """ "ITA-2025 section 124" names the same provision as "ITA-2025 s.124"."""
    from app.prompt import extract_citations

    assert extract_citations("covered under ITA-2025 section 124") == {
        ("ITA-2025", "124")
    }
    assert extract_citations("see ITA-1961 s.80C") == {("ITA-1961", "80C")}
    assert extract_citations("DEPT-GUIDANCE s.individual-salaried says") == {
        ("DEPT-GUIDANCE", "individual-salaried")
    }
