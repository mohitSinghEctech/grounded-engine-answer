"""The out-of-scope marker, pinned.

The regex has to be strict in one direction and forgiving in the other: it
must not fire on an answer that merely discusses scope, and it must still
fire when the model wraps the line in bold or a bullet, which it does
unbidden.
"""

from app.scope import NOT_IN_CORPUS, OUT_OF_SCOPE, REFUSALS, split_marker


def test_plain_marker_is_detected_and_stripped():
    flagged, remainder = split_marker(f"{OUT_OF_SCOPE}\nThe provisions cover rates.")

    assert flagged == "out_of_scope"
    assert remainder == "The provisions cover rates."


def test_marker_alone_falls_back_to_the_canonical_refusal():
    flagged, remainder = split_marker(OUT_OF_SCOPE)

    assert flagged == "out_of_scope"
    assert remainder == REFUSALS["out_of_scope"]


def test_not_in_corpus_is_a_distinct_signal():
    flagged, remainder = split_marker(f"{NOT_IN_CORPUS}\nOnly rates are covered.")

    assert flagged == "not_in_corpus"
    assert remainder == "Only rates are covered."


def test_not_in_corpus_tolerates_wrapping_too():
    flagged, _ = split_marker("**REFUSE: NOT-IN-CORPUS**\nnothing here")

    assert flagged == "not_in_corpus"


def test_markdown_wrapping_still_matches():
    """Models bold things and add bullets without being asked."""
    for wrapped in (
        "**REFUSE: OUT_OF_SCOPE**\nOnly rates are covered.",
        "- REFUSE: OUT_OF_SCOPE\nOnly rates are covered.",
        "## REFUSE: OUT_OF_SCOPE\nOnly rates are covered.",
        "REFUSE:OUT_OF_SCOPE\nOnly rates are covered.",
        "refuse: out-of-scope\nOnly rates are covered.",
        "  \n REFUSE: OUT OF SCOPE \nOnly rates are covered.",
    ):
        flagged, remainder = split_marker(wrapped)

        assert flagged == "out_of_scope", wrapped
        assert remainder == "Only rates are covered."


def test_an_answer_that_merely_mentions_scope_does_not_match():
    """The whole point of anchoring: a discussion is not a signal."""
    answer = (
        "The deduction under (ITA-2025 s.123) applies. A question about which "
        "regime to choose would be REFUSE: OUT_OF_SCOPE under rule 3."
    )

    flagged, remainder = split_marker(answer)

    assert flagged is None
    assert remainder == answer


def test_an_ordinary_answer_is_returned_untouched():
    answer = "Health insurance premium is deductible under (ITA-1961 s.80D)."

    flagged, remainder = split_marker(answer)

    assert flagged is None
    assert remainder == answer
