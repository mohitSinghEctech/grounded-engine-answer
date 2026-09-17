"""The request-shaping rules, pinned.

These were all established against the live APIs: OpenAI rejects
reasoning_effort on gpt-4.1-mini with "Unrecognized request argument
supplied: reasoning_effort", a 400 on the first call. A unit test is the
only thing that keeps a future edit from reintroducing that.
"""

import pytest

from app.vendors import VENDORS, completion_kwargs, resolve


def test_unknown_vendor_names_the_known_ones():
    with pytest.raises(ValueError, match="gemini, openai"):
        resolve("anthropic")


@pytest.mark.parametrize("name", sorted(VENDORS))
def test_every_vendor_defaults_to_an_economy_model(name):
    """The cheap default is the point; a costly one should not slip in."""
    vendor = resolve(name)

    assert vendor.is_economy(vendor.default_model)


def test_openai_chat_model_gets_max_tokens_and_no_reasoning_effort():
    kwargs = completion_kwargs(
        vendor=resolve("openai"),
        model="gpt-4.1-mini",
        max_tokens=1200,
        reasoning_effort="low",
    )

    assert kwargs == {"max_tokens": 1200}


def test_openai_reasoning_model_gets_completion_tokens_and_effort():
    kwargs = completion_kwargs(
        vendor=resolve("openai"),
        model="gpt-5-mini",
        max_tokens=1200,
        reasoning_effort="low",
    )

    assert kwargs == {"max_completion_tokens": 1200, "reasoning_effort": "low"}


def test_pinned_model_dates_resolve_like_their_alias():
    """gpt-5-mini-2025-08-07 must shape the same as gpt-5-mini."""
    vendor = resolve("openai")

    assert vendor.is_reasoning("gpt-5-mini-2025-08-07")
    assert not vendor.is_reasoning("gpt-4.1-mini-2025-04-14")


def test_gemini_keeps_reasoning_effort_on_a_flash_model():
    kwargs = completion_kwargs(
        vendor=resolve("gemini"),
        model="gemini-3.7-flash",
        max_tokens=1200,
        reasoning_effort="low",
    )

    assert kwargs["reasoning_effort"] == "low"


def test_reasoning_effort_is_omitted_when_empty():
    kwargs = completion_kwargs(
        vendor=resolve("openai"),
        model="gpt-5-mini",
        max_tokens=1200,
        reasoning_effort="",
    )

    assert "reasoning_effort" not in kwargs


def test_gemini_keeps_max_tokens_despite_accepting_reasoning_effort():
    """The two concerns are separate, and conflating them broke Gemini.

    Sending max_completion_tokens to Gemini is accepted but behaves
    differently: with a ceiling of 20 and 79 reasoning tokens, the model
    spent the whole allowance thinking and returned no content.
    """
    kwargs = completion_kwargs(
        vendor=resolve("gemini"),
        model="gemini-3.7-flash",
        max_tokens=1200,
        reasoning_effort="low",
    )

    assert kwargs == {"max_tokens": 1200, "reasoning_effort": "low"}
