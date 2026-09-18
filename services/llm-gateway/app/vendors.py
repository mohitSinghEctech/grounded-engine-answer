"""Which vendor, which model, and what that combination will accept.

Every vendor here speaks the OpenAI wire format, but none of them speak the
same dialect of it. Two differences have bitten this service:

  reasoning_effort   Gemini's compatibility layer takes it on every model.
                     OpenAI takes it only on reasoning models and rejects it
                     outright otherwise - "Unrecognized request argument
                     supplied: reasoning_effort", a 400, on every call.

  token ceiling      Older chat models take max_tokens. Reasoning models
                     take max_completion_tokens and refuse max_tokens.

Sending the wrong one is not a degraded answer, it is a hard 400 on the
first request, so the choice cannot be left to the caller. Switching vendor
or model is a change to this file plus LLM_VENDOR and LLM_MODEL - never a
change to the client.

Defaults are deliberately the cheapest capable model each vendor offers. A
non-mini model still works; it just says so in the log, because an eval set
is run tens of times and the model choice is the whole bill.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Vendor:
    """One provider's endpoint and the dialect its models expect."""

    name: str
    base_url: str

    # Cheapest model that can still follow the citation rules in the prompt.
    default_model: str

    # Parameter name for the output ceiling on non-reasoning models.
    token_limit_param: str = "max_tokens"

    # Models that accept reasoning_effort. Matched by prefix, so a pinned
    # date suffix (gpt-5-mini-2025-08-07) resolves the same as the alias.
    reasoning_prefixes: tuple[str, ...] = ()

    # Whether a reasoning model also needs max_completion_tokens in place of
    # max_tokens. True for OpenAI, where reasoning models reject max_tokens
    # outright. False for Gemini, which accepts reasoning_effort on every
    # model but has served max_tokens in production for days - and where
    # switching to max_completion_tokens made a small budget fail outright:
    # 79 reasoning tokens consumed a ceiling of 20, so the model spent the
    # whole allowance thinking and returned no content at all.
    reasoning_needs_completion_tokens: bool = True

    # Substrings marking a model as cheap. Used only to decide whether to
    # warn; it never changes the request.
    economy_markers: tuple[str, ...] = field(default=("mini", "nano", "flash"))

    def is_reasoning(self, model: str) -> bool:
        return model.startswith(self.reasoning_prefixes)

    def is_economy(self, model: str) -> bool:
        return any(marker in model for marker in self.economy_markers)


VENDORS: dict[str, Vendor] = {
    "openai": Vendor(
        name="openai",
        base_url="https://api.openai.com/v1/",
        default_model="gpt-4.1-mini",
        token_limit_param="max_tokens",
        reasoning_prefixes=("o1", "o3", "o4", "gpt-5"),
    ),
    "gemini": Vendor(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-3.7-flash",
        token_limit_param="max_tokens",
        # Gemini accepts reasoning_effort across the board, so every model
        # counts as a reasoning model for request-shaping purposes.
        reasoning_prefixes=("gemini",),
        # Gemini takes max_tokens, not max_completion_tokens.
        reasoning_needs_completion_tokens=False,
    ),
}


def resolve(name: str) -> Vendor:
    """Look up a vendor, failing at startup rather than on first request."""
    try:
        return VENDORS[name]
    except KeyError:
        known = ", ".join(sorted(VENDORS))
        raise ValueError(
            f"Unknown LLM_VENDOR {name!r}. Known vendors: {known}."
        ) from None


def completion_kwargs(
    vendor: Vendor,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    """The vendor-specific half of a chat completion request.

    Everything else about the call - model, messages - is identical across
    vendors, so only the parts that differ live here.
    """
    reasoning = vendor.is_reasoning(model)

    ceiling = (
        "max_completion_tokens"
        if reasoning and vendor.reasoning_needs_completion_tokens
        else vendor.token_limit_param
    )

    kwargs: dict[str, Any] = {ceiling: max_tokens}

    if reasoning and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    return kwargs


def describe(vendor: Vendor, model: str) -> None:
    """Log the resolved choice once, at startup, and flag an expensive one."""
    logger.info(
        "LLM vendor resolved | vendor=%s | model=%s | base_url=%s | "
        "reasoning=%s | token_param=%s",
        vendor.name,
        model,
        vendor.base_url,
        vendor.is_reasoning(model),
        "max_completion_tokens"
        if vendor.is_reasoning(model) and vendor.reasoning_needs_completion_tokens
        else vendor.token_limit_param,
    )

    if not vendor.is_economy(model):
        logger.warning(
            "Model %r is not an economy model. An eval run is 40 requests of "
            "roughly 3,000 prompt tokens each and gets repeated; %r is the "
            "cheap default for this vendor.",
            model,
            vendor.default_model,
        )
