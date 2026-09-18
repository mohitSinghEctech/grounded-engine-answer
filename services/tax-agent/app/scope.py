"""The one refusal the pipeline cannot work out for itself.

Two of the three refusal reasons are structural. Retrieve nothing and the
model is never called. Cite nothing and the answer is discarded. Neither
needs an opinion.

"You could answer this but you should not" is different. Whether a question
asks for advice - which option to choose, what is best, what someone's tax
comes to - is a judgment about meaning, and no amount of code inspecting
citations can reach it. The first baseline showed the cost: asked for the
best tax-saving mutual fund, the model correctly declined *and* cited a few
sections while explaining what the corpus does hold, so `refused = not
cited` recorded "did not refuse" on a textbook-correct refusal.

So the model signals and this module decides. Prompt rule 3 asks for one
exact line when a question is out of scope; `split_marker` finds it and the
router turns that into `out_of_scope`. The model cannot refuse by being
persuasive or hedging in prose - only by emitting the token - and it cannot
suppress a refusal the structural checks have already made.
"""

from __future__ import annotations

import re

#: What the prompt asks the model to emit as its first line, one per case.
#: Deliberately phrases that cannot occur in ordinary prose about tax law.
OUT_OF_SCOPE = "REFUSE: OUT_OF_SCOPE"
NOT_IN_CORPUS = "REFUSE: NOT_IN_CORPUS"

#: Kept for callers that only care about the rule 3 marker.
MARKER = OUT_OF_SCOPE

#: Each tolerates the wrappers models add unbidden - bold, a leading bullet,
#: a trailing colon - without matching a mere mention of the phrase mid
#: sentence, which is why every pattern is anchored to the start.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "out_of_scope",
        re.compile(
            r"\A[\s>*_#-]*REFUSE\s*:\s*OUT[_\s-]?OF[_\s-]?SCOPE[\s*_:.-]*",
            re.IGNORECASE,
        ),
    ),
    (
        "not_in_corpus",
        re.compile(
            r"\A[\s>*_#-]*REFUSE\s*:\s*NOT[_\s-]?IN[_\s-]?CORPUS[\s*_:.-]*",
            re.IGNORECASE,
        ),
    ),
)

REFUSALS = {
    "out_of_scope": (
        "I can only state what the provisions say, not advise which option to "
        "choose or work out what someone owes."
    ),
    "not_in_corpus": ("The provisions available to me do not cover this question."),
}

#: Backwards-compatible name for the rule 3 refusal text.
REFUSAL = REFUSALS["out_of_scope"]


def split_marker(answer: str) -> tuple[str | None, str]:
    """Separate a refusal signal from whatever followed it.

    Returns the signal name (or None) and the answer with the marker
    removed, so the model's own account of what the provisions *do* cover
    survives into the refusal instead of being thrown away.
    """
    stripped = answer.lstrip()

    for name, pattern in _PATTERNS:
        match = pattern.match(stripped)

        if match:
            remainder = stripped[match.end() :].strip()

            return name, remainder or REFUSALS[name]

    return None, answer
