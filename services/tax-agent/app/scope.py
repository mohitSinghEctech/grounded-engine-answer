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

#: What rule 3 asks the model to emit as its first line. Deliberately not a
#: phrase that could occur in ordinary prose about tax law.
MARKER = "REFUSE: OUT_OF_SCOPE"

#: Tolerates the wrappers models add unbidden - bold, a leading bullet, a
#: trailing colon - without matching a mere mention of the phrase mid
#: sentence, which is why it is anchored to the start.
_MARKER = re.compile(
    r"\A[\s>*_#-]*REFUSE\s*:\s*OUT[_\s-]?OF[_\s-]?SCOPE[\s*_:.-]*",
    re.IGNORECASE,
)

REFUSAL = (
    "I can only state what the provisions say, not advise which option to "
    "choose or work out what someone owes."
)


def split_marker(answer: str) -> tuple[bool, str]:
    """Separate the out-of-scope signal from whatever followed it.

    Returns whether the marker was present and the answer with it removed,
    so the model's own account of what the provisions *do* cover survives
    into the refusal instead of being thrown away.
    """
    match = _MARKER.match(answer.lstrip())

    if not match:
        return False, answer

    remainder = answer.lstrip()[match.end() :].strip()

    return True, remainder or REFUSAL
