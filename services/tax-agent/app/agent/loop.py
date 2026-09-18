"""The loop. Three ways out, and only one of them trusts the model.

    exit 1  the model answers            - no tool_calls in its reply
    exit 2  the model calls cannot_answer - declared failure, with a reason
    exit 3  the step or token budget runs out - the backstop

Exits 1 and 2 belong to the model, which is right: it is the only thing
that knows whether it has enough to answer. Exit 3 exists because that
judgement is not reliable. A tool that errors on every call would
otherwise be answered by another attempt, forever, each turn costing more
than the last as the conversation grows - until a timeout somewhere else
kills a request that has by then spent real money.

So the bound is written as `for step in range(...)`, not `while True` with
a break. The difference is that the ceiling is structural: there is no
condition anyone can forget to update, and no model behaviour that can
extend it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools import GAVE_UP, SCHEMAS, ToolBox
from app.progress import Progress
from app.services.base import Generation, LLMGateway, Passage, Retriever

logger = logging.getLogger(__name__)

SYSTEM = """You answer questions about Indian income tax law.

Rules:
1. Use the tools to find provisions. Never answer from memory.
2. Cite every claim as ACT s.NUMBER, using the act CODE exactly: \
(ITA-1961 s.80D), (ITA-2025 s.123). "Section 123 of the Income-tax Act, \
2025" names the same provision but does NOT count as a citation. Only cite \
provisions a tool actually returned to you.
3. When a question names a section in one Act and asks about the other - \
"corresponds to", "replaces", "previously", "what changed" - call \
map_section first, then get_section on BOTH numbers, then answer from those \
two texts. map_section returns numbers only, so a counterpart you have not \
fetched with get_section is a number you cannot cite.
4. If the provisions you found do not cover the question, call \
cannot_answer rather than writing a vague answer.
5. Do not advise which option to choose and do not compute anyone's tax. \
State what the provisions say."""


@dataclass
class AgentRun:
    """What the loop did, for the caller and for the eval harness."""

    #: The final prose. Empty when it gave up or ran out of budget.
    text: str = ""

    #: Every provision any tool returned. The citation check runs against
    #: this union - agency does not get to bypass verification.
    passages: list[Passage] = field(default_factory=list)

    #: Ordered tool names, e.g. ["map_section", "get_section"]. This is the
    #: trajectory: two runs can reach the same answer at four times the
    #: cost, and only this column shows it.
    trail: list[str] = field(default_factory=list)

    steps: int = 0
    redundant_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""

    #: "answered" | "gave_up" | "budget_exhausted"
    outcome: str = "answered"
    give_up_reason: str = ""

    @property
    def finish_reason(self) -> str:
        return "stop" if self.outcome == "answered" else self.outcome


async def run_agent(
    question: str,
    retriever: Retriever,
    gateway: LLMGateway,
    progress: Progress,
    top_k: int = 6,
    max_tokens: int = 1200,
    max_steps: int = 6,
    token_budget: int = 40_000,
) -> AgentRun:
    """Let the model drive, within bounds it cannot move."""
    tools = ToolBox(retriever=retriever, top_k=top_k, progress=progress)
    run = AgentRun()

    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    # (name, arguments) already executed. A model that repeats a call
    # verbatim is looping; returning the cached answer plus a note is
    # cheaper than running it again and far cheaper than letting it spin.
    executed: dict[tuple[str, str], Any] = {}

    for step in range(max_steps):
        run.steps = step + 1

        # "turn", not "step": the SSE frame already uses "step" for the
        # step name, and a detail key of the same name collides with it.
        await progress.emit("planning", turn=run.steps, of=max_steps)

        reply: Generation = await gateway.generate(
            messages=conversation,
            max_tokens=max_tokens,
            tools=SCHEMAS,
        )

        run.model = reply.model or run.model
        run.prompt_tokens += reply.prompt_tokens
        run.completion_tokens += reply.completion_tokens
        run.total_tokens += reply.total_tokens

        # ── exit 1: it answered ───────────────────────────────────────
        if not reply.tool_calls:
            run.text = reply.text
            run.outcome = "answered"
            run.passages = tools.seen  # the citation check needs these

            logger.info(
                "Agent answered | steps=%s | tools=%s | total_tokens=%s",
                run.steps,
                run.trail,
                run.total_tokens,
            )

            return run

        # The assistant's own turn must go back in BEFORE the results, or
        # the tool messages refer to a request that is not in the
        # conversation and the provider rejects the next call.
        conversation.append(
            {
                "role": "assistant",
                "content": reply.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in reply.tool_calls
                ],
            }
        )

        gave_up_with: dict[str, Any] | None = None

        # One reply can request several calls in parallel, so this is
        # always a loop - and every one needs a result message back.
        for call in reply.tool_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                # The model wrote the JSON, so malformed JSON is its
                # mistake to correct, not a crash for us.
                result: Any = {"error": "Arguments were not valid JSON."}
                arguments = {}
            else:
                key = (call.name, json.dumps(arguments, sort_keys=True))

                if key in executed:
                    run.redundant_calls += 1
                    result = {
                        "repeat": True,
                        "note": "You already made this exact call.",
                        "result": executed[key],
                    }
                else:
                    await progress.emit(
                        "tool_call", tool=call.name, arguments=arguments
                    )

                    result = await tools.run(call.name, arguments)
                    executed[key] = result

                    await progress.emit(
                        "tool_result",
                        tool=call.name,
                        found=bool(isinstance(result, dict) and result.get("found")),
                    )

            run.trail.append(call.name)

            if isinstance(result, dict) and result.get("stop") == GAVE_UP:
                gave_up_with = result

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result)[:6000],
                }
            )

        # ── exit 2: the model declared it cannot answer ───────────────
        if gave_up_with is not None:
            run.outcome = "gave_up"
            run.give_up_reason = str(gave_up_with.get("reason", ""))
            run.passages = tools.seen

            logger.info(
                "Agent gave up | steps=%s | tools=%s | reason=%s",
                run.steps,
                run.trail,
                run.give_up_reason[:160],
            )

            return run

        # ── exit 3a: the token budget ─────────────────────────────────
        if run.total_tokens >= token_budget:
            break

    # ── exit 3b: out of steps ────────────────────────────────────────
    run.outcome = "budget_exhausted"
    run.passages = tools.seen

    # Rare enough that it should be read as a defect, not a normal path:
    # either the tools are failing or a tool description is misleading.
    logger.warning(
        "Agent budget exhausted | steps=%s | tools=%s | total_tokens=%s",
        run.steps,
        run.trail,
        run.total_tokens,
    )

    return run
