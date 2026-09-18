"""The loop's three exits, and the guards that stop it spinning.

The tests that matter here are the failure ones. A loop that works when
the model behaves is easy; what has to be pinned is what happens when it
does not - because that is the difference between an agent and an outage.
"""

import asyncio
import functools
import json

import pytest

from app.agent.loop import run_agent
from app.agent.tools import SCHEMAS, ToolBox
from app.progress import Progress, Step
from app.services.base import Generation, Passage, ToolCall


def sync(async_test):
    """Run an async test without adding a pytest plugin for one module."""

    @functools.wraps(async_test)
    def wrapper(*args, **kwargs):
        return asyncio.run(async_test(*args, **kwargs))

    return wrapper


def passage(section="80D", act="ITA-1961", maps_to=None):
    return Passage(
        text=f"text of {section}",
        score=0.9,
        act=act,
        section_number=section,
        section_title=f"Title of {section}",
        page_start=12,
        corpus_date="2026-09-14",
        maps_to=maps_to,
    )


def answer(text="Premium deduction (ITA-1961 s.80D)."):
    return Generation(
        text=text,
        model="stub",
        prompt_tokens=100,
        completion_tokens=20,
        reasoning_tokens=0,
        total_tokens=120,
        finish_reason="stop",
    )


def wants(name, arguments, call_id="call_1"):
    """A reply that asks for one tool instead of answering."""
    return Generation(
        text="",
        model="stub",
        prompt_tokens=100,
        completion_tokens=10,
        reasoning_tokens=0,
        total_tokens=110,
        finish_reason="tool_calls",
        tool_calls=(ToolCall(id=call_id, name=name, arguments=json.dumps(arguments)),),
    )


class ScriptedGateway:
    """Replies in order, and records every conversation it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.conversations = []
        self.tools_seen = []

    async def generate(self, prompt=None, max_tokens=2000, *, messages=None, tools=()):
        self.conversations.append(list(messages or []))
        self.tools_seen.append(list(tools))

        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class StubRetriever:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def search(self, question, top_k, tax_year=None, progress=None):
        self.calls.append(question)

        if not self.results:
            return []

        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class Recorder(Progress):
    def __init__(self):
        super().__init__(queue=None)
        self.steps: list[Step] = []

    async def emit(self, name, **detail):
        self.steps.append(Step(name=name, detail=detail))

    def names(self):
        return [s.name for s in self.steps]


# --------------------------------------------------------------------------
# exit 1 — the model answers
# --------------------------------------------------------------------------


@sync
async def test_answers_after_one_tool_call():
    gateway = ScriptedGateway(
        wants("search_provisions", {"query": "health insurance deduction"}),
        answer(),
    )
    retriever = StubRetriever([passage()])
    recorder = Recorder()

    run = await run_agent(
        question="What can I deduct for health insurance?",
        retriever=retriever,
        gateway=gateway,
        progress=recorder,
    )

    assert run.outcome == "answered"
    assert run.text == "Premium deduction (ITA-1961 s.80D)."
    assert run.trail == ["search_provisions"]
    assert run.steps == 2
    # Tokens accumulate across turns, not just the last call.
    assert run.total_tokens == 230
    assert "tool_call" in recorder.names()


@sync
async def test_provisions_survive_to_the_answer():
    """Without this the citation check has nothing to verify against.

    A run that returned no passages would score every correct agent
    answer as ungrounded - a silent, total failure of the eval.
    """
    gateway = ScriptedGateway(wants("get_section", {"section_number": "80D"}), answer())

    run = await run_agent(
        question="What is section 80D?",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
    )

    assert [p.section_number for p in run.passages] == ["80D"]


@sync
async def test_the_conversation_is_assembled_in_the_right_order():
    """assistant turn, then one tool message per call, ids matching.

    Get this wrong and the provider rejects the next request - so the
    shape is a contract, not a style preference.
    """
    gateway = ScriptedGateway(
        wants("get_section", {"section_number": "80D"}, call_id="call_abc"),
        answer(),
    )

    await run_agent(
        question="80D?",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
    )

    second = gateway.conversations[1]
    roles = [m["role"] for m in second]

    assert roles == ["system", "user", "assistant", "tool"]
    assert second[2]["tool_calls"][0]["id"] == "call_abc"
    assert second[3]["tool_call_id"] == "call_abc"


@sync
async def test_every_turn_is_offered_the_tools():
    """The model is stateless: the menu has to be re-sent every call."""
    gateway = ScriptedGateway(wants("get_section", {"section_number": "80D"}), answer())

    await run_agent(
        question="80D?",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
    )

    assert all(sent == list(SCHEMAS) for sent in gateway.tools_seen)


# --------------------------------------------------------------------------
# exit 2 — the model declares failure
# --------------------------------------------------------------------------


@sync
async def test_cannot_answer_stops_the_loop_with_a_reason():
    gateway = ScriptedGateway(
        wants("cannot_answer", {"reason": "No provision covers crypto staking."}),
        answer("should never be reached"),
    )
    recorder = Recorder()

    run = await run_agent(
        question="How is crypto staking taxed?",
        retriever=StubRetriever([]),
        gateway=gateway,
        progress=recorder,
    )

    assert run.outcome == "gave_up"
    assert "crypto staking" in run.give_up_reason
    assert run.text == ""
    assert run.finish_reason == "gave_up"
    assert "gave_up" in recorder.names()


# --------------------------------------------------------------------------
# exit 3 — the backstop
# --------------------------------------------------------------------------


@sync
async def test_a_model_that_never_answers_is_stopped_by_the_step_budget():
    """The case `while True` would not survive.

    Here the model asks for the same tool forever. Nothing in its own
    behaviour ends the request; only the bound does.
    """
    gateway = ScriptedGateway(wants("search_provisions", {"query": "again"}))

    run = await run_agent(
        question="anything",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
        max_steps=3,
    )

    assert run.outcome == "budget_exhausted"
    assert run.steps == 3
    assert len(gateway.conversations) == 3


@sync
async def test_the_token_budget_also_stops_it():
    gateway = ScriptedGateway(wants("search_provisions", {"query": "again"}))

    run = await run_agent(
        question="anything",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
        max_steps=50,
        token_budget=300,  # 110 tokens a turn, so it stops on turn 3
    )

    assert run.outcome == "budget_exhausted"
    assert run.steps < 50


# --------------------------------------------------------------------------
# misbehaviour that must not crash the request
# --------------------------------------------------------------------------


@sync
async def test_an_identical_repeated_call_is_counted_not_re_run():
    gateway = ScriptedGateway(
        wants("get_section", {"section_number": "80D"}, call_id="a"),
        wants("get_section", {"section_number": "80D"}, call_id="b"),
        answer(),
    )
    retriever = StubRetriever([passage()])

    run = await run_agent(
        question="80D?",
        retriever=retriever,
        gateway=gateway,
        progress=Recorder(),
    )

    assert run.redundant_calls == 1
    # The retriever ran once, not twice - the repeat was served from cache.
    assert len(retriever.calls) == 1


@sync
async def test_an_unknown_tool_name_comes_back_as_data():
    """The model misremembered. That is recoverable, so it must not raise."""
    gateway = ScriptedGateway(wants("summarise_everything", {"x": 1}), answer())

    run = await run_agent(
        question="anything",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
    )

    assert run.outcome == "answered"
    tool_message = gateway.conversations[1][-1]
    assert "No tool named" in tool_message["content"]
    assert "search_provisions" in tool_message["content"]


@sync
async def test_malformed_arguments_come_back_as_data():
    broken = Generation(
        text="",
        model="stub",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        total_tokens=15,
        finish_reason="tool_calls",
        tool_calls=(ToolCall(id="c1", name="get_section", arguments="{not json"),),
    )
    gateway = ScriptedGateway(broken, answer())

    run = await run_agent(
        question="anything",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
    )

    assert run.outcome == "answered"
    assert "not valid JSON" in gateway.conversations[1][-1]["content"]


@sync
async def test_wrong_arguments_come_back_as_data():
    gateway = ScriptedGateway(wants("get_section", {"wrong_kwarg": "80D"}), answer())

    run = await run_agent(
        question="anything",
        retriever=StubRetriever([passage()]),
        gateway=gateway,
        progress=Recorder(),
    )

    assert run.outcome == "answered"
    assert "Bad arguments" in gateway.conversations[1][-1]["content"]


# --------------------------------------------------------------------------
# the tools themselves
# --------------------------------------------------------------------------


@sync
async def test_get_section_rejects_a_near_miss():
    """Similarity search will happily return s.80C when asked for s.80D."""
    box = ToolBox(
        retriever=StubRetriever([passage(section="80C")]),
        top_k=6,
        progress=Recorder(),
    )

    result = await box.run("get_section", {"section_number": "80D"})

    assert result["found"] is False
    assert "No section 80D" in result["note"]


@sync
async def test_map_section_returns_the_counterpart_when_the_corpus_knows_it():
    box = ToolBox(
        retriever=StubRetriever([passage(section="19", act="ITA-2025", maps_to="16")]),
        top_k=6,
        progress=Recorder(),
    )

    result = await box.run("map_section", {"section_number": "19", "act": "ITA-2025"})

    assert result["found"] is True
    assert result["to"] == {"act": "ITA-1961", "section": "16"}


@sync
async def test_map_section_tells_the_model_not_to_guess():
    """maps_to_1961 is populated for a handful of sections, so this is the
    common case - and the note is what stops it inventing a number."""
    box = ToolBox(
        retriever=StubRetriever([passage(section="19", act="ITA-2025")]),
        top_k=6,
        progress=Recorder(),
    )

    result = await box.run("map_section", {"section_number": "19"})

    assert result["found"] is False
    assert "not guess" in result["note"]


@sync
async def test_act_filter_keeps_only_that_act():
    box = ToolBox(
        retriever=StubRetriever(
            [passage(act="ITA-1961"), passage(section="19", act="ITA-2025")]
        ),
        top_k=6,
        progress=Recorder(),
    )

    result = await box.run("search_provisions", {"query": "salary", "act": "ITA-2025"})

    assert [p["act"] for p in result["provisions"]] == ["ITA-2025"]


@sync
async def test_the_toolbox_remembers_across_tools_without_duplicating():
    retriever = StubRetriever([passage(), passage()])
    box = ToolBox(retriever=retriever, top_k=6, progress=Recorder())

    await box.run("search_provisions", {"query": "one"})
    await box.run("search_provisions", {"query": "two"})

    # Same provision both times, so the citation set holds it once.
    assert len(box.seen) == 1


def test_every_schema_says_when_not_to_use_it_or_what_it_returns():
    """Descriptions are the prompt. A bare label is how tool choice fails."""
    for schema in SCHEMAS:
        description = schema["function"]["description"]

        assert len(description) > 80, schema["function"]["name"]
        assert any(
            hint in description
            for hint in ("Do NOT", "Use this", "Use whenever", "Call this", "Returns")
        ), schema["function"]["name"]


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s["function"]["name"])
def test_every_schema_is_shaped_the_way_the_api_expects(schema):
    assert schema["type"] == "function"
    function = schema["function"]

    assert set(function) == {"name", "description", "parameters"}
    assert function["parameters"]["type"] == "object"
    # Required arguments must actually be declared as properties.
    for name in function["parameters"].get("required", []):
        assert name in function["parameters"]["properties"]
