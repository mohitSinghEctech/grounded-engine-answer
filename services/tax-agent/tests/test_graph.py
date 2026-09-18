"""The graph, and the two branches only it can take.

Two things are pinned here. First, parity: with both branch flags off the
graph must answer exactly as the straight line does, because both call the
same stages in `app/pipeline/steps.py`. Parity is what lets an eval run
attribute a score change to a branch rather than to the rewrite.

Second, the branches: each fires only in the case it was built for, and
each is bounded, so a model that misbehaves every time cannot make the
graph loop.
"""

import asyncio
import functools
from pathlib import Path

import pytest

from app.config import Settings
from app.graph.build import run_graph_pipeline
from app.pipeline import steps
from app.progress import Progress, Step
from app.routers.ask import run_linear_pipeline
from app.schemas import AskRequest
from app.services.base import Generation, Passage


def sync(async_test):
    """Run an async test function without a pytest plugin.

    The rest of this suite is synchronous and the project's test
    dependencies are just pytest, so rather than add pytest-asyncio for
    one module, each async test gets its own event loop here.
    `functools.wraps` keeps the signature pytest reads, so fixtures and
    `parametrize` still work.
    """

    @functools.wraps(async_test)
    def wrapper(*args, **kwargs):
        return asyncio.run(async_test(*args, **kwargs))

    return wrapper


def make_settings(**overrides) -> Settings:
    return Settings(
        llm_gateway_url="http://gateway.invalid",
        embedding_api_key="test-key",
        corpus_date="2026-09-14",
        **overrides,
    )


def passage(section_number: str = "80D", score: float = 0.9) -> Passage:
    return Passage(
        text=f"text of {section_number}",
        score=score,
        act="ITA-1961",
        section_number=section_number,
        section_title="Deduction in respect of health insurance premia",
        page_start=12,
        corpus_date="2026-09-14",
    )


class ScriptedRetriever:
    """Returns a prepared result per call, and records how it was asked.

    Two results means "the filtered search found nothing, the unfiltered
    one did" can be set up exactly, which is the only situation the widen
    branch exists for.
    """

    def __init__(self, *results: list[Passage]):
        self.results = list(results)
        self.calls: list[int | None] = []

    async def search(self, question, top_k, tax_year=None, progress=None):
        self.calls.append(tax_year)

        if not self.results:
            return []

        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


class ScriptedGateway:
    """Answers with the next scripted reply, and keeps what it was sent.

    A reply is either a string (wrapped as plain prose) or a whole
    `Generation`, which is how a test scripts a tool call.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts: list[str | None] = []
        self.conversations: list[list[dict]] = []

    async def generate(
        self, prompt=None, max_tokens=2000, *, messages=None, tools=()
    ) -> Generation:
        self.prompts.append(prompt)

        if messages is not None:
            self.conversations.append(list(messages))

        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

        if isinstance(reply, Generation):
            return reply

        return Generation(
            text=reply,
            model="stub-model",
            prompt_tokens=100,
            completion_tokens=20,
            reasoning_tokens=0,
            total_tokens=120,
            finish_reason="stop",
        )


class Recorder(Progress):
    """A Progress that keeps the steps instead of queueing them."""

    def __init__(self):
        super().__init__(queue=None)
        self.steps: list[Step] = []

    async def emit(self, name: str, **detail) -> None:
        self.steps.append(Step(name=name, detail=detail))

    def names(self) -> list[str]:
        return [step.name for step in self.steps]


# --------------------------------------------------------------------------
# parity
# --------------------------------------------------------------------------

#: Every field that describes the outcome. Timings are excluded: they differ
#: run to run by microseconds and say nothing about behaviour.
OUTCOME_FIELDS = (
    "answer",
    "refused",
    "refusal_reason",
    "citations",
    "unsupported_citations",
    "corpus_date",
    "tax_year",
    "tax_year_source",
    "retrieved",
    "model",
    "finish_reason",
)


@sync
@pytest.mark.parametrize(
    "answer_text",
    [
        pytest.param("Premium deduction (ITA-1961 s.80D).", id="grounded"),
        pytest.param("REFUSE: OUT_OF_SCOPE I cannot advise.", id="out_of_scope"),
        pytest.param("Something with no citation at all.", id="not_grounded"),
        pytest.param("See ITA-1961 s.99Z.", id="fabricated"),
    ],
)
async def test_graph_matches_linear_outcome(answer_text):
    """Same question, same fakes, same answer - whichever orchestrator runs."""
    settings = make_settings(pipeline="graph")

    linear = await run_linear_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        ScriptedGateway(answer_text),
        settings,
    )
    graph = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        ScriptedGateway(answer_text),
        settings,
    )

    for field in OUTCOME_FIELDS:
        assert getattr(graph, field) == getattr(linear, field), field


@sync
async def test_graph_matches_linear_when_nothing_retrieved():
    settings = make_settings()

    linear = await run_linear_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([]),
        ScriptedGateway("x"),
        settings,
    )
    graph = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([]),
        ScriptedGateway("x"),
        settings,
    )

    assert graph.refusal_reason == linear.refusal_reason == "nothing_retrieved"

    for field in OUTCOME_FIELDS:
        assert getattr(graph, field) == getattr(linear, field), field


@sync
async def test_graph_never_calls_the_model_with_nothing_retrieved():
    """The structural guarantee has to survive the rewrite."""
    gateway = ScriptedGateway("should never be asked")

    response = await run_graph_pipeline(
        AskRequest(question="80D?"), ScriptedRetriever([]), gateway, make_settings()
    )

    assert gateway.prompts == []
    assert response.refused


@sync
async def test_graph_emits_the_same_steps_as_linear():
    """The UI and eval/watch.py read these names, so they are a contract."""
    settings = make_settings()

    linear_steps = Recorder()
    await run_linear_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        ScriptedGateway("Premium (ITA-1961 s.80D)."),
        settings,
        progress=linear_steps,
    )

    graph_steps = Recorder()
    await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        ScriptedGateway("Premium (ITA-1961 s.80D)."),
        settings,
        progress=graph_steps,
    )

    assert graph_steps.names() == linear_steps.names()


# --------------------------------------------------------------------------
# branch: widen on thin retrieval
# --------------------------------------------------------------------------


@sync
async def test_widen_searches_again_without_the_year_filter():
    retriever = ScriptedRetriever([], [passage()])
    gateway = ScriptedGateway("Premium (ITA-1961 s.80D).")
    recorder = Recorder()

    response = await run_graph_pipeline(
        AskRequest(question="80D?", tax_year=2024),
        retriever,
        gateway,
        make_settings(graph_widen_on_thin_retrieval=True),
        progress=recorder,
    )

    # Asked with the filter, then without it.
    assert retriever.calls == [2024, None]
    assert "widening" in recorder.names()
    assert not response.refused
    assert response.refusal_reason == "none"


@sync
async def test_widen_refuses_when_the_wider_search_is_also_empty():
    """One extra search, not a loop."""
    retriever = ScriptedRetriever([], [])

    response = await run_graph_pipeline(
        AskRequest(question="80D?", tax_year=2024),
        retriever,
        ScriptedGateway("never asked"),
        make_settings(graph_widen_on_thin_retrieval=True),
    )

    assert retriever.calls == [2024, None]
    assert response.refusal_reason == "nothing_retrieved"


@sync
async def test_widen_does_not_fire_without_a_year_to_drop():
    """No filter was applied, so there is nothing for widening to relax."""
    retriever = ScriptedRetriever([])

    await run_graph_pipeline(
        AskRequest(question="80D?"),
        retriever,
        ScriptedGateway("never asked"),
        make_settings(graph_widen_on_thin_retrieval=True),
    )

    assert retriever.calls == [None]


@sync
async def test_widen_is_off_by_default():
    retriever = ScriptedRetriever([])

    await run_graph_pipeline(
        AskRequest(question="80D?", tax_year=2024),
        retriever,
        ScriptedGateway("never asked"),
        make_settings(),
    )

    assert retriever.calls == [2024]


# --------------------------------------------------------------------------
# branch: retry on fabrication
# --------------------------------------------------------------------------


@sync
async def test_retry_asks_again_naming_the_invented_citation():
    gateway = ScriptedGateway(
        "See ITA-1961 s.99Z.",  # invented
        "Premium (ITA-1961 s.80D).",  # corrected
    )
    recorder = Recorder()

    response = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_retry_on_fabrication=True),
        progress=recorder,
    )

    assert len(gateway.prompts) == 2
    # The correction has to name the specific reference, not just repeat
    # the rule the first attempt already had.
    assert "ITA-1961 s.99Z" in gateway.prompts[1]
    assert "retrying" in recorder.names()

    assert response.refusal_reason == "none"
    assert response.unsupported_citations == []
    assert [c.section_number for c in response.citations] == ["80D"]


@sync
async def test_retry_gives_up_after_one_extra_attempt():
    """A model that fabricates every time must not loop."""
    gateway = ScriptedGateway("See ITA-1961 s.99Z.")

    response = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_retry_on_fabrication=True),
    )

    assert len(gateway.prompts) == 2
    assert response.refusal_reason == "not_grounded"
    assert response.unsupported_citations == ["ITA-1961 s.99Z"]


@sync
async def test_retry_fires_on_a_mixed_answer():
    """Real citations plus one invented - the shape the first rule missed.

    This test is the inverse of the one it replaces. The old rule held
    that a partly fabricated answer is still grounded and re-asking risks
    the good part, so it did nothing; the eval then showed that ALL
    twelve fabrications are this shape, so the branch never ran at all.
    Nothing is left to trade off against zero coverage.
    """
    gateway = ScriptedGateway(
        "Premium (ITA-1961 s.80D), see also ITA-1961 s.99Z.",  # mixed
        "Premium (ITA-1961 s.80D).",  # corrected
    )

    response = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_retry_on_fabrication=True),
    )

    assert len(gateway.prompts) == 2
    assert "ITA-1961 s.99Z" in gateway.prompts[1]
    assert response.unsupported_citations == []
    assert [c.section_number for c in response.citations] == ["80D"]


@sync
async def test_mixed_answer_stands_when_the_retry_does_not_fix_it():
    """The named risk of the wider rule, pinned down.

    A retry that fabricates again must not cost the answer its real
    citation: the second attempt is accepted as grounded, with the
    invented reference still reported to the caller rather than hidden.
    """
    gateway = ScriptedGateway("Premium (ITA-1961 s.80D), see also ITA-1961 s.99Z.")

    response = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_retry_on_fabrication=True),
    )

    assert len(gateway.prompts) == 2
    assert response.refusal_reason == "none"
    assert response.unsupported_citations == ["ITA-1961 s.99Z"]
    assert [c.section_number for c in response.citations] == ["80D"]


@sync
async def test_retry_does_not_fire_on_an_uncited_answer():
    """Nothing cited is a retrieval failure; a corrected prompt cannot fix it."""
    gateway = ScriptedGateway("No citation here.")

    response = await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_retry_on_fabrication=True),
    )

    assert len(gateway.prompts) == 1
    assert response.refusal_reason == "not_grounded"


@sync
async def test_retry_is_off_by_default():
    gateway = ScriptedGateway("See ITA-1961 s.99Z.")

    await run_graph_pipeline(
        AskRequest(question="80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(),
    )

    assert len(gateway.prompts) == 1


# --------------------------------------------------------------------------
# the retry note itself
# --------------------------------------------------------------------------


def test_prompt_is_unchanged_without_a_note():
    """A first attempt must be byte-identical to the pre-graph prompt."""
    from app.prompt import build_prompt

    assert build_prompt("q", [passage()]) == build_prompt("q", [passage()], note=None)


def test_note_lands_before_the_answer_cue():
    from app.prompt import build_prompt

    prompt = build_prompt("q", [passage()], note="CORRECTION")

    assert prompt.index("CORRECTION") < prompt.index("Answer:")
    assert prompt.endswith("Answer:")


def test_has_fabrication_covers_the_mixed_case_that_fabricated_only_missed():
    both = steps.Verification(
        declared=None,
        answer_text="x",
        cited=[passage()],
        unsupported={("ITA-1961", "99Z")},
    )
    invented_only = steps.Verification(
        declared=None, answer_text="x", cited=[], unsupported={("ITA-1961", "99Z")}
    )
    nothing = steps.Verification(
        declared=None, answer_text="x", cited=[], unsupported=set()
    )

    # The old rule: only an all-invented answer. Zero of forty questions.
    assert not both.fabricated_only
    assert invented_only.fabricated_only
    assert not nothing.fabricated_only

    # The rule in force: any invented reference, mixed or not.
    assert both.has_fabrication
    assert invented_only.has_fabrication
    assert not nothing.has_fabrication


# --------------------------------------------------------------------------
# entry branch: pipeline or agent
# --------------------------------------------------------------------------


def tool_reply(name, arguments, call_id="c1"):
    """A model reply that asks for a tool instead of answering."""
    import json as _json

    from app.services.base import ToolCall

    return Generation(
        text="",
        model="stub-model",
        prompt_tokens=100,
        completion_tokens=10,
        reasoning_tokens=0,
        total_tokens=110,
        finish_reason="tool_calls",
        tool_calls=(ToolCall(id=call_id, name=name, arguments=_json.dumps(arguments)),),
    )


@sync
async def test_the_agent_is_off_by_default():
    """A behaviour change has to earn its place in its own eval run."""
    gateway = ScriptedGateway("Premium (ITA-1961 s.80D).")

    response = await run_graph_pipeline(
        AskRequest(question="What changed for salary income between the Acts?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(),
    )

    # One plain prompt call, and no conversation: the pipeline path.
    assert len(gateway.prompts) == 1
    assert gateway.conversations == []
    assert response.tools_called == []
    assert response.agent_steps is None


@sync
async def test_a_comparison_question_takes_the_agent_path():
    gateway = ScriptedGateway(
        tool_reply("search_provisions", {"query": "salary income"}),
        "Salary is charged under (ITA-1961 s.80D).",
    )
    recorder = Recorder()

    response = await run_graph_pipeline(
        AskRequest(question="What changed for salary income between the Acts?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_agent_on_comparison=True),
        progress=recorder,
    )

    assert response.tools_called == ["search_provisions"]
    assert response.agent_steps == 2
    assert "planning" in recorder.names()
    # It rejoined the shared path: the citation was verified, not trusted.
    assert [c.section_number for c in response.citations] == ["80D"]


@sync
async def test_a_simple_question_still_takes_the_cheap_path():
    """Agency on a one-hop question costs ~3x the tokens for no benefit."""
    gateway = ScriptedGateway("Premium (ITA-1961 s.80D).")

    response = await run_graph_pipeline(
        AskRequest(question="What is the deduction limit under section 80D?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_agent_on_comparison=True),
    )

    assert response.tools_called == []
    assert response.refusal_reason == "none"


@sync
async def test_cannot_answer_becomes_a_declared_not_in_corpus_refusal():
    """The honest version of the marker that failed as a prompt rule.

    Inferred from an absent citation it was a guess; declared through a
    tool call it is a fact, and it carries the model's own reason.
    """
    gateway = ScriptedGateway(
        tool_reply("cannot_answer", {"reason": "No provision covers this."}),
    )

    response = await run_graph_pipeline(
        AskRequest(question="What changed for crypto between the 1961 and 2025 Acts?"),
        ScriptedRetriever([]),
        gateway,
        make_settings(graph_agent_on_comparison=True),
    )

    assert response.refused
    assert response.refusal_reason == "not_in_corpus"
    assert "No provision covers this." in response.answer
    assert response.citations == []


@sync
async def test_an_exhausted_budget_is_its_own_refusal_reason():
    """Not the model's verdict on the corpus - our backstop firing."""
    gateway = ScriptedGateway(tool_reply("search_provisions", {"query": "x"}))

    response = await run_graph_pipeline(
        AskRequest(question="What is the difference between the Acts?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_agent_on_comparison=True, agent_max_steps=2),
    )

    assert response.refusal_reason == "budget_exhausted"
    assert response.agent_steps == 2


@sync
async def test_an_uncited_agent_answer_is_still_caught():
    """Agency is not a side door around citation verification."""
    gateway = ScriptedGateway(
        tool_reply("search_provisions", {"query": "salary"}),
        "Salary is taxed, broadly speaking.",  # no citation at all
    )

    response = await run_graph_pipeline(
        AskRequest(question="What changed for salary between the Acts?"),
        ScriptedRetriever([passage()]),
        gateway,
        make_settings(graph_agent_on_comparison=True),
    )

    assert response.refusal_reason == "not_grounded"


def test_the_router_agrees_with_the_eval_set():
    """The invariant, read off the file rather than copied out of it.

    A question carries `expected_tools` if and only if the router sends it
    to the agent. Either half breaking is a real defect: a question with a
    declared path that never reaches the agent scores agent_used=False
    forever, and a question that reaches the agent without one is running
    at triple cost with its trajectory unscored.

    This reads eval/questions.yaml directly, because the test below it
    copies the phrasings in, and a copy is exactly what went stale the
    first time. Skipped inside the service container, which ships without
    the eval set.
    """
    import yaml

    from app.pipeline.steps import wants_agent

    path = Path(__file__).resolve().parents[3] / "eval" / "questions.yaml"

    if not path.exists():
        pytest.skip("eval set not present (running inside the service image)")

    questions = yaml.safe_load(path.read_text())["questions"]

    declared = {q["id"] for q in questions if q.get("expected_tools")}
    routed = {q["id"] for q in questions if wants_agent(q["question"])}

    assert declared == routed, (
        f"declared but not routed: {sorted(declared - routed)}; "
        f"routed but not declared: {sorted(routed - declared)}"
    )
    # Not an empty-set tautology: the branch exists for these four.
    assert len(routed) == 4


def test_the_router_fires_on_the_real_eval_questions():
    """Pinned against eval/questions.yaml, not against invented phrasings.

    The first version of this router was written from imagination and
    matched none of these four - they say "corresponds to", "replaces",
    "previously" and "was section X", where the keyword list had
    "replaced". Zero of forty questions took the path the branch existed
    for, and every test passed, because the tests were invented from the
    same imagination as the router.
    """
    from app.pipeline.steps import wants_agent

    # The four that name a section in one Act and ask for its counterpart.
    for question in (
        "Section 80C of the 1961 Act corresponds to which section of the 2025 Act?",
        "Which provision of the 2025 Act replaces section 80D on health "
        "insurance premium?",
        "Where is the deduction for rent paid found in the 2025 Act, "
        "previously section 80GG?",
        "Interest for default in furnishing a return was section 234A in the 1961 Act. "
        "What is it in the 2025 Act?",
        # and the phrasings the first version did catch
        "What changed for house property income in the 2025 Act?",
        "What is the difference between the two Acts on salary?",
    ):
        assert wants_agent(question), question

    # One-hop questions must stay on the cheap path: agency costs ~3x.
    for question in (
        "What is the deduction limit under section 80D?",
        "Is interest under section 234A charged monthly?",
        "Which ITR form does a salaried individual file?",
        "What is section 139 about?",
        "Which section of the 2025 Act covers deduction for contributions to the "
        "pension scheme of the Central Government?",
    ):
        assert not wants_agent(question), question
