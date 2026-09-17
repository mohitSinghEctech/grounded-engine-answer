"""The status stream: its frames, its ordering, and what it must never send.

Two properties matter most, and both are easy to break by accident:

  * `/ask` and `/ask/stream` must stay identical, because they are the same
    pipeline and a UI switching between them should see the same answer.
  * the stream must never carry answer text before citations are checked.
    That is the whole reason token streaming was left out, so it is worth a
    test rather than a comment.
"""

import json

import pytest

from app.progress import STEPS, Progress, Step


def parse(body: str) -> list[tuple[str, dict]]:
    """Server-Sent Events text into (event name, payload) pairs."""
    frames = []
    event = "message"

    for line in body.splitlines():
        if line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            frames.append((event, json.loads(line[len("data: ") :])))

    return frames


# --- the emitter itself -------------------------------------------------


@pytest.mark.anyio
async def test_a_discarded_emitter_accepts_everything_and_keeps_nothing():
    """So the pipeline never has to ask whether anyone is listening."""
    progress = Progress.discarded()

    assert not progress.listening

    await progress.emit("embedding", model="stub")
    await progress.finish()


def test_a_step_serialises_to_one_sse_frame():
    frame = Step("retrieved", {"count": 6}).to_sse()

    assert frame == 'event: status\ndata: {"step": "retrieved", "count": 6}\n\n'


def test_the_event_name_can_be_overridden():
    assert Step("done", {}).to_sse(event="result").startswith("event: result\n")


# --- the stream ---------------------------------------------------------


def test_the_stream_reports_every_stage_in_order(client):
    response = client.post(
        "/ask/stream",
        json={"question": "health insurance premium deduction", "max_tokens": 200},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = parse(response.text)
    steps = [payload["step"] for event, payload in frames if event == "status"]

    # Each stage present, and in the order STEPS declares.
    for stage in ("received", "embedding", "retrieved", "generating", "verified"):
        assert stage in steps, stage

    positions = [STEPS.index(s) for s in steps if s in STEPS]
    assert positions == sorted(positions), steps


def test_the_stream_ends_with_the_same_body_ask_returns(client):
    question = {"question": "health insurance premium deduction", "max_tokens": 200}

    plain = client.post("/ask", json=question).json()
    frames = parse(client.post("/ask/stream", json=question).text)

    results = [payload for event, payload in frames if event == "result"]

    assert len(results) == 1

    streamed = results[0]

    # Timings differ between two runs; everything that describes the answer
    # must not.
    for field in ("answer", "refused", "refusal_reason", "citations", "retrieved"):
        assert streamed[field] == plain[field], field


def test_no_answer_text_appears_before_verification(client):
    """The reason token streaming was left out, enforced.

    A fabricated citation is only detectable once the answer is complete,
    so no frame before `verified` may contain the answer.
    """
    response = client.post(
        "/ask/stream",
        json={"question": "health insurance premium deduction", "max_tokens": 200},
    )

    frames = parse(response.text)
    verified_at = next(
        index
        for index, (event, payload) in enumerate(frames)
        if payload.get("step") == "verified"
    )

    answer = next(payload["answer"] for event, payload in frames if event == "result")

    for event, payload in frames[:verified_at]:
        assert answer not in json.dumps(payload), payload["step"]


def test_the_verified_frame_separates_real_citations_from_invented_ones(
    client, gateway
):
    gateway.text = (
        "Real (ITA-1961 s.80D) and invented (ITA-1961 s.99ZZ) and "
        "(DEPT-GUIDANCE s.nope-ay9)."
    )

    frames = parse(
        client.post(
            "/ask/stream",
            json={"question": "health insurance premium", "max_tokens": 200},
        ).text
    )

    verified = next(p for _, p in frames if p.get("step") == "verified")

    assert verified["cited"] == ["ITA-1961 s.80D"]
    assert verified["invented"] == [
        "DEPT-GUIDANCE s.nope-ay9",
        "ITA-1961 s.99ZZ",
    ]


def test_a_structural_refusal_says_the_model_was_never_called(client, gateway):
    """Nothing retrieved means nothing to ground on, so no model call."""
    frames = parse(
        client.post(
            "/ask/stream",
            json={"question": "qqqq zzzz wwww", "max_tokens": 200},
        ).text
    )

    refused = next(p for _, p in frames if p.get("step") == "refused")

    assert refused["reason"] == "nothing_retrieved"
    assert refused["model_called"] is False
    assert gateway.calls == 0


def test_a_named_section_is_announced_before_it_is_fetched(client):
    frames = parse(
        client.post(
            "/ask/stream",
            json={"question": "what does section 234A say?", "max_tokens": 200},
        ).text
    )

    looking = next(p for _, p in frames if p.get("step") == "looking_up")

    assert looking["sections"] == ["234A"]


# --- CORS, without which no browser page can call this service ----------


def test_cors_is_off_unless_asked_for(client):
    """An API with no browser client should not advertise itself to one."""
    response = client.options(
        "/ask/stream",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_a_configured_origin(cors_client):
    """Origin: null is what a page opened from a file:// URL sends."""
    response = cors_client.options(
        "/ask/stream",
        headers={
            "Origin": "null",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.headers.get("access-control-allow-origin") == "*"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_a_detail_key_cannot_overwrite_the_step_name():
    """It did once: the agent's `planning` step sent step=1, and the frame
    arrived with a 1 where the step name belonged, so every reader showed
    a number instead of a phrase."""
    import json

    from app.progress import Step

    frame = Step("planning", {"step": 1, "of": 6}).to_sse()
    payload = json.loads(frame.split("data: ")[1])

    assert payload["step"] == "planning"
