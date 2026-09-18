"""Tool calling, at the HTTP edge.

Three things are worth pinning: the menu reaches the provider, the calls
come back out, and an answer-less reply is no longer an error - which was
a deliberate loosening of a check that used to guard against a broken
upstream, so it needs a test on both sides of the line.
"""

import pytest

from app.services.base import LLMResult, ToolCall

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_provisions",
        "description": "Search Indian income-tax provisions by meaning.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def result(text="ok", tool_calls=()):
    return LLMResult(
        text=text,
        model="stub-model",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=0,
        total_tokens=15,
        finish_reason="tool_calls" if tool_calls else "stop",
        tool_calls=tool_calls,
    )


def test_messages_and_tools_reach_the_client(client, fake_llm_client):
    """The gateway is a pass-through here; it must not reshape either."""
    fake_llm_client.result = result()

    conversation = [
        {"role": "system", "content": "Answer from tools only."},
        {"role": "user", "content": "What is the 80D limit?"},
    ]

    response = client.post(
        "/generate",
        json={"messages": conversation, "tools": [SEARCH_TOOL], "max_tokens": 400},
    )

    assert response.status_code == 200
    sent = fake_llm_client.seen[-1]
    assert sent["messages"] == conversation
    assert sent["tools"] == [SEARCH_TOOL]
    assert sent["prompt"] is None


def test_tool_calls_are_returned_to_the_caller(client, fake_llm_client):
    fake_llm_client.result = result(
        text="",
        tool_calls=(
            ToolCall(
                id="call_1",
                name="search_provisions",
                arguments='{"query": "health insurance premium deduction"}',
            ),
        ),
    )

    body = client.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": "80D?"}],
            "tools": [SEARCH_TOOL],
        },
    ).json()

    assert body["answer"] == ""
    assert body["tool_calls"] == [
        {
            "id": "call_1",
            "name": "search_provisions",
            # A raw string, not a parsed object: the gateway does not
            # interpret arguments it cannot validate.
            "arguments": '{"query": "health insurance premium deduction"}',
        }
    ]
    assert body["finish_reason"] == "tool_calls"


def test_a_plain_prompt_still_works_and_sends_no_tools(client, fake_llm_client):
    """The single-shot path is the one every existing caller uses."""
    fake_llm_client.result = result(text="Section 80D allows...")

    body = client.post("/generate", json={"prompt": "What is 80D?"}).json()

    assert body["answer"] == "Section 80D allows..."
    assert body["tool_calls"] == []
    assert fake_llm_client.seen[-1]["tools"] == []
    assert fake_llm_client.seen[-1]["messages"] is None


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="neither"),
        pytest.param(
            {"prompt": "hi", "messages": [{"role": "user", "content": "hi"}]},
            id="both",
        ),
    ],
)
def test_prompt_and_messages_are_mutually_exclusive(client, payload):
    """Otherwise there is a question about which one the model actually saw."""
    assert client.post("/generate", json=payload).status_code == 422
