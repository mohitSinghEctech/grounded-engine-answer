import logging
from collections.abc import Sequence
from typing import Any

import httpx

from app.config import Settings
from app.errors import (
    GatewayRateLimited,
    GatewayTimeout,
    GatewayUnavailable,
    InvalidGatewayResponse,
)
from app.services.base import Generation, ToolCall

logger = logging.getLogger(__name__)


class HttpLLMGateway:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def generate(
        self,
        prompt: str | None = None,
        max_tokens: int = 2000,
        *,
        messages: Sequence[dict[str, Any]] | None = None,
        tools: Sequence[dict[str, Any]] = (),
    ) -> Generation:
        # The gateway rejects both-or-neither, so send exactly one.
        body: dict[str, Any] = {"max_tokens": max_tokens}

        if messages is not None:
            body["messages"] = list(messages)
        else:
            body["prompt"] = prompt

        if tools:
            body["tools"] = list(tools)

        try:
            response = await self.client.post("/generate", json=body)

        except httpx.TimeoutException as exc:
            logger.error("Gateway request failed | error=GATEWAY_TIMEOUT")
            raise GatewayTimeout() from exc

        except httpx.RequestError as exc:
            logger.error(
                "Gateway request failed | error=GATEWAY_UNAVAILABLE | reason=%s",
                type(exc).__name__,
            )
            raise GatewayUnavailable() from exc

        if response.status_code >= 400:
            self._raise_for_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidGatewayResponse("Gateway response was not JSON.") from exc

        try:
            return Generation(
                text=payload["answer"],
                model=payload["model"],
                prompt_tokens=payload["prompt_tokens"],
                completion_tokens=payload["completion_tokens"],
                reasoning_tokens=payload.get("reasoning_tokens", 0),
                total_tokens=payload["total_tokens"],
                finish_reason=payload["finish_reason"],
                tool_calls=tuple(
                    ToolCall(
                        id=call["id"],
                        name=call["name"],
                        arguments=call["arguments"],
                    )
                    # .get, not [] - an older gateway has no such key, and
                    # "no tools requested" is the correct reading of that.
                    for call in payload.get("tool_calls", [])
                ),
            )
        except KeyError as exc:
            raise InvalidGatewayResponse(
                f"Gateway response missing field: {exc.args[0]}"
            ) from exc

    def _raise_for_error(self, response: httpx.Response) -> None:
        gateway_code = "UNKNOWN"

        try:
            gateway_code = response.json().get("error_code", "UNKNOWN")
        except ValueError:
            pass

        logger.error(
            "Gateway returned an error | status_code=%s | gateway_error_code=%s",
            response.status_code,
            gateway_code,
        )

        if response.status_code == 429:
            retry_after = None
            header = response.headers.get("retry-after")

            if header:
                try:
                    retry_after = float(header)
                except ValueError:
                    retry_after = None

            raise GatewayRateLimited(retry_after_seconds=retry_after)

        if response.status_code == 504:
            raise GatewayTimeout(
                f"The LLM gateway timed out upstream ({gateway_code})."
            )

        if response.status_code >= 500:
            raise GatewayUnavailable(
                f"The LLM gateway reported an upstream failure ({gateway_code})."
            )

        raise InvalidGatewayResponse(
            f"The LLM gateway rejected the request ({gateway_code})."
        )
