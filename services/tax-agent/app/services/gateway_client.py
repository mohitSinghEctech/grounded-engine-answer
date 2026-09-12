import logging

import httpx

from app.config import Settings
from app.errors import (
    GatewayRateLimited,
    GatewayTimeout,
    GatewayUnavailable,
    InvalidGatewayResponse,
)
from app.services.base import Generation

logger = logging.getLogger(__name__)


class HttpLLMGateway:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings

    async def generate(self, prompt: str, max_tokens: int) -> Generation:
        try:
            response = await self.client.post(
                "/ask",
                json={"question": prompt, "max_tokens": max_tokens},
            )

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
                total_tokens=payload["total_tokens"],
                finish_reason=payload["finish_reason"],
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