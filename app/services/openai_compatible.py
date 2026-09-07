import asyncio
import logging
import time

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.config import Settings
from app.errors import (
    InvalidUpstreamResponse,
    UpstreamRateLimited,
    UpstreamTimeout,
    UpstreamUnavailable,
)
from app.retry import calculate_retry_delay
from app.services.base import LLMResult

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(self, client: AsyncOpenAI, settings: Settings):
        self.client = client
        self.settings = settings

    async def _generate_once(self, prompt: str, max_tokens: int) -> LLMResult:

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                reasoning_effort=self.settings.reasoning_effort,
            )

        except APITimeoutError as exc:
            logger.error("LLM request failed | error=UPSTREAM_TIMEOUT")
            raise UpstreamTimeout() from exc

        except RateLimitError as exc:
            retry_after = None

            if exc.response is not None:
                retry_after_header = exc.response.headers.get("retry-after")

                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)

                        if retry_after < 0:
                            retry_after = None

                    except ValueError:
                        retry_after = None

            logger.error(
                "LLM request failed | "
                "error=UPSTREAM_RATE_LIMITED | "
                "retry_after_seconds=%s",
                retry_after,
            )

            raise UpstreamRateLimited(
                retry_after_seconds=retry_after,
            ) from exc

        except APIConnectionError as exc:
            logger.error("LLM request failed | error=UPSTREAM_UNAVAILABLE")
            raise UpstreamUnavailable() from exc

        except APIStatusError as exc:
            if 500 <= exc.status_code < 600:
                logger.error(
                    "LLM request failed | error=UPSTREAM_UNAVAILABLE | status_code=%s",
                    exc.status_code,
                )
                raise UpstreamUnavailable() from exc

            logger.error(
                "LLM request failed | error=INVALID_UPSTREAM_RESPONSE | status_code=%s",
                exc.status_code,
            )
            raise InvalidUpstreamResponse() from exc

        if not response.choices:
            raise InvalidUpstreamResponse("LLM returned no choices")

        content = response.choices[0].message.content

        if content is None:
            raise InvalidUpstreamResponse("LLM returned no message content")

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            logger.warning(
                "LLM response truncated | model=%s",
                response.model,
            )

        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else 0

        reasoning_tokens = max(0, total_tokens - prompt_tokens - completion_tokens)

        return LLMResult(
            text=content,
            model=response.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    async def generate(self, prompt: str, max_tokens: int) -> LLMResult:
        retry_start = time.perf_counter()

        for attempt in range(self.settings.max_retries + 1):
            try:
                return await self._generate_once(prompt=prompt, max_tokens=max_tokens)

            except (UpstreamRateLimited, UpstreamUnavailable) as exc:
                if attempt >= self.settings.max_retries:
                    raise

                if (
                    isinstance(exc, UpstreamRateLimited)
                    and exc.retry_after_seconds is not None
                ):
                    delay = min(
                        exc.retry_after_seconds,
                        self.settings.retry_max_delay_seconds,
                    )
                    delay_source = "retry-after"
                else:
                    delay = calculate_retry_delay(
                        attempt=attempt,
                        base_delay=self.settings.retry_base_delay_seconds,
                        max_delay=self.settings.retry_max_delay_seconds,
                    )
                    delay_source = "exponential-backoff"

                elapsed = time.perf_counter() - retry_start
                remaining_time = self.settings.max_retry_time_seconds - elapsed

                if delay > remaining_time:
                    logger.warning(
                        "LLM retry skipped; retry time budget exceeded | "
                        "elapsed_seconds=%.2f | "
                        "remaining_seconds=%.2f | "
                        "delay_seconds=%.2f",
                        elapsed,
                        max(0.0, remaining_time),
                        delay,
                    )
                    raise

                logger.warning(
                    "LLM request failed; retrying | "
                    "attempt=%s | max_retries=%s | "
                    "error=%s | delay_seconds=%.2f | source=%s",
                    attempt + 1,
                    self.settings.max_retries,
                    exc.error_code,
                    delay,
                    delay_source,
                )

                await asyncio.sleep(delay)
        raise RuntimeError("Retry loop exited without result")
