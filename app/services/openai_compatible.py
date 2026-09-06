import logging

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.config import Settings
from app.services.base import LLMClient, LLMResult
from app.errors import (
    InvalidUpstreamResponse,
    UpstreamTimeout,
    UpstreamRateLimited,
    UpstreamUnavailable,
)

logger = logging.getLogger(__name__)

class OpenAICompatibleClient:
    
    def __init__(
        self,
        client: AsyncOpenAI,
        settings: Settings
    ):
        self.client = client
        self.settings = settings
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int
    ) -> LLMResult:
        
        try:
            response = await self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                reasoning_effort=self.settings.reasoning_effort
            )
            
        except APITimeoutError as exc:
            raise UpstreamTimeout() from exc
        
        except RateLimitError as exc:
            raise UpstreamRateLimited() from exc
        
        except APIConnectionError as exc:
            raise UpstreamUnavailable() from exc
        
        except APIStatusError as exc:
            if 500 <= exc.status_code < 600:
                raise UpstreamUnavailable() from exc
            raise InvalidUpstreamResponse() from exc
        
        if not response.choices:
            raise InvalidUpstreamResponse(
                "LLM returned no choices"
            )
            
        content = response.choices[0].message.content
        
        if content is None:
            raise InvalidUpstreamResponse(
                "LLM returned no message content"
            )
            
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
        
        reasoning_tokens = max(
            0,
            total_tokens - prompt_tokens - completion_tokens
        )
        
        return LLMResult(
            text=content,
            model=response.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason
        )