from openai import AsyncOpenAI

from app.config import Settings
from app.services.base import LLMClient, LLMResult

class OpenAICompatibleClient(LLMClient):
    
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
        
        response = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
        )
        
        usage = response.usage
        
        return LLMResult(
            text=response.choices[0].message.content,
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )