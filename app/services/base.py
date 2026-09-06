from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int
    finish_reason: str
    
class LLMClient(Protocol):
    async def generate(
        self,
        prompt: str,
        max_tokens: int
    ) -> LLMResult: ...
    