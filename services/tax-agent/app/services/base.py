from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    

class LLMGateway(Protocol):
    async def generate(self, prompt: str, max_tokens: int) -> Generation: ...