from fastapi import Request

from app.services.base import LLMClient


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client
