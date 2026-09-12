from fastapi import Request

from app.services.base import LLMGateway


def get_llm_gateway(request: Request) -> LLMGateway:
    return request.app.state.llm_gateway