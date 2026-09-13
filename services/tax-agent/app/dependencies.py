from fastapi import Request

from app.services.base import LLMGateway, Retriever


def get_llm_gateway(request: Request) -> LLMGateway:
    return request.app.state.llm_gateway


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever
