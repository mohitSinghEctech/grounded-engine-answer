import logging

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_retriever
from app.schemas import ErrorResponse, PassageOut, SearchRequest, SearchResponse
from app.services.base import Retriever
from app.tax_year import extract_tax_year

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        422: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
    settings: Settings = Depends(get_settings),
):
    if request.tax_year is not None:
        tax_year, source = request.tax_year, "request"
    else:
        detected = extract_tax_year(request.question)
        tax_year, source = detected, ("question" if detected else "none")

    logger.info(
        "Search started | tax_year=%s | source=%s | top_k=%s",
        tax_year,
        source,
        request.top_k,
    )

    passages = await retriever.search(
        question=request.question,
        top_k=request.top_k,
        tax_year=tax_year,
    )

    return SearchResponse(
        query=request.question,
        tax_year=tax_year,
        tax_year_source=source,
        collection=settings.qdrant_collection,
        passages=[PassageOut(**vars(p)) for p in passages],
    )
