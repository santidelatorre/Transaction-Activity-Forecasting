"""Read-only HTTP endpoints backed by the official baseline artifacts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from transaction_forecasting.api import service

router = APIRouter(prefix="/api/v1")


@router.get("/experiments")
def experiments(
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    return service.get_experiments(limit=limit, offset=offset)


@router.get("/health")
def health() -> dict[str, object]:
    return service.get_health()


@router.get("/overview")
def overview(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=1, le=100),
    q: str = Query(default="", max_length=100),
) -> dict[str, object]:
    return service.get_overview(page=page, page_size=page_size, search=q)


@router.get("/clients/{client_id}")
def client_detail(client_id: str) -> dict[str, object]:
    result = service.get_client(client_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Client prediction is unavailable.")
    return result


@router.get("/results")
def results() -> dict[str, object]:
    return service.get_results()


@router.get("/submission.csv")
def download_submission() -> Response:
    csv_content = service.submission_csv()
    if csv_content is None:
        raise HTTPException(status_code=404, detail="No validated submission is available.")
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="submission_v1.csv"'},
    )
