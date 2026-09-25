"""FastAPI entry point; a missing bundle is a real 503, never mock data."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from transaction_forecasting.product.adapter import ModelAdapter, UnknownClient
from transaction_forecasting.product.agent import investigate
from transaction_forecasting.product.provenance import DEFAULT_BUNDLE, ROOT, ArtifactUnavailable


@lru_cache(maxsize=1)
def get_adapter() -> ModelAdapter:
    try:
        return ModelAdapter(
            Path(os.environ.get("DEMO_BUNDLE", str(DEFAULT_BUNDLE))),
            Path(os.environ.get("DEMO_DATA_DIR", str(ROOT / "data/raw/ubs_2026"))),
        )
    except (OSError, ValueError, KeyError) as error:
        raise ArtifactUnavailable("Frozen model bundle is invalid; prepare it again") from error


Adapter = Annotated[ModelAdapter, Depends(get_adapter)]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Recurring Insights",
        version="4.0.0",
        description="Read-only evidence and investigation for frozen V3-A",
    )

    @app.exception_handler(ArtifactUnavailable)
    async def unavailable(_request, exception):
        return JSONResponse(status_code=503, content={"detail": str(exception)})

    @app.exception_handler(UnknownClient)
    async def unknown(_request, _exception):
        return JSONResponse(status_code=404, content={"detail": "Unknown TEST client"})

    @app.get("/api/v1/health")
    def health(adapter: Adapter):
        metadata = adapter.get_model_metadata()
        return {
            "status": "ready",
            "model_version": metadata["model_version"],
            "base_sha": metadata["base_sha"],
            "horizon_days": 90,
            "model_artifact_sha256": metadata["artifact_sha256"],
        }

    @app.get("/api/v1/model")
    def metadata(adapter: Adapter):
        return adapter.get_model_metadata()

    @app.get("/api/v1/metrics")
    def metrics(adapter: Adapter):
        return adapter.get_global_metrics()

    @app.get("/api/v1/cases")
    def cases(adapter: Adapter):
        return adapter.get_cases()

    @app.get("/api/v1/clients/{client_id}")
    def client(client_id: str, adapter: Adapter):
        return adapter.explain_prediction(client_id)

    @app.get("/api/v1/clients/{client_id}/prediction")
    def prediction(client_id: str, adapter: Adapter):
        return adapter.predict_client(client_id)

    @app.post("/api/v1/clients/{client_id}/investigate")
    def investigation(
        client_id: str,
        adapter: Adapter,
        max_steps: int = Query(8, ge=1, le=8),
    ):
        return investigate(adapter, client_id, max_steps=max_steps)

    frontend = ROOT / "frontend/dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
