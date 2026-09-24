"""FastAPI app for the React forecasting dashboard."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from transaction_forecasting.api.router import router
from transaction_forecasting.api.service import PROJECT_ROOT, ArtifactUnavailable

app = FastAPI(
    title="Transaction Activity Forecasting",
    description="Dashboard API for UBS predictions, validation artifacts and experiment history.",
    version="0.2.0",
)
app.include_router(router)


@app.exception_handler(ArtifactUnavailable)
async def artifact_error_handler(_request, exception: ArtifactUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exception)})


frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
