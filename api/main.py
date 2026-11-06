"""FastAPI application entrypoint (plan D6/D7).

Artifacts and the database engine are initialised once in the lifespan handler,
not at import time, so tests can swap in fakes and startup failures are visible.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import customers, monitoring, predictions
from api.state import load_artifacts
from src.serving import db
from src.utils.logger import get_logger

logger = get_logger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model artifacts and database engine.")
    try:
        app.state.artifacts = load_artifacts()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Artifact load failed at startup: {exc}")
        app.state.artifacts = None
    db.ensure_engine()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Vantara Retail Solutions — Customer Behavior Prediction API",
    description=(
        "Machine Learning & Deep Learning REST API for calibrated churn risk, "
        "90-day customer value, value-at-risk prioritisation, segmentation and XAI."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics (plan D7).
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Prometheus instrumentation unavailable: {exc}")

app.include_router(predictions.router, tags=["Predictions & Metadata"])
app.include_router(customers.router, tags=["Customers & Segments"])
app.include_router(monitoring.router, tags=["Monitoring & Revenue"])


@app.get("/")
def root():
    return {
        "message": "Vantara Customer Behavior Prediction Platform API",
        "documentation": "/docs",
        "health_check": "/health",
        "metadata": "/metadata",
        "at_risk": "/customers/at-risk",
        "segments": "/segments",
        "drift": "/monitoring/drift",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
