"""Shared application state: model artifacts loaded once at API startup (plan D6).

Artifacts are attached to ``app.state.artifacts`` in a FastAPI lifespan handler so
tests can swap in fakes and the app does not fail at import time.
"""

import time
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger("api_state")


def load_artifacts() -> Dict[str, Any]:
    """Load every serving artifact named by the model card (via the scorer)."""
    from src.serving.batch_score import load_artifacts as _load

    return _load()


def get_artifacts(request) -> Dict[str, Any]:
    return request.app.state.artifacts


def get_card(request) -> Dict[str, Any]:
    return request.app.state.artifacts["card"]


def log_prediction(customer_id: Optional[float], latency_ms: float, endpoint: str = "/predict") -> None:
    """Best-effort write to ``prediction_log``; never breaks a request."""
    try:
        from src.serving import db
        from src.serving.models import PredictionLog

        if db.get_engine() is None:
            db.ensure_engine()
        if db.get_engine() is None:
            return
        with db.get_session() as session:
            session.add(PredictionLog(
                customer_id=customer_id,
                model_version="api",
                latency_ms=latency_ms,
                endpoint=endpoint,
            ))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"prediction_log write skipped: {exc}")


class Timer:
    """Small latency helper for logging single-prediction latency."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._t0) * 1000.0
        return False
