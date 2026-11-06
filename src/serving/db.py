"""Database engine/session helpers shared by the batch scorer and the API (plan D6).

``DATABASE_URL`` selects the backend. When it is unset the API still starts:
read endpoints degrade gracefully and the scorer falls back to SQLite so the
project runs without Docker.
"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.serving.models import Base
from src.utils.logger import get_logger

logger = get_logger("serving_db")

_FALLBACK_URL = "sqlite:///./vantara.db"
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def _make_engine(url: str) -> Engine:
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    # SQLAlchemy 2.1 defaults bare `postgresql://` to the psycopg (v3) driver;
    # we ship psycopg2-binary, so pin the driver explicitly.
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def init_engine(url: Optional[str] = None, create_all: bool = True) -> Optional[Engine]:
    """(Re)initialise the global engine. Returns None if no DB is configured."""
    global _engine, _SessionLocal
    url = url or get_database_url()
    if not url:
        return None
    _engine = _make_engine(url)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)
    if create_all:
        Base.metadata.create_all(_engine)
    logger.info(f"Database engine initialised ({url.split('@')[-1]})")
    return _engine


def get_engine() -> Optional[Engine]:
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context-managed session; raises if the engine has not been initialised."""
    if _SessionLocal is None:
        raise RuntimeError("Database engine not initialised. Call init_engine() first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Alias used by callers/tests.
get_session = session_scope


def ensure_engine() -> Optional[Engine]:
    """Initialise from the environment, falling back to SQLite when unset."""
    global _engine
    if _engine is not None:
        return _engine
    url = get_database_url() or _FALLBACK_URL
    return init_engine(url)


def reset_engine() -> None:
    """Test helper: drop the cached engine/session factory."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
