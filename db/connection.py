"""
Database engine and session management.

Deliberately framework-agnostic: nothing here knows about FastAPI (that is a
later phase). This module is usable directly from plain scripts, pytest, and
eventually FastAPI's dependency-injection system without modification.

Connection string comes from the environment (`DATABASE_URL`), never
hardcoded, so the same code works against a local dev DB, a CI test DB, and
whatever the eventual deployment target is, just by changing the environment.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://fraud_app:fraud_app@localhost:5432/fraud_detection"


def get_database_url() -> str:
    """Read the database URL from the environment. Falls back to a sane
    local-development default (matching the docker-compose service suggested
    in the setup instructions) rather than raising, so a fresh checkout can
    still run `pytest` after `docker compose up -d` without extra config.
    """
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine(database_url: str | None = None) -> Engine:
    """Return a process-wide SQLAlchemy engine, creating it on first use.

    `database_url` is accepted explicitly (rather than only ever reading the
    environment) so tests can point this at a dedicated test database without
    mutating process environment variables.
    """
    global _engine
    if _engine is None or database_url is not None:
        url = database_url or get_database_url()
        _engine = create_engine(url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a process-wide session factory bound to `get_engine()`."""
    global _SessionLocal
    if _SessionLocal is None or database_url is not None:
        _SessionLocal = sessionmaker(bind=get_engine(database_url), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    """Context-manager session with commit-on-success / rollback-on-error
    semantics, for use in scripts and repository-level integration tests:

        with session_scope() as session:
            ...
            # commits automatically if no exception was raised

    A later FastAPI dependency (`api/deps.py`, not part of this phase) will
    likely wrap `get_session_factory()` directly instead of this
    context manager, since request-scoped session lifecycle in FastAPI has
    its own conventions -- this helper is for scripts/tests, not routes.
    """
    session_factory = get_session_factory(database_url)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Clear the cached engine/session factory. Only needed in tests that
    deliberately swap `DATABASE_URL` between test cases."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
