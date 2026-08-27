"""
Fixtures for the database test suite.

Deliberately targets a REAL PostgreSQL database, not SQLite. This layer uses
Postgres-specific types (native ARRAY, JSONB, NUMERIC) and constraints
(CHECK, composite PK) that SQLite does not faithfully emulate -- a green
SQLite run would not actually prove the schema works, so we don't offer that
shortcut.

Point `DATABASE_URL` (env var) at a disposable test database before running
this suite -- see the project setup instructions for a docker-compose
snippet. Every test in this package runs inside a transaction that is rolled
back afterwards, so tests never depend on run order or leave data behind for
the next run.

The `config` fixture mirrors the one already used by `tests/unit/` in
`core/` (a plain `core.config.Config()` with defaults) so the same
`generate_fixture_dataset(config)` used in `core/`'s own tests can be reused
here for integration testing against the database.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.config import Config
from db.connection import get_engine, get_session_factory
from db.models import Base


@pytest.fixture(scope="session")
def config() -> Config:
    return Config()


@pytest.fixture(scope="session")
def _database_available() -> bool:
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def engine(_database_available: bool):
    if not _database_available:
        pytest.skip(
            "Test database unreachable; set DATABASE_URL and ensure Postgres is running."
        )

    eng = get_engine()
    Base.metadata.create_all(eng)

    yield eng

    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def session(engine) -> Iterator[Session]:
    connection = engine.connect()
    transaction = connection.begin()

    session_factory = get_session_factory()
    session_factory.configure(bind=connection)
    db_session = session_factory()

    try:
        yield db_session
    finally:
        transaction.rollback()
        db_session.close()
        connection.close()