"""
Shared FastAPI dependencies.

`get_db` is the one piece of glue every database-backed route needs: it asks
`db.connection.get_session_factory()` (already built in Phase 1) for a
session, hands it to the route function, and guarantees the session is
closed afterwards -- even if the route raises an error.

Nothing in `core/` or `db/` is modified to support this; FastAPI is simply
calling functions that already exist.
"""

from __future__ import annotations

from typing import Iterator

from sqlalchemy.orm import Session

from db.connection import get_session_factory


def get_db() -> Iterator[Session]:
    """FastAPI dependency. Use it in a route like:

        @app.get("/transactions")
        def list_transactions(db: Session = Depends(get_db)):
            ...

    FastAPI calls this function for you, sends the yielded `Session` into
    your route as `db`, and runs the code after `yield` once the route
    finishes -- so the session always gets closed, whether the route
    succeeds or raises an exception.
    """
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()