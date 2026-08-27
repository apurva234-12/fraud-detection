"""
PostgreSQL persistence layer (Student B's portion of the project).

Submodules:
    models.py      SQLAlchemy ORM models / schema
    connection.py  Engine + session management (reads DATABASE_URL from env)
    repository.py  Read/write functions; the only module that imports from `core/`

Nothing in `core/` imports from this package, and nothing in this package
modifies `core/` -- see `db/repository.py` module docstring for the
integration contract.
"""
