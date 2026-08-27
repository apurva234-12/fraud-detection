"""
Alembic environment script.

Wired to `db.models.Base.metadata` for autogenerate support, and to the same
`DATABASE_URL` environment variable convention used by `db.connection`, so
migrations run against whatever database the app itself would connect to --
no separate/duplicated connection config to keep in sync.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project root importable (so `db.models` resolves) regardless of
# the working directory `alembic` is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.connection import get_database_url  # noqa: E402
from db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate compares against this metadata to build migration diffs.
target_metadata = Base.metadata

# Override the (blank) sqlalchemy.url from alembic.ini with DATABASE_URL,
# same source of truth db.connection.get_database_url() uses.
config.set_main_option("sqlalchemy.url", get_database_url())


def run_migrations_offline() -> None:
    """Generate SQL scripts without a live DB connection (`alembic upgrade
    head --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection -- the normal case."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
