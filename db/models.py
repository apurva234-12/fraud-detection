"""
SQLAlchemy ORM models for the PostgreSQL persistence layer.

Scope decision (mirrors `docs/design_decisions.md` §1 and the approved V1
plan): this layer stores RAW / AUDIT data only --

    accounts, transactions, transaction_rejects, ingestion_batches,
    transaction_scenario_labels, account_scenario_labels

It does NOT persist `core.features.account_features` or
`core.graph.metrics.graph_metrics` output. Those remain computed on demand
from data read back out of `transactions`, exactly as `core/` already
computes them from an in-memory DataFrame. This keeps V1 simple and avoids
two sources of truth for derived analytics.

This module has NO dependency on anything in `core/` and vice versa -- it
only knows how to store the same *shape* of data that `core.data.schema`
already defines (transaction_id, sender/receiver account id, amount_inr,
timestamp, transaction_type). Conversion between `core.data.schema.Transaction`
objects and these ORM rows happens in `db/repository.py`, never here.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    ARRAY,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for every ORM model in this package."""


# Closed set of allowed transaction types, mirrored from
# `core.config.Config.allowed_transaction_types` default. Enforced here via a
# CHECK constraint (not a native Postgres ENUM) so that widening the allowed
# set later is a plain migration (ALTER TABLE ... DROP/ADD CONSTRAINT)
# instead of the more painful ALTER TYPE dance native enums require.
ALLOWED_TRANSACTION_TYPES: tuple[str, ...] = (
    "UPI",
    "NEFT",
    "IMPS",
    "RTGS",
    "CASH_DEPOSIT",
    "CASH_WITHDRAWAL",
)


class Account(Base):
    """An account, derived purely from the set of sender/receiver IDs seen
    across ingested transactions. `core/` has no `Account` entity of its own
    (see `core/graph/builder.py` docstring) -- this table, and the accounts
    in it, are entirely our addition.

    `core.graph.builder.build_transaction_graph` accepts an explicit
    `extra_isolated_accounts` parameter for accounts with zero transactions.
    This table is the natural source for that list: query for accounts with
    no rows in `transactions` and pass their IDs through.
    """

    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String, primary_key=True)
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"Account(account_id={self.account_id!r})"


class IngestionBatch(Base):
    """One record per call to the ingestion pipeline (e.g. one CSV load or
    one synthetic-dataset generation run persisted to the DB). Gives us an
    audit trail: which transactions/rejects came from which load, and under
    which config.

    `config_snapshot` stores a JSON dump of the `core.config.Config` used for
    that batch's validation. `core/` never persists `Config` itself (it is a
    frozen dataclass, YAML-overridable, in-memory only) -- capturing it here
    is purely for our own reproducibility/audit purposes and does not require
    any change to `core.config`.
    """

    __tablename__ = "ingestion_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    loaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    valid_count: Mapped[int] = mapped_column(Integer, default=0)
    reject_count: Mapped[int] = mapped_column(Integer, default=0)
    config_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="ingestion_batch")
    rejects: Mapped[list["TransactionReject"]] = relationship(back_populates="ingestion_batch")

    def __repr__(self) -> str:  # pragma: no cover
        return f"IngestionBatch(id={self.id}, source_filename={self.source_filename!r})"


class Transaction(Base):
    """Mirrors `core.data.schema.Transaction` field-for-field. Populated only
    from `core.data.schema.Transaction` objects that have ALREADY passed
    `core.data.validators.validate_batch` -- this table never receives raw,
    unvalidated rows directly.

    `transaction_id` is the PRIMARY KEY. This is deliberate and important:
    `core.data.validators.validate_batch` only enforces first-seen-wins
    duplicate detection WITHIN a single batch (see
    `docs/design_decisions.md` §5) -- it has no notion of transactions
    already persisted from a previous batch. The PK here is what enforces
    GLOBAL uniqueness across every batch ever loaded; a repeat insert of an
    already-persisted transaction_id will raise an IntegrityError, which the
    repository layer surfaces rather than swallows.

    `amount_inr` is stored as NUMERIC(14, 2), not FLOAT/DOUBLE PRECISION --
    standard practice for money, avoids binary floating-point rounding drift
    that `core`'s Python `float` representation is fine with in memory but
    which we don't want to accumulate at rest.
    """

    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint("amount_inr > 0", name="ck_transactions_amount_positive"),
        CheckConstraint(
            f"transaction_type IN ({', '.join(repr(t) for t in ALLOWED_TRANSACTION_TYPES)})",
            name="ck_transactions_type_allowed",
        ),
        Index("ix_transactions_sender_account_id", "sender_account_id"),
        Index("ix_transactions_receiver_account_id", "receiver_account_id"),
        Index("ix_transactions_timestamp", "timestamp"),
        Index("ix_transactions_transaction_type", "transaction_type"),
        Index("ix_transactions_sender_timestamp", "sender_account_id", "timestamp"),
        Index("ix_transactions_receiver_timestamp", "receiver_account_id", "timestamp"),
    )

    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    sender_account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.account_id"), nullable=False
    )
    receiver_account_id: Mapped[str] = mapped_column(
        String, ForeignKey("accounts.account_id"), nullable=False
    )
    amount_inr: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String, nullable=False)
    ingested_batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_batches.id"), nullable=True
    )

    ingestion_batch: Mapped["IngestionBatch | None"] = relationship(back_populates="transactions")

    def __repr__(self) -> str:  # pragma: no cover
        return f"Transaction(transaction_id={self.transaction_id!r})"


class TransactionReject(Base):
    """Mirrors one row of `core.data.validators.ValidationOutcome.rejects`.

    Unlike `Transaction`, `transaction_id` here is NOT unique and NOT the
    primary key -- a reject may have a null/duplicate transaction_id (e.g.
    the row that failed MISSING_FIELD before an ID could even be read, or a
    row rejected specifically because its ID was a duplicate). A surrogate
    `id` is the primary key instead.

    `reasons` and `details` are Postgres native TEXT[] arrays, chosen
    because `RowValidationResult`/`ValidationOutcome` already represent them
    as Python lists (a row can fail multiple rules at once) -- no lossy
    flattening into a single string column.
    """

    __tablename__ = "transaction_rejects"
    __table_args__ = (
        Index("ix_transaction_rejects_ingested_batch_id", "ingested_batch_id"),
        Index("ix_transaction_rejects_transaction_id", "transaction_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    transaction_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reasons: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    details: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    ingested_batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_batches.id"), nullable=True
    )

    ingestion_batch: Mapped["IngestionBatch | None"] = relationship(back_populates="rejects")

    def __repr__(self) -> str:  # pragma: no cover
        return f"TransactionReject(id={self.id}, transaction_id={self.transaction_id!r})"


class TransactionScenarioLabel(Base):
    """Mirrors one row of `core.data.synthetic.SyntheticDataset.transaction_scenarios`
    (transaction_id -> scenario, 1:1 -- every transaction, including NORMAL
    baseline ones, gets exactly one scenario label).

    Deliberately a SEPARATE table from `transactions`, per
    `docs/design_decisions.md` §9: these labels exist purely to evaluate the
    detection pipeline later and must never be joined into `transactions` as
    if they were a real fraud signal, and must never be treated as ground
    truth for anything beyond synthetic-data evaluation.
    """

    __tablename__ = "transaction_scenario_labels"

    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.transaction_id"), primary_key=True
    )
    scenario: Mapped[str] = mapped_column(String, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TransactionScenarioLabel(transaction_id={self.transaction_id!r}, scenario={self.scenario!r})"


class AccountScenarioLabel(Base):
    """Mirrors one row of `core.data.synthetic.SyntheticDataset.account_scenarios`
    (account_id -> scenario role, 1:MANY -- an account can hold multiple
    scenario roles at once, e.g. both PASS_THROUGH_INTERMEDIARY and
    CIRCULAR_FLOW_PARTICIPANT). Composite primary key reflects that.

    Same isolation rule as `TransactionScenarioLabel`: synthetic-only,
    never implied to be real fraud/AML signal.
    """

    __tablename__ = "account_scenario_labels"

    account_id: Mapped[str] = mapped_column(
        ForeignKey("accounts.account_id"), primary_key=True
    )
    scenario: Mapped[str] = mapped_column(String, primary_key=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AccountScenarioLabel(account_id={self.account_id!r}, scenario={self.scenario!r})"
