"""
Repository layer -- the integration seam between `core/` and PostgreSQL.

This is the ONLY module that imports from `core/`. It never modifies
anything in `core/`; it only consumes the objects `core/` already produces:

    core.data.validators.ValidationOutcome   (via core.data.loader / core.data.synthetic)
    core.data.schema.Transaction
    core.data.synthetic.SyntheticDataset
    core.data.schema.transactions_to_dataframe

Everything else in `db/` (models.py, connection.py) has zero knowledge of
`core/` and could be reused for an entirely different data source.

Design note: bulk transaction inserts are intentionally NOT upserts. If a
`transaction_id` already exists in the database, the insert raises
`sqlalchemy.exc.IntegrityError` (via the `transactions.transaction_id`
primary key) rather than silently overwriting or skipping it. This is
deliberate: `core.data.validators.validate_batch` only guarantees
first-seen-wins duplicate detection WITHIN a single batch (see
`docs/design_decisions.md` §5) -- it has no idea what has already been
persisted from a previous batch. A duplicate ID across batches is a
data-integrity problem the caller needs to see, not something this layer
should paper over.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.data.schema import TRANSACTION_DF_COLUMNS, Transaction as CoreTransaction
from core.data.validators import ValidationOutcome

from db.models import (
    Account,
    AccountScenarioLabel,
    IngestionBatch,
    Transaction,
    TransactionReject,
    TransactionScenarioLabel,
)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


def upsert_accounts(session: Session, account_ids: Iterable[str]) -> int:
    """Ensure every account_id in `account_ids` exists in the `accounts`
    table. Accounts that already exist are left untouched. Returns the
    number of NEW accounts created.

    This is how the `accounts` table gets populated -- `core/` has no
    `Account` entity of its own (see `core/graph/builder.py`); every account
    we know about comes from the distinct sender/receiver IDs seen across
    ingested transactions.
    """
    account_ids = set(account_ids)
    if not account_ids:
        return 0

    existing = set(
        session.scalars(select(Account.account_id).where(Account.account_id.in_(account_ids)))
    )
    new_ids = account_ids - existing
    for account_id in new_ids:
        session.add(Account(account_id=account_id))
    if new_ids:
        session.flush()
    return len(new_ids)


def get_isolated_accounts(session: Session) -> list[str]:
    """Return account_ids present in `accounts` with zero rows in
    `transactions` (as either sender or receiver).

    This is the natural source for `core.graph.builder.build_transaction_graph`'s
    `extra_isolated_accounts` parameter -- see that module's docstring, which
    explicitly anticipates an "accounts table with zero activity" as the
    source of such accounts.
    """
    active_ids = set(session.scalars(select(Transaction.sender_account_id))) | set(
        session.scalars(select(Transaction.receiver_account_id))
    )
    all_ids = set(session.scalars(select(Account.account_id)))
    return sorted(all_ids - active_ids)


# ---------------------------------------------------------------------------
# Ingestion batches
# ---------------------------------------------------------------------------


def create_ingestion_batch(
    session: Session,
    source_filename: Optional[str],
    valid_count: int,
    reject_count: int,
    config_snapshot: Optional[dict] = None,
) -> IngestionBatch:
    """Record one ingestion run. Returns the created batch with `id` populated
    (flushed, not committed -- caller controls the transaction boundary)."""
    batch = IngestionBatch(
        source_filename=source_filename,
        valid_count=valid_count,
        reject_count=reject_count,
        config_snapshot=config_snapshot,
    )
    session.add(batch)
    session.flush()
    return batch


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def insert_transactions(
    session: Session,
    transactions: list[CoreTransaction],
    batch_id: Optional[int] = None,
) -> int:
    """Convert a list of validated `core.data.schema.Transaction` objects into
    `db.models.Transaction` rows and insert them.

    Does NOT validate anything -- these are assumed to have already passed
    `core.data.validators.validate_batch`. Does NOT create accounts -- call
    `upsert_accounts` first with the sender/receiver IDs, or use
    `ingest_validation_outcome` which does both in the right order.

    Raises `sqlalchemy.exc.IntegrityError` on the first transaction_id that
    already exists in the database (see module docstring).
    """
    if not transactions:
        return 0

    rows = [
        Transaction(
            transaction_id=t.transaction_id,
            sender_account_id=t.sender_account_id,
            receiver_account_id=t.receiver_account_id,
            amount_inr=t.amount_inr,
            timestamp=t.timestamp,
            transaction_type=t.transaction_type.value,
            ingested_batch_id=batch_id,
        )
        for t in transactions
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)


def get_transaction(session: Session, transaction_id: str) -> Optional[Transaction]:
    return session.get(Transaction, transaction_id)


def transactions_for_account(session: Session, account_id: str) -> list[Transaction]:
    """All transactions where `account_id` is either sender or receiver."""
    stmt = select(Transaction).where(
        (Transaction.sender_account_id == account_id) | (Transaction.receiver_account_id == account_id)
    )
    return list(session.scalars(stmt))


def transactions_dataframe(session: Session) -> pd.DataFrame:
    """Read every transaction back out of the database as a DataFrame shaped
    exactly like `core.data.schema.TRANSACTION_DF_COLUMNS` -- i.e. directly
    consumable by `core.features.compute_account_features` and
    `core.graph.build_transaction_graph` with zero adaptation on their side.

    This is the read-path half of the integration seam: `core/` never reads
    from the database directly, it only ever receives a plain DataFrame.
    """
    stmt = select(
        Transaction.transaction_id,
        Transaction.sender_account_id,
        Transaction.receiver_account_id,
        Transaction.amount_inr,
        Transaction.timestamp,
        Transaction.transaction_type,
    )
    rows = session.execute(stmt).all()
    df = pd.DataFrame.from_records(rows, columns=list(TRANSACTION_DF_COLUMNS))
    if not df.empty:
        # amount_inr comes back from Postgres NUMERIC as Decimal; core's
        # Transaction.amount_inr is a plain float, so cast back at the
        # boundary to match what core/ expects to receive.
        df["amount_inr"] = df["amount_inr"].astype(float)
    return df


# ---------------------------------------------------------------------------
# Rejects
# ---------------------------------------------------------------------------


def insert_rejects(
    session: Session,
    rejects: pd.DataFrame,
    batch_id: Optional[int] = None,
) -> int:
    """Persist every row of a `core.data.validators.ValidationOutcome.rejects`
    DataFrame (columns: row_index, transaction_id, reasons, details).

    Nothing here re-validates or filters -- every row handed in is stored,
    matching the "never silently drop" requirement the reject report itself
    exists to satisfy (`docs/design_decisions.md` §3).
    """
    if rejects.empty:
        return 0

    rows = [
        TransactionReject(
            row_index=int(record["row_index"]),
            transaction_id=record.get("transaction_id"),
            reasons=list(record["reasons"]),
            details=list(record["details"]),
            ingested_batch_id=batch_id,
        )
        for record in rejects.to_dict(orient="records")
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)


# ---------------------------------------------------------------------------
# Synthetic scenario labels (optional -- only used when persisting output of
# core.data.synthetic.generate_synthetic_dataset / generate_fixture_dataset)
# ---------------------------------------------------------------------------


def insert_transaction_scenario_labels(session: Session, transaction_scenarios: pd.DataFrame) -> int:
    """Persist `SyntheticDataset.transaction_scenarios` (columns:
    transaction_id, scenario). Requires the referenced transactions to
    already be persisted (FK to transactions.transaction_id)."""
    if transaction_scenarios.empty:
        return 0
    rows = [
        TransactionScenarioLabel(transaction_id=r["transaction_id"], scenario=r["scenario"])
        for r in transaction_scenarios.to_dict(orient="records")
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)


def insert_account_scenario_labels(session: Session, account_scenarios: pd.DataFrame) -> int:
    """Persist `SyntheticDataset.account_scenarios` (columns: account_id,
    scenario). Requires the referenced accounts to already exist (FK to
    accounts.account_id) -- call `upsert_accounts` first."""
    if account_scenarios.empty:
        return 0
    rows = [
        AccountScenarioLabel(account_id=r["account_id"], scenario=r["scenario"])
        for r in account_scenarios.to_dict(orient="records")
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)


# ---------------------------------------------------------------------------
# High-level glue: the one function most callers actually need
# ---------------------------------------------------------------------------


def ingest_validation_outcome(
    session: Session,
    outcome: ValidationOutcome,
    source_filename: Optional[str] = None,
    config_snapshot: Optional[dict] = None,
) -> IngestionBatch:
    """Persist a full `core.data.validators.ValidationOutcome` (as returned by
    `core.data.loader.load_transactions_csv` or produced from
    `core.data.synthetic` output run through `validate_batch`):

      1. Create an `ingestion_batches` row recording this run.
      2. Upsert every account referenced by the valid transactions.
      3. Insert the valid transactions.
      4. Insert the reject rows (if any), linked to the same batch.

    Does not commit -- wrap the call in `db.connection.session_scope()` (or
    manage the session/transaction yourself) so a failure partway through
    (e.g. a duplicate transaction_id) rolls back the whole batch rather than
    leaving it half-persisted.
    """
    batch = create_ingestion_batch(
        session,
        source_filename=source_filename,
        valid_count=len(outcome.valid_transactions),
        reject_count=len(outcome.rejects),
        config_snapshot=config_snapshot,
    )

    account_ids = set()
    for txn in outcome.valid_transactions:
        account_ids.add(txn.sender_account_id)
        account_ids.add(txn.receiver_account_id)
    upsert_accounts(session, account_ids)

    insert_transactions(session, outcome.valid_transactions, batch_id=batch.id)
    insert_rejects(session, outcome.rejects, batch_id=batch.id)

    return batch
