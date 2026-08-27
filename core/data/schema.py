"""
Canonical transaction data model.

Design decisions (see docs/design_decisions.md for full rationale):

- A `Transaction` is a single, immutable, already-validated record. Constructing
  one directly does NOT perform validation -- validation is a separate,
  explicit step (see `core.data.validators`) so that we can collect *all*
  problems with a batch of raw input rather than fail on the first bad field,
  and so invalid rows can be routed to a rejects report instead of raised as
  exceptions or silently skipped.
- Scenario labels used by the synthetic generator (e.g. "HIGH_FANOUT",
  "CIRCULAR_FLOW") are intentionally NOT a field on `Transaction`. They live in
  a separate structure (see `core.data.synthetic.SyntheticDataset`) so they can
  never accidentally leak into feature engineering, graph construction, or a
  future model as if they were real fraud labels.
- Currency is implicit and fixed: INR. There is no currency field because
  multi-currency data is explicitly out of scope for this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence

import pandas as pd


class TransactionType(str, Enum):
    """Closed set of supported Indian digital/electronic banking transaction
    types. Deliberately does not include cards, crypto, or investment
    instruments -- see project scope notes in the README."""

    UPI = "UPI"
    NEFT = "NEFT"
    IMPS = "IMPS"
    RTGS = "RTGS"
    CASH_DEPOSIT = "CASH_DEPOSIT"
    CASH_WITHDRAWAL = "CASH_WITHDRAWAL"


# Columns that MUST be present (though not necessarily non-null -- nullness is
# checked separately) on any raw input row before validation is attempted.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "sender_account_id",
    "receiver_account_id",
    "amount_inr",
    "timestamp",
    "transaction_type",
)


@dataclass(frozen=True, slots=True)
class Transaction:
    """A single validated transaction. Amount is INR. Timestamp is timezone-aware
    and normalised to the project timezone (IST) by the validator before this
    object is constructed."""

    transaction_id: str
    sender_account_id: str
    receiver_account_id: str
    amount_inr: float
    timestamp: datetime
    transaction_type: TransactionType

    def __post_init__(self) -> None:
        # Defensive invariants -- these should already be guaranteed by the
        # validator, but a frozen dataclass constructed elsewhere (e.g. in a
        # test) should never silently hold an impossible value.
        if self.amount_inr <= 0:
            raise ValueError("Transaction.amount_inr must be strictly positive")
        if self.timestamp.tzinfo is None:
            raise ValueError("Transaction.timestamp must be timezone-aware")


# Canonical column order used whenever transactions are materialised as a
# DataFrame -- kept in one place so every module produces/consumes the same
# shape instead of re-deriving column order ad hoc.
TRANSACTION_DF_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "sender_account_id",
    "receiver_account_id",
    "amount_inr",
    "timestamp",
    "transaction_type",
)


def transactions_to_dataframe(transactions: Sequence[Transaction]) -> pd.DataFrame:
    """Materialise a list of validated `Transaction` objects as a DataFrame in
    canonical column order. `transaction_type` is kept as its string value
    (not the Enum object) so downstream pandas operations (groupby, value
    filtering) behave normally."""

    if not transactions:
        return pd.DataFrame(columns=TRANSACTION_DF_COLUMNS)

    records = [
        {
            "transaction_id": t.transaction_id,
            "sender_account_id": t.sender_account_id,
            "receiver_account_id": t.receiver_account_id,
            "amount_inr": t.amount_inr,
            "timestamp": t.timestamp,
            "transaction_type": t.transaction_type.value,
        }
        for t in transactions
    ]
    return pd.DataFrame.from_records(records, columns=list(TRANSACTION_DF_COLUMNS))
