from __future__ import annotations

from datetime import datetime

import pytest
import pytz

from core.data.schema import Transaction, TransactionType, transactions_to_dataframe


IST = pytz.timezone("Asia/Kolkata")


def _txn(**overrides) -> Transaction:
    defaults = dict(
        transaction_id="TXN1",
        sender_account_id="ACC1",
        receiver_account_id="ACC2",
        amount_inr=100.0,
        timestamp=IST.localize(datetime(2024, 1, 1, 10, 0, 0)),
        transaction_type=TransactionType.UPI,
    )
    defaults.update(overrides)
    return Transaction(**defaults)


def test_valid_transaction_constructs():
    txn = _txn()
    assert txn.transaction_id == "TXN1"
    assert txn.transaction_type == TransactionType.UPI


def test_non_positive_amount_rejected():
    with pytest.raises(ValueError):
        _txn(amount_inr=0.0)
    with pytest.raises(ValueError):
        _txn(amount_inr=-5.0)


def test_naive_timestamp_rejected():
    with pytest.raises(ValueError):
        _txn(timestamp=datetime(2024, 1, 1, 10, 0, 0))  # no tzinfo


def test_transaction_is_frozen():
    txn = _txn()
    with pytest.raises(Exception):
        txn.amount_inr = 500.0  # type: ignore[misc]


def test_transactions_to_dataframe_empty():
    df = transactions_to_dataframe([])
    assert df.empty
    assert list(df.columns) == [
        "transaction_id",
        "sender_account_id",
        "receiver_account_id",
        "amount_inr",
        "timestamp",
        "transaction_type",
    ]


def test_transactions_to_dataframe_single_row():
    df = transactions_to_dataframe([_txn()])
    assert len(df) == 1
    assert df.iloc[0]["transaction_id"] == "TXN1"
    assert df.iloc[0]["transaction_type"] == "UPI"  # stored as string, not Enum
