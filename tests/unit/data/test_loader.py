from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.config import Config
from core.data.loader import SchemaError, load_transactions_csv, valid_transactions_dataframe


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    path = tmp_path / "transactions.csv"
    df.to_csv(path, index=False)
    return path


def test_missing_file_raises(config: Config, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_transactions_csv(tmp_path / "does_not_exist.csv", config)


def test_missing_required_column_raises_schema_error(config: Config, tmp_path: Path):
    path = _write_csv(
        tmp_path,
        [{"transaction_id": "TXN1", "sender_account_id": "A", "amount_inr": 10}],  # no receiver/timestamp/type
    )
    with pytest.raises(SchemaError):
        load_transactions_csv(path, config)


def test_valid_csv_loads_all_rows(config: Config, tmp_path: Path):
    rows = [
        {
            "transaction_id": "TXN1",
            "sender_account_id": "ACC1",
            "receiver_account_id": "ACC2",
            "amount_inr": 100.0,
            "timestamp": "2024-01-01 10:00:00",
            "transaction_type": "UPI",
        },
        {
            "transaction_id": "TXN2",
            "sender_account_id": "ACC2",
            "receiver_account_id": "ACC3",
            "amount_inr": 250.0,
            "timestamp": "2024-01-02 11:00:00",
            "transaction_type": "NEFT",
        },
    ]
    path = _write_csv(tmp_path, rows)
    outcome = load_transactions_csv(path, config)
    assert len(outcome.valid_transactions) == 2
    assert outcome.rejects.empty

    df = valid_transactions_dataframe(outcome)
    assert len(df) == 2
    assert set(df["transaction_id"]) == {"TXN1", "TXN2"}


def test_csv_with_invalid_rows_quarantines_not_drops(config: Config, tmp_path: Path):
    rows = [
        {
            "transaction_id": "TXN1",
            "sender_account_id": "ACC1",
            "receiver_account_id": "ACC2",
            "amount_inr": 100.0,
            "timestamp": "2024-01-01 10:00:00",
            "transaction_type": "UPI",
        },
        {
            "transaction_id": "TXN2",
            "sender_account_id": "ACC2",
            "receiver_account_id": "ACC2",  # self-transaction
            "amount_inr": -5.0,  # also invalid amount
            "timestamp": "not-a-date",  # also invalid timestamp
            "transaction_type": "UPI",
        },
    ]
    path = _write_csv(tmp_path, rows)
    outcome = load_transactions_csv(path, config)
    assert len(outcome.valid_transactions) == 1
    assert len(outcome.rejects) == 1
    reasons = outcome.rejects.iloc[0]["reasons"]
    assert "SELF_TRANSACTION" in reasons
    assert "NON_POSITIVE_AMOUNT" in reasons
    assert "MALFORMED_TIMESTAMP" in reasons


def test_empty_csv_with_headers_only(config: Config, tmp_path: Path):
    path = tmp_path / "empty.csv"
    path.write_text("transaction_id,sender_account_id,receiver_account_id,amount_inr,timestamp,transaction_type\n")
    outcome = load_transactions_csv(path, config)
    assert outcome.valid_transactions == []
    assert outcome.rejects.empty
