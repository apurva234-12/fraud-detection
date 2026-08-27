from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
import pytz

from core.config import Config
from core.features.transaction_features import ACCOUNT_FEATURE_COLUMNS, compute_account_features


IST = pytz.timezone("Asia/Kolkata")


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "transaction_id", "sender_account_id", "receiver_account_id",
        "amount_inr", "timestamp", "transaction_type",
    ])


def _ts(day: int, hour: int = 10) -> pd.Timestamp:
    return pd.Timestamp(IST.localize(datetime(2024, 1, day, hour, 0, 0)))


def test_empty_input_returns_empty_correctly_columned(config: Config):
    df = compute_account_features(_df([]), config)
    assert df.empty
    assert list(df.columns) == list(ACCOUNT_FEATURE_COLUMNS)


def test_single_transaction_basic_features(config: Config):
    rows = [
        {
            "transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
            "amount_inr": 100.0, "timestamp": _ts(1), "transaction_type": "UPI",
        }
    ]
    df = compute_account_features(_df(rows), config).set_index("account_id")

    assert df.loc["A", "txn_count_sent"] == 1
    assert df.loc["A", "txn_count_received"] == 0
    assert df.loc["A", "total_amount_sent"] == 100.0
    assert df.loc["A", "inflow_outflow_ratio"] == 0.0  # A never receives

    assert df.loc["B", "txn_count_received"] == 1
    assert df.loc["B", "txn_count_sent"] == 0
    assert df.loc["B", "inflow_outflow_ratio"] == float("inf")  # B only receives


def test_unique_counterparties_counts_distinct_not_total(config: Config):
    rows = [
        {"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(1), "transaction_type": "UPI"},
        {"transaction_id": "T2", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 20.0, "timestamp": _ts(1, 11), "transaction_type": "UPI"},
        {"transaction_id": "T3", "sender_account_id": "A", "receiver_account_id": "C",
         "amount_inr": 30.0, "timestamp": _ts(1, 12), "transaction_type": "UPI"},
    ]
    df = compute_account_features(_df(rows), config).set_index("account_id")
    assert df.loc["A", "txn_count_sent"] == 3
    assert df.loc["A", "unique_counterparties_sent"] == 2  # B and C, not 3


def test_inflow_outflow_ratio_balanced_account(config: Config):
    rows = [
        {"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 100.0, "timestamp": _ts(1), "transaction_type": "UPI"},
        {"transaction_id": "T2", "sender_account_id": "B", "receiver_account_id": "A",
         "amount_inr": 50.0, "timestamp": _ts(2), "transaction_type": "UPI"},
    ]
    df = compute_account_features(_df(rows), config).set_index("account_id")
    # A sends 100, receives 50 -> ratio 0.5
    assert df.loc["A", "inflow_outflow_ratio"] == pytest.approx(0.5)
    # B sends 50, receives 100 -> ratio 2.0
    assert df.loc["B", "inflow_outflow_ratio"] == pytest.approx(2.0)


def test_active_days_and_first_last_seen(config: Config):
    rows = [
        {"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(1), "transaction_type": "UPI"},
        {"transaction_id": "T2", "sender_account_id": "A", "receiver_account_id": "C",
         "amount_inr": 10.0, "timestamp": _ts(5), "transaction_type": "UPI"},
    ]
    df = compute_account_features(_df(rows), config).set_index("account_id")
    assert df.loc["A", "active_days"] == pytest.approx(4.0)


def test_max_daily_txn_count_velocity_proxy(config: Config):
    rows = [
        {"transaction_id": f"T{i}", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(1, hour=i), "transaction_type": "UPI"}
        for i in range(5)  # 5 transactions same day
    ] + [
        {"transaction_id": "T99", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0, "timestamp": _ts(2), "transaction_type": "UPI"}  # 1 on another day
    ]
    df = compute_account_features(_df(rows), config).set_index("account_id")
    assert df.loc["A", "max_daily_txn_count"] == 5
