from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytz

from core.graph.builder import build_transaction_graph, to_weighted_digraph


IST = pytz.timezone("Asia/Kolkata")


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[
        "transaction_id", "sender_account_id", "receiver_account_id",
        "amount_inr", "timestamp", "transaction_type",
    ])


def _ts(day: int = 1) -> pd.Timestamp:
    return pd.Timestamp(IST.localize(datetime(2024, 1, day, 10, 0, 0)))


def test_empty_transactions_gives_empty_graph():
    graph = build_transaction_graph(_df([]))
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_single_transaction_creates_two_nodes_one_edge():
    rows = [{"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
             "amount_inr": 100.0, "timestamp": _ts(), "transaction_type": "UPI"}]
    graph = build_transaction_graph(_df(rows))
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.has_edge("A", "B")
    assert not graph.has_edge("B", "A")


def test_repeated_transactions_between_same_pair_create_multiple_edges():
    rows = [
        {"transaction_id": f"T{i}", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 10.0 * i, "timestamp": _ts(), "transaction_type": "UPI"}
        for i in range(1, 4)
    ]
    graph = build_transaction_graph(_df(rows))
    assert graph.number_of_edges() == 3  # not collapsed into 1
    edge_data = graph.get_edge_data("A", "B")
    assert len(edge_data) == 3
    amounts = sorted(d["amount_inr"] for d in edge_data.values())
    assert amounts == [10.0, 20.0, 30.0]


def test_edge_attributes_preserved():
    rows = [{"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
             "amount_inr": 555.5, "timestamp": _ts(), "transaction_type": "IMPS"}]
    graph = build_transaction_graph(_df(rows))
    edge = graph.get_edge_data("A", "B")["T1"]
    assert edge["amount_inr"] == 555.5
    assert edge["transaction_type"] == "IMPS"
    assert edge["transaction_id"] == "T1"


def test_extra_isolated_accounts_included_as_nodes_with_no_edges():
    rows = [{"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
             "amount_inr": 10.0, "timestamp": _ts(), "transaction_type": "UPI"}]
    graph = build_transaction_graph(_df(rows), extra_isolated_accounts=["Z"])
    assert "Z" in graph.nodes
    assert graph.degree("Z") == 0


def test_weighted_digraph_aggregates_multi_edges():
    rows = [
        {"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 100.0, "timestamp": _ts(), "transaction_type": "UPI"},
        {"transaction_id": "T2", "sender_account_id": "A", "receiver_account_id": "B",
         "amount_inr": 50.0, "timestamp": _ts(), "transaction_type": "UPI"},
    ]
    graph = build_transaction_graph(_df(rows))
    weighted = to_weighted_digraph(graph)
    assert weighted.number_of_edges() == 1  # collapsed
    assert weighted["A"]["B"]["weight"] == 150.0
    assert weighted["A"]["B"]["txn_count"] == 2


def test_self_transaction_would_create_self_loop_if_present():
    """Builder itself does not forbid self-loops -- rejection happens at
    validation time. This test documents that the builder is a dumb
    materialiser and does not silently drop or alter what it's given."""
    rows = [{"transaction_id": "T1", "sender_account_id": "A", "receiver_account_id": "A",
             "amount_inr": 10.0, "timestamp": _ts(), "transaction_type": "UPI"}]
    graph = build_transaction_graph(_df(rows))
    assert graph.has_edge("A", "A")
