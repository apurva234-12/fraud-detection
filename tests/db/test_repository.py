from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from core.config import Config
from core.data.schema import Transaction as CoreTransaction, TransactionType
from core.data.synthetic import generate_fixture_dataset
from core.data.validators import ValidationOutcome, validate_batch

# NOTE: imported from the specific submodules confirmed while inspecting the
# repo (core/features/transaction_features.py, referenced by name in
# core/graph/metrics.py's docstring; core/graph/builder.py). If
# core/features/__init__.py and core/graph/__init__.py turn out to re-export
# these at the package level, these two lines can be simplified -- but that
# re-export was never confirmed, so the explicit submodule path is used here
# rather than assumed.
from core.features.transaction_features import compute_account_features
from core.graph.builder import build_transaction_graph

from db import repository
from db.models import Account, Transaction


def _core_txn(**overrides) -> CoreTransaction:
    defaults = dict(
        transaction_id="TXN1",
        sender_account_id="ACC1",
        receiver_account_id="ACC2",
        amount_inr=100.0,
        timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        transaction_type=TransactionType.UPI,
    )
    defaults.update(overrides)
    return CoreTransaction(**defaults)


class TestUpsertAccounts:
    def test_creates_new_accounts(self, session):
        created = repository.upsert_accounts(session, ["ACC1", "ACC2", "ACC3"])
        assert created == 3
        assert session.query(Account).count() == 3

    def test_skips_existing_accounts(self, session):
        repository.upsert_accounts(session, ["ACC1"])
        created_second_time = repository.upsert_accounts(session, ["ACC1", "ACC2"])
        assert created_second_time == 1
        assert session.query(Account).count() == 2

    def test_empty_input_is_a_noop(self, session):
        assert repository.upsert_accounts(session, []) == 0


class TestInsertTransactions:
    def test_insert_single_transaction(self, session):
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        count = repository.insert_transactions(session, [_core_txn()])
        assert count == 1
        assert session.get(Transaction, "TXN1") is not None

    def test_amount_and_type_survive_the_conversion(self, session):
        """core.data.schema.Transaction.amount_inr is a float and
        transaction_type is an Enum -- confirms both convert correctly to
        the DB's Decimal / plain string representation."""
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        repository.insert_transactions(session, [_core_txn(amount_inr=1234.56)])
        row = session.get(Transaction, "TXN1")
        assert row.amount_inr == Decimal("1234.56")
        assert row.transaction_type == "UPI"

    def test_duplicate_transaction_id_across_calls_raises(self, session):
        """The scenario core.data.validators cannot catch on its own: the
        same transaction_id appearing in two SEPARATE batches/calls."""
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        repository.insert_transactions(session, [_core_txn()])

        with pytest.raises(IntegrityError):
            repository.insert_transactions(
                session, [_core_txn(amount_inr=999.0)]
            )

    def test_empty_list_is_a_noop(self, session):
        assert repository.insert_transactions(session, []) == 0


class TestInsertRejects:
    def test_insert_rejects_from_dataframe(self, session):
        rejects = pd.DataFrame(
            [
                {
                    "row_index": 0,
                    "transaction_id": "TXN1",
                    "reasons": ["NON_POSITIVE_AMOUNT"],
                    "details": ["amount_inr -5 is <= 0"],
                }
            ]
        )
        count = repository.insert_rejects(session, rejects)
        assert count == 1

    def test_empty_dataframe_is_a_noop(self, session):
        empty = pd.DataFrame(columns=["row_index", "transaction_id", "reasons", "details"])
        assert repository.insert_rejects(session, empty) == 0


class TestTransactionsDataframe:
    def test_round_trip_shape_matches_core_schema(self, session):
        """Read-path half of the integration seam: what we hand back to
        core.features / core.graph must be exactly the DataFrame shape
        `core.data.schema.transactions_to_dataframe` produces."""
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        repository.insert_transactions(session, [_core_txn()])
        session.flush()

        df = repository.transactions_dataframe(session)
        assert list(df.columns) == [
            "transaction_id",
            "sender_account_id",
            "receiver_account_id",
            "amount_inr",
            "timestamp",
            "transaction_type",
        ]
        assert len(df) == 1
        assert df.iloc[0]["amount_inr"] == 100.0
        assert isinstance(df.iloc[0]["amount_inr"], float)  # Decimal cast back to float

    def test_dataframe_feeds_directly_into_core_features(self, session):
        """Proves the read path actually works end-to-end with core/ code,
        not just that the column names match."""
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        repository.insert_transactions(
            session,
            [
                _core_txn(transaction_id="TXN1", sender_account_id="ACC1", receiver_account_id="ACC2"),
                _core_txn(transaction_id="TXN2", sender_account_id="ACC2", receiver_account_id="ACC1"),
            ],
        )
        session.flush()

        df = repository.transactions_dataframe(session)
        features = compute_account_features(df, Config())
        assert set(features["account_id"]) == {"ACC1", "ACC2"}

    def test_dataframe_feeds_directly_into_core_graph(self, session):
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        repository.insert_transactions(session, [_core_txn()])
        session.flush()

        df = repository.transactions_dataframe(session)
        graph = build_transaction_graph(df)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1


class TestGetIsolatedAccounts:
    def test_account_with_no_transactions_is_isolated(self, session):
        repository.upsert_accounts(session, ["ACC1", "ACC2", "ACC_ISOLATED"])
        repository.insert_transactions(session, [_core_txn()])  # ACC1 -> ACC2 only
        session.flush()

        isolated = repository.get_isolated_accounts(session)
        assert isolated == ["ACC_ISOLATED"]

    def test_no_isolated_accounts_returns_empty_list(self, session):
        repository.upsert_accounts(session, ["ACC1", "ACC2"])
        repository.insert_transactions(session, [_core_txn()])
        session.flush()
        assert repository.get_isolated_accounts(session) == []


class TestIngestValidationOutcome:
    def test_persists_valid_transactions_and_rejects_together(self, session):
        outcome = ValidationOutcome(
            valid_transactions=[_core_txn()],
            rejects=pd.DataFrame(
                [{"row_index": 1, "transaction_id": None, "reasons": ["MISSING_FIELD"], "details": ["x"]}]
            ),
        )
        batch = repository.ingest_validation_outcome(session, outcome, source_filename="test.csv")
        session.flush()

        assert batch.id is not None
        assert batch.valid_count == 1
        assert batch.reject_count == 1
        assert session.query(Transaction).count() == 1
        assert session.query(Account).count() == 2  # ACC1, ACC2 derived automatically

    def test_creates_accounts_before_transactions_so_fk_is_satisfied(self, session):
        """If account creation happened after (or not at all), this would
        fail with an IntegrityError on the transaction insert."""
        outcome = ValidationOutcome(
            valid_transactions=[_core_txn(sender_account_id="NEW1", receiver_account_id="NEW2")],
            rejects=pd.DataFrame(columns=["row_index", "transaction_id", "reasons", "details"]),
        )
        repository.ingest_validation_outcome(session, outcome)
        session.flush()  # should not raise
        assert session.get(Account, "NEW1") is not None
        assert session.get(Account, "NEW2") is not None


class TestIntegrationWithSyntheticFixture:
    """Feeds core.data.synthetic.generate_fixture_dataset(config) --
    Student A's own deterministic test fixture -- straight through the
    ingestion path into the database. This is the strongest available proof
    that db/ actually integrates with core/ correctly, using data neither of
    us hand-crafted for the other side.
    """

    def test_fixture_dataset_ingests_without_error(self, session, config: Config):
        dataset = generate_fixture_dataset(config)

        # generate_fixture_dataset produces schema-valid Transaction objects
        # directly (see docs/design_decisions.md §9); run them back through
        # validate_batch to get the ValidationOutcome shape ingest_validation_outcome
        # expects, exactly as a real pipeline run would.
        rows = [
            {
                "transaction_id": t.transaction_id,
                "sender_account_id": t.sender_account_id,
                "receiver_account_id": t.receiver_account_id,
                "amount_inr": t.amount_inr,
                "timestamp": t.timestamp,
                "transaction_type": t.transaction_type.value,
            }
            for t in dataset.transactions
        ]
        outcome = validate_batch(rows, config)
        assert outcome.rejects.empty  # generator output should always be schema-valid

        batch = repository.ingest_validation_outcome(
            session, outcome, source_filename="synthetic_fixture"
        )
        session.flush()

        assert batch.valid_count == len(dataset.transactions)
        assert session.query(Transaction).count() == len(dataset.transactions)

    def test_fixture_scenario_labels_persist_and_stay_isolated(self, session, config: Config):
        """Confirms transaction_scenario_labels / account_scenario_labels
        round-trip correctly and are queried as a SEPARATE table, never
        joined into transactions -- the isolation docs/design_decisions.md §9
        requires."""
        dataset = generate_fixture_dataset(config)
        rows = [
            {
                "transaction_id": t.transaction_id,
                "sender_account_id": t.sender_account_id,
                "receiver_account_id": t.receiver_account_id,
                "amount_inr": t.amount_inr,
                "timestamp": t.timestamp,
                "transaction_type": t.transaction_type.value,
            }
            for t in dataset.transactions
        ]
        outcome = validate_batch(rows, config)
        repository.ingest_validation_outcome(session, outcome, source_filename="synthetic_fixture")
        session.flush()

        repository.upsert_accounts(session, dataset.account_scenarios["account_id"].unique())
        label_count = repository.insert_transaction_scenario_labels(session, dataset.transaction_scenarios)
        account_label_count = repository.insert_account_scenario_labels(session, dataset.account_scenarios)

        assert label_count == len(dataset.transaction_scenarios)
        assert account_label_count == len(dataset.account_scenarios)

        # Confirm the base Transaction row itself carries NO scenario field --
        # the isolation the docs require is structural, not just a convention.
        sample_txn = session.get(Transaction, dataset.transactions[0].transaction_id)
        assert not hasattr(sample_txn, "scenario")
