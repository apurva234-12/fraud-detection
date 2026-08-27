from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError

from db.models import Account, IngestionBatch, Transaction, TransactionReject


def _make_transaction(**overrides) -> Transaction:
    defaults = dict(
        transaction_id="TXN1",
        sender_account_id="ACC1",
        receiver_account_id="ACC2",
        amount_inr=Decimal("100.00"),
        timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        transaction_type="UPI",
    )
    defaults.update(overrides)
    return Transaction(**defaults)


class TestAccountModel:
    def test_create_account(self, session):
        session.add(Account(account_id="ACC1"))
        session.flush()
        fetched = session.get(Account, "ACC1")
        assert fetched is not None
        assert fetched.account_id == "ACC1"

    def test_duplicate_account_id_rejected(self, session):
        session.add(Account(account_id="ACC1"))
        session.flush()
        session.add(Account(account_id="ACC1"))
        with pytest.raises(IntegrityError):
            session.flush()


class TestTransactionModel:
    def test_create_valid_transaction(self, session):
        session.add_all([Account(account_id="ACC1"), Account(account_id="ACC2")])
        session.flush()
        session.add(_make_transaction())
        session.flush()

        fetched = session.get(Transaction, "TXN1")
        assert fetched is not None
        assert fetched.amount_inr == Decimal("100.00")
        assert fetched.transaction_type == "UPI"

    def test_duplicate_transaction_id_rejected(self, session):
        """Global uniqueness enforcement -- the one thing core.data.validators
        does NOT guarantee across batches (see docs/design_decisions.md §5)."""
        session.add_all([Account(account_id="ACC1"), Account(account_id="ACC2")])
        session.flush()
        session.add(_make_transaction())
        session.flush()

        session.add(_make_transaction(amount_inr=Decimal("999.00")))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_non_positive_amount_rejected_by_check_constraint(self, session):
        session.add_all([Account(account_id="ACC1"), Account(account_id="ACC2")])
        session.flush()
        session.add(_make_transaction(amount_inr=Decimal("0.00")))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_disallowed_transaction_type_rejected_by_check_constraint(self, session):
        session.add_all([Account(account_id="ACC1"), Account(account_id="ACC2")])
        session.flush()
        session.add(_make_transaction(transaction_type="BITCOIN_TRANSFER"))
        with pytest.raises(IntegrityError):
            session.flush()

    def test_foreign_key_to_nonexistent_account_rejected(self, session):
        """sender_account_id / receiver_account_id must reference an existing
        account -- proves the FK constraint is actually enforced, not just
        declared."""
        session.add(_make_transaction())  # no accounts created first
        with pytest.raises(IntegrityError):
            session.flush()

    def test_all_six_allowed_transaction_types_accepted(self, session):
        session.add_all([Account(account_id="ACC1"), Account(account_id="ACC2")])
        session.flush()
        allowed = ("UPI", "NEFT", "IMPS", "RTGS", "CASH_DEPOSIT", "CASH_WITHDRAWAL")
        for i, txn_type in enumerate(allowed):
            session.add(_make_transaction(transaction_id=f"TXN{i}", transaction_type=txn_type))
        session.flush()  # should not raise
        count = session.query(Transaction).count()
        assert count == len(allowed)


class TestIngestionBatchModel:
    def test_create_batch_with_config_snapshot(self, session):
        batch = IngestionBatch(
            source_filename="synthetic_fixture.csv",
            valid_count=10,
            reject_count=2,
            config_snapshot={"random_seed": 42, "timezone": "Asia/Kolkata"},
        )
        session.add(batch)
        session.flush()
        assert batch.id is not None
        fetched = session.get(IngestionBatch, batch.id)
        assert fetched.config_snapshot["random_seed"] == 42

    def test_transactions_link_back_to_batch(self, session):
        session.add_all([Account(account_id="ACC1"), Account(account_id="ACC2")])
        batch = IngestionBatch(source_filename="f.csv", valid_count=1, reject_count=0)
        session.add(batch)
        session.flush()

        session.add(_make_transaction(ingested_batch_id=batch.id))
        session.flush()

        fetched_batch = session.get(IngestionBatch, batch.id)
        assert len(fetched_batch.transactions) == 1
        assert fetched_batch.transactions[0].transaction_id == "TXN1"


class TestTransactionRejectModel:
    def test_reject_with_null_transaction_id(self, session):
        """A row can be rejected before a transaction_id was even readable
        (e.g. MISSING_FIELD on transaction_id itself) -- transaction_id must
        be nullable here, unlike on Transaction."""
        reject = TransactionReject(
            row_index=0,
            transaction_id=None,
            reasons=["MISSING_FIELD"],
            details=["required field 'transaction_id' is missing"],
        )
        session.add(reject)
        session.flush()
        assert reject.id is not None

    def test_reject_reasons_is_a_list(self, session):
        """A row can fail multiple rules at once -- reasons/details must
        round-trip as lists, not a flattened string."""
        reject = TransactionReject(
            row_index=1,
            transaction_id="TXN2",
            reasons=["SELF_TRANSACTION", "NON_POSITIVE_AMOUNT", "MALFORMED_TIMESTAMP"],
            details=["detail a", "detail b", "detail c"],
        )
        session.add(reject)
        session.flush()

        fetched = session.get(TransactionReject, reject.id)
        assert fetched.reasons == ["SELF_TRANSACTION", "NON_POSITIVE_AMOUNT", "MALFORMED_TIMESTAMP"]
        assert len(fetched.details) == 3

    def test_multiple_rejects_can_share_a_duplicate_transaction_id(self, session):
        """Unlike Transaction.transaction_id, TransactionReject.transaction_id
        is NOT unique -- duplicate-ID rejects are expected."""
        session.add_all(
            [
                TransactionReject(row_index=0, transaction_id="TXN1", reasons=["DUPLICATE_ID"], details=["d1"]),
                TransactionReject(row_index=1, transaction_id="TXN1", reasons=["DUPLICATE_ID"], details=["d2"]),
            ]
        )
        session.flush()  # should not raise
        count = session.query(TransactionReject).filter_by(transaction_id="TXN1").count()
        assert count == 2
