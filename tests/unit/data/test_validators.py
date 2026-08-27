from __future__ import annotations

from datetime import datetime

import pytest

from core.config import Config
from core.data.validators import RejectReason, validate_batch, validate_row


def _row(**overrides) -> dict:
    defaults = dict(
        transaction_id="TXN1",
        sender_account_id="ACC1",
        receiver_account_id="ACC2",
        amount_inr="100.0",
        timestamp="2024-01-01 10:00:00",
        transaction_type="UPI",
    )
    defaults.update(overrides)
    return defaults


class TestSingleRowValidation:
    def test_valid_row_passes(self, config: Config):
        result = validate_row(_row(), config)
        assert result.is_valid
        assert result.transaction is not None
        assert result.transaction.amount_inr == 100.0
        assert result.reasons == []

    def test_missing_field_rejected(self, config: Config):
        row = _row()
        del row["amount_inr"]
        result = validate_row(row, config)
        assert not result.is_valid
        assert RejectReason.MISSING_FIELD in result.reasons

    def test_none_field_treated_as_missing(self, config: Config):
        result = validate_row(_row(sender_account_id=None), config)
        assert not result.is_valid
        assert RejectReason.MISSING_FIELD in result.reasons

    def test_empty_string_field_treated_as_missing(self, config: Config):
        result = validate_row(_row(transaction_id="   "), config)
        assert not result.is_valid
        assert RejectReason.MISSING_FIELD in result.reasons

    def test_non_positive_amount_rejected(self, config: Config):
        result = validate_row(_row(amount_inr="0"), config)
        assert not result.is_valid
        assert RejectReason.NON_POSITIVE_AMOUNT in result.reasons

        result_negative = validate_row(_row(amount_inr="-50"), config)
        assert not result_negative.is_valid
        assert RejectReason.NON_POSITIVE_AMOUNT in result_negative.reasons

    def test_non_numeric_amount_rejected(self, config: Config):
        result = validate_row(_row(amount_inr="not-a-number"), config)
        assert not result.is_valid
        assert RejectReason.INVALID_AMOUNT_TYPE in result.reasons

    def test_malformed_timestamp_rejected(self, config: Config):
        result = validate_row(_row(timestamp="not-a-date"), config)
        assert not result.is_valid
        assert RejectReason.MALFORMED_TIMESTAMP in result.reasons

    def test_invalid_transaction_type_rejected(self, config: Config):
        result = validate_row(_row(transaction_type="BITCOIN_TRANSFER"), config)
        assert not result.is_valid
        assert RejectReason.INVALID_TRANSACTION_TYPE in result.reasons

    def test_self_transaction_rejected_by_default(self, config: Config):
        result = validate_row(_row(sender_account_id="ACC1", receiver_account_id="ACC1"), config)
        assert not result.is_valid
        assert RejectReason.SELF_TRANSACTION in result.reasons

    def test_self_transaction_allowed_when_configured(self):
        cfg = Config(allow_self_transactions=True)
        result = validate_row(_row(sender_account_id="ACC1", receiver_account_id="ACC1"), cfg)
        assert result.is_valid

    def test_multiple_failures_all_collected(self, config: Config):
        # Bad amount AND bad timestamp on the same row -- both must surface,
        # not just the first one encountered.
        result = validate_row(_row(amount_inr="-1", timestamp="garbage"), config)
        assert not result.is_valid
        assert RejectReason.NON_POSITIVE_AMOUNT in result.reasons
        assert RejectReason.MALFORMED_TIMESTAMP in result.reasons

    def test_naive_timestamp_localised_to_configured_timezone(self, config: Config):
        result = validate_row(_row(timestamp="2024-06-15 12:00:00"), config)
        assert result.is_valid
        assert result.transaction is not None
        assert str(result.transaction.timestamp.tzinfo) == "Asia/Kolkata"

    def test_datetime_object_input_accepted(self, config: Config):
        result = validate_row(_row(timestamp=datetime(2024, 1, 1, 10, 0, 0)), config)
        assert result.is_valid


class TestBatchValidation:
    def test_empty_batch(self, config: Config):
        outcome = validate_batch([], config)
        assert outcome.valid_transactions == []
        assert outcome.rejects.empty

    def test_single_valid_row(self, config: Config):
        outcome = validate_batch([_row()], config)
        assert len(outcome.valid_transactions) == 1
        assert outcome.rejects.empty

    def test_duplicate_transaction_id_rejected(self, config: Config):
        rows = [_row(transaction_id="TXN1"), _row(transaction_id="TXN1", amount_inr="999")]
        outcome = validate_batch(rows, config)
        assert len(outcome.valid_transactions) == 1
        assert outcome.valid_transactions[0].amount_inr == 100.0  # first occurrence wins
        assert len(outcome.rejects) == 1
        assert RejectReason.DUPLICATE_ID.value in outcome.rejects.iloc[0]["reasons"]

    def test_invalid_rows_never_silently_dropped(self, config: Config):
        rows = [_row(transaction_id="TXN1"), _row(transaction_id="TXN2", amount_inr="-5")]
        outcome = validate_batch(rows, config)
        assert len(outcome.valid_transactions) == 1
        assert len(outcome.rejects) == 1
        assert outcome.rejects.iloc[0]["transaction_id"] == "TXN2"

    def test_raise_on_any_invalid(self):
        cfg = Config(raise_on_any_invalid=True)
        rows = [_row(transaction_id="TXN1", amount_inr="-5")]
        with pytest.raises(ValueError):
            validate_batch(rows, cfg)

    def test_repeated_valid_transactions_between_same_pair_all_kept(self, config: Config):
        rows = [_row(transaction_id=f"TXN{i}") for i in range(5)]
        outcome = validate_batch(rows, config)
        assert len(outcome.valid_transactions) == 5
        assert outcome.rejects.empty
