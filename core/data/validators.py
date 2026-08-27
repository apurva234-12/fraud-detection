"""
Validation of raw transaction rows into canonical `Transaction` objects.

Core principle (explicitly required by the project brief): invalid financial
records are NEVER silently dropped. Every rejected row is captured with a
machine-readable reason code and a human-readable detail message in a rejects
report, so a reviewer can always answer "what happened to row X and why".

Validation is deliberately NOT fail-fast at the field level: a single row is
checked against every rule and *all* failing rules are recorded, so a bad
timestamp and a bad amount on the same row are both visible, not just the
first one encountered.

Cross-row rules (currently: duplicate transaction_id) are applied after
per-row checks, using transaction_id-first-seen-wins semantics: the first
occurrence of a transaction_id is validated normally; every subsequent
occurrence is rejected as DUPLICATE_ID regardless of whether its content is
otherwise valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Optional
import math

import pandas as pd
from dateutil import parser as date_parser
import pytz

from core.config import Config
from core.data.schema import REQUIRED_COLUMNS, Transaction, TransactionType


class RejectReason(str, Enum):
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_TRANSACTION_ID = "INVALID_TRANSACTION_ID"
    DUPLICATE_ID = "DUPLICATE_ID"
    INVALID_ACCOUNT_ID = "INVALID_ACCOUNT_ID"
    SELF_TRANSACTION = "SELF_TRANSACTION"
    NON_POSITIVE_AMOUNT = "NON_POSITIVE_AMOUNT"
    INVALID_AMOUNT_TYPE = "INVALID_AMOUNT_TYPE"
    MALFORMED_TIMESTAMP = "MALFORMED_TIMESTAMP"
    INVALID_TRANSACTION_TYPE = "INVALID_TRANSACTION_TYPE"


@dataclass
class RowValidationResult:
    is_valid: bool
    transaction: Optional[Transaction]
    reasons: list[RejectReason] = field(default_factory=list)
    details: list[str] = field(default_factory=list)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_amount(raw: Any, config: Config) -> tuple[Optional[float], list[RejectReason], list[str]]:
    reasons: list[RejectReason] = []
    details: list[str] = []
    try:
        amount = float(raw)
    except (TypeError, ValueError):
        reasons.append(RejectReason.INVALID_AMOUNT_TYPE)
        details.append(f"amount_inr {raw!r} is not numeric")
        return None, reasons, details

    if math.isnan(amount) or math.isinf(amount):
        reasons.append(RejectReason.INVALID_AMOUNT_TYPE)
        details.append(f"amount_inr {raw!r} is NaN/Inf")
        return None, reasons, details

    if amount < config.min_amount_inr:
        reasons.append(RejectReason.NON_POSITIVE_AMOUNT)
        details.append(f"amount_inr {amount} is <= 0 or below minimum {config.min_amount_inr}")
        return None, reasons, details

    return amount, reasons, details


def _parse_timestamp(raw: Any, config: Config) -> tuple[Optional[datetime], list[RejectReason], list[str]]:
    reasons: list[RejectReason] = []
    details: list[str] = []
    tz = pytz.timezone(config.timezone)

    if isinstance(raw, datetime):
        parsed = raw
    else:
        try:
            parsed = date_parser.parse(str(raw))
        except (ValueError, OverflowError, TypeError):
            reasons.append(RejectReason.MALFORMED_TIMESTAMP)
            details.append(f"timestamp {raw!r} could not be parsed")
            return None, reasons, details

    # Normalise to project timezone (IST). Naive timestamps are ASSUMED to
    # already be in the project timezone and are localised, not converted;
    # timezone-aware timestamps are converted.
    if parsed.tzinfo is None:
        parsed = tz.localize(parsed)
    else:
        parsed = parsed.astimezone(tz)

    return parsed, reasons, details


def validate_row(
    row: Mapping[str, Any],
    config: Config,
) -> RowValidationResult:
    """Validate a single raw row (e.g. one line of a CSV, or one dict) against
    every rule. Does NOT check for duplicate transaction_id -- that is a
    cross-row rule applied by `validate_batch`."""

    reasons: list[RejectReason] = []
    details: list[str] = []

    # 1. Required fields present at all (column existed) and non-missing.
    for col in REQUIRED_COLUMNS:
        if col not in row or _is_missing(row.get(col)):
            reasons.append(RejectReason.MISSING_FIELD)
            details.append(f"required field '{col}' is missing")

    if reasons:
        # Cannot safely proceed to type-specific checks if required fields
        # are absent -- but we still return everything found so far rather
        # than raising.
        return RowValidationResult(is_valid=False, transaction=None, reasons=reasons, details=details)

    transaction_id = row["transaction_id"]
    sender_id = row["sender_account_id"]
    receiver_id = row["receiver_account_id"]

    if not isinstance(transaction_id, str) or not transaction_id.strip():
        reasons.append(RejectReason.INVALID_TRANSACTION_ID)
        details.append(f"transaction_id {transaction_id!r} is not a non-empty string")

    if not isinstance(sender_id, str) or not sender_id.strip():
        reasons.append(RejectReason.INVALID_ACCOUNT_ID)
        details.append(f"sender_account_id {sender_id!r} is not a non-empty string")

    if not isinstance(receiver_id, str) or not receiver_id.strip():
        reasons.append(RejectReason.INVALID_ACCOUNT_ID)
        details.append(f"receiver_account_id {receiver_id!r} is not a non-empty string")

    if (
        isinstance(sender_id, str)
        and isinstance(receiver_id, str)
        and sender_id == receiver_id
        and not config.allow_self_transactions
    ):
        reasons.append(RejectReason.SELF_TRANSACTION)
        details.append(f"sender_account_id == receiver_account_id == {sender_id!r}")

    amount, amount_reasons, amount_details = _parse_amount(row["amount_inr"], config)
    reasons.extend(amount_reasons)
    details.extend(amount_details)

    timestamp, ts_reasons, ts_details = _parse_timestamp(row["timestamp"], config)
    reasons.extend(ts_reasons)
    details.extend(ts_details)

    raw_type = row["transaction_type"]
    txn_type: Optional[TransactionType] = None
    type_str = str(raw_type).strip().upper() if not _is_missing(raw_type) else ""
    if type_str not in config.allowed_transaction_types:
        reasons.append(RejectReason.INVALID_TRANSACTION_TYPE)
        details.append(f"transaction_type {raw_type!r} is not in allowed set {config.allowed_transaction_types}")
    else:
        txn_type = TransactionType(type_str)

    if reasons:
        return RowValidationResult(is_valid=False, transaction=None, reasons=reasons, details=details)

    transaction = Transaction(
        transaction_id=transaction_id,
        sender_account_id=sender_id,
        receiver_account_id=receiver_id,
        amount_inr=amount,  # type: ignore[arg-type]
        timestamp=timestamp,  # type: ignore[arg-type]
        transaction_type=txn_type,  # type: ignore[arg-type]
    )
    return RowValidationResult(is_valid=True, transaction=transaction, reasons=[], details=[])


@dataclass
class ValidationOutcome:
    valid_transactions: list[Transaction]
    rejects: pd.DataFrame  # columns: transaction_id, reasons, details, raw_row_index


def validate_batch(rows: Iterable[Mapping[str, Any]], config: Config) -> ValidationOutcome:
    """Validate a full batch of raw rows, including cross-row duplicate-ID
    detection. Row order is preserved; the first occurrence of a given
    transaction_id is the one considered for per-row validity."""

    valid_transactions: list[Transaction] = []
    reject_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for idx, row in enumerate(rows):
        result = validate_row(row, config)

        raw_id = row.get("transaction_id") if isinstance(row, Mapping) else None
        if result.is_valid:
            assert result.transaction is not None
            if result.transaction.transaction_id in seen_ids:
                reject_records.append(
                    {
                        "row_index": idx,
                        "transaction_id": result.transaction.transaction_id,
                        "reasons": [RejectReason.DUPLICATE_ID.value],
                        "details": [
                            f"transaction_id {result.transaction.transaction_id!r} duplicates an earlier row"
                        ],
                    }
                )
                continue
            seen_ids.add(result.transaction.transaction_id)
            valid_transactions.append(result.transaction)
        else:
            reject_records.append(
                {
                    "row_index": idx,
                    "transaction_id": raw_id,
                    "reasons": [r.value for r in result.reasons],
                    "details": result.details,
                }
            )

    rejects_df = pd.DataFrame(
        reject_records,
        columns=["row_index", "transaction_id", "reasons", "details"],
    )

    if config.raise_on_any_invalid and not rejects_df.empty:
        raise ValueError(
            f"{len(rejects_df)} row(s) failed validation and raise_on_any_invalid=True. "
            f"First failure: {rejects_df.iloc[0].to_dict()}"
        )

    return ValidationOutcome(valid_transactions=valid_transactions, rejects=rejects_df)
