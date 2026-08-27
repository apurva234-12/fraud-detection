from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from api.deps import get_db
from db import repository
from core.config import Config
from core.data.synthetic import generate_fixture_dataset
from core.data.validators import validate_batch

app = FastAPI(
    title="Fraudulent Transaction Detection API",
    description="API for graph-based anomaly detection and AML risk analysis.",
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/transactions")
def list_transactions(db: Session = Depends(get_db)):
    df = repository.transactions_dataframe(db)
    return df.to_dict(orient="records")
@app.post("/ingest/synthetic")
def ingest_synthetic_data(db: Session = Depends(get_db)):
    config = Config()
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

    batch = repository.ingest_validation_outcome(
        db,
        outcome,
        source_filename="synthetic_fixture_api"
    )

    db.commit()

    return {
        "batch_id": batch.id,
        "valid_count": batch.valid_count,
        "reject_count": batch.reject_count,
    }