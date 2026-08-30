from datetime import datetime, timezone

from fastapi import APIRouter, status

from app.schemas.sensor import SensorIngestionPayload
from app.services.ml_client import ml_client

router = APIRouter()


@router.post(
    "/ingest/sensors",
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_sensor(payload: SensorIngestionPayload):
    received_at = datetime.now(timezone.utc)

    age_seconds = max(
        0,
        int((received_at - payload.timestamp).total_seconds()),
    )

    prediction = ml_client.predict(payload)

    return {
        "status": "accepted",
        "device_id": payload.device_id,
        "observed_at": payload.timestamp,
        "received_at": received_at,
        "age_seconds": age_seconds,
        "prediction": prediction.model_dump(),
    }