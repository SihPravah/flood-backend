from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.config import settings
from app.schemas.prediction import PredictionResponse
from app.schemas.sensor import SensorIngestionPayload
from app.services.data_state import DataStateService
from app.services.dependencies import (
    get_data_state_service,
    get_ml_intelligence_service,
    get_monitoring_state_store,
)
from app.services.errors import DataStateServiceError, MLIntelligenceServiceError
from app.services.intelligence import MLIntelligenceService
from app.services.state_store import MonitoringStateStore

router = APIRouter()


@router.post(
    "/ingest/sensors",
    status_code=status.HTTP_202_ACCEPTED,
)
def ingest_sensor(
    payload: SensorIngestionPayload,
    scenario_stage: str | None = Query(default=None),
    data_service: DataStateService = Depends(get_data_state_service),
    ml_service: MLIntelligenceService = Depends(get_ml_intelligence_service),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    received_at = payload.received_at or datetime.now(timezone.utc)
    payload = payload.model_copy(update={"received_at": received_at})

    age_seconds = max(
        0,
        int((received_at - payload.timestamp).total_seconds()),
    )

    try:
        ingestion = data_service.ingest_sensor(
            payload,
            received_at=received_at,
        )
        stage = scenario_stage or ingestion.scenario_stage
        snapshot = ml_service.build_map_intelligence(
            ingestion.fused_state,
            stage,
        )
    except DataStateServiceError as exc:
        status_code = (
            status.HTTP_400_BAD_REQUEST
            if "SIMULATED" in str(exc)
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except MLIntelligenceServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    state_store.record_snapshot(snapshot, scenario_stage=stage)
    state_store.record_events(snapshot.events)

    prediction = PredictionResponse(
        device_id=payload.device_id,
        timestamp=payload.timestamp,
        risk_score=snapshot.model_metadata.risk_score,
        risk_level=snapshot.model_metadata.risk_level,
        confidence=snapshot.model_metadata.confidence,
        prediction_mode=snapshot.model_metadata.runtime_status,
        model_version=snapshot.model_metadata.model_version,
    )

    return {
        "status": "accepted",
        "device_id": payload.device_id,
        "catchment_id": ingestion.catchment_id,
        "observed_at": payload.timestamp,
        "received_at": received_at,
        "age_seconds": age_seconds,
        "provenance": ingestion.provenance,
        "canonical_location": ingestion.canonical_location,
        "fused_state": ingestion.fused_state,
        "prediction": prediction.model_dump(),
        "snapshot_id": snapshot.snapshot_id,
        "city": snapshot.city,
        "source_health": snapshot.source_health,
        "mode": "DEMO" if settings.demo_mode else "OPERATIONAL",
    }
