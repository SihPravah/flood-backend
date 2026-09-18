from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings
from app.schemas.sensor import SensorIngestionPayload
from app.services.errors import DataStateServiceError
from app.services.intelligence import (
    CATCHMENT_ID,
    PROFILES,
    SENSOR_PRIMARY_ID,
    SENSOR_SECONDARY_ID,
    ScenarioStage,
)


@dataclass(frozen=True)
class DataIngestionResult:
    catchment_id: str
    observed_at: datetime
    received_at: datetime
    canonical_location: dict[str, float | str]
    provenance: str
    fused_state: dict[str, Any]
    scenario_stage: ScenarioStage


class DataStateService:
    def ingest_sensor(
        self,
        payload: SensorIngestionPayload,
        *,
        received_at: datetime,
    ) -> DataIngestionResult:
        raise NotImplementedError

    def get_fused_catchment_state(
        self,
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> dict[str, Any]:
        raise NotImplementedError


class DemoDataStateService(DataStateService):
    def get_fused_catchment_state(
        self,
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> dict[str, Any]:
        if not settings.demo_mode:
            raise DataStateServiceError(
                "Demo fused-state fallback is disabled in operational mode."
            )

        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = datetime(
            2026,
            9,
            9,
            8,
            0,
            tzinfo=timezone.utc,
        ) + timedelta(minutes=profile.offset_minutes)

        return {
            "catchment_id": catchment_id or CATCHMENT_ID,
            "state_time": generated_at.isoformat(),
            "rainfall": {
                "intensity": {
                    "value": profile.rainfall_intensity_mm_per_hr,
                    "status": "SIMULATED",
                    "confidence": profile.confidence,
                    "age_minutes": 2.0,
                },
                "rain_15m": _window(round(profile.rain_1h_mm / 4, 2)),
                "rain_30m": _window(round(profile.rain_1h_mm / 2, 2)),
                "rain_1h": _window(profile.rain_1h_mm),
                "rain_3h": _window(profile.rain_3h_mm),
                "rain_6h": _window(profile.rain_3h_mm + 26.0),
                "rain_24h": _window(profile.rain_24h_mm),
            },
            "soil": {
                "saturation": profile.soil_saturation,
                "status": "SIMULATED",
                "confidence": profile.confidence,
                "age_minutes": 3.0,
            },
            "data_quality": {
                "overall_score": 0.90 if stage != "SEVERE" else 0.74,
                "missing_sources": []
                if stage != "SEVERE"
                else [SENSOR_SECONDARY_ID],
                "temporal_freshness": "GOOD"
                if stage in {"NORMAL", "WATCH", "WARNING"}
                else "DEGRADED",
            },
        }

    def ingest_sensor(
        self,
        payload: SensorIngestionPayload,
        *,
        received_at: datetime,
    ) -> DataIngestionResult:
        if payload.provenance == "SIMULATED" and not settings.demo_mode:
            raise DataStateServiceError(
                "Operational mode cannot silently ingest SIMULATED sensor data."
            )

        stage = _stage_from_sensor_payload(payload)
        fused_state = _fused_state_from_sensor_payload(
            payload,
            stage=stage,
            received_at=received_at,
        )
        return DataIngestionResult(
            catchment_id=fused_state["catchment_id"],
            observed_at=payload.timestamp,
            received_at=received_at,
            canonical_location=payload.canonical_location,
            provenance=payload.provenance,
            fused_state=fused_state,
            scenario_stage=stage,
        )


def _window(
    value_mm: float,
    *,
    status: str = "DERIVED",
    quality: str = "GOOD",
) -> dict[str, Any]:
    return {
        "value_mm": value_mm,
        "status": status,
        "coverage_fraction": 1.0,
        "largest_gap_minutes": 5.0,
        "latest_observation_age_minutes": 2.0,
        "observation_count": 6,
        "quality": quality,
    }


def _stage(value: str) -> ScenarioStage:
    normalized = value.upper()
    if normalized not in PROFILES:
        raise DataStateServiceError(f"Unknown demo scenario stage: {value}")
    return normalized  # type: ignore[return-value]


def _stage_from_sensor_payload(
    payload: SensorIngestionPayload,
) -> ScenarioStage:
    rainfall = payload.sensor_metrics.rainfall_mm_per_hr
    soil = payload.sensor_metrics.soil_moisture_percentage
    if rainfall >= 70.0 or soil >= 88.0:
        return "SEVERE"
    if rainfall >= 40.0 or soil >= 75.0:
        return "WARNING"
    if rainfall >= 15.0 or soil >= 55.0:
        return "WATCH"
    return "NORMAL"


def _fused_state_from_sensor_payload(
    payload: SensorIngestionPayload,
    *,
    stage: ScenarioStage,
    received_at: datetime,
) -> dict[str, Any]:
    profile = PROFILES[stage]
    rainfall = payload.sensor_metrics.rainfall_mm_per_hr
    soil = payload.sensor_metrics.soil_moisture_percentage / 100.0
    age_minutes = max(
        0.0,
        (received_at - payload.timestamp).total_seconds() / 60.0,
    )
    temporal_freshness = "GOOD" if age_minutes <= 15.0 else "DEGRADED"
    quality = temporal_freshness
    source_status = payload.provenance

    return {
        "catchment_id": CATCHMENT_ID,
        "state_time": payload.timestamp.isoformat(),
        "rainfall": {
            "intensity": {
                "value": rainfall,
                "status": source_status,
                "confidence": 0.95 if source_status == "OBSERVED" else profile.confidence,
                "age_minutes": round(age_minutes, 1),
            },
            "rain_15m": _window(round(rainfall / 4.0, 2), quality=quality),
            "rain_30m": _window(round(rainfall / 2.0, 2), quality=quality),
            "rain_1h": _window(round(rainfall, 2), quality=quality),
            "rain_3h": _window(round(max(rainfall, profile.rain_3h_mm), 2), quality=quality),
            "rain_6h": _window(round(max(rainfall + 24.0, profile.rain_3h_mm + 26.0), 2), quality=quality),
            "rain_24h": _window(round(max(rainfall + 80.0, profile.rain_24h_mm), 2), quality=quality),
        },
        "soil": {
            "saturation": round(soil, 3),
            "status": source_status,
            "confidence": 0.90 if source_status == "OBSERVED" else profile.confidence,
            "age_minutes": round(age_minutes, 1),
        },
        "data_quality": {
            "overall_score": 0.92
            if source_status == "OBSERVED" and temporal_freshness == "GOOD"
            else profile.confidence,
            "missing_sources": [],
            "temporal_freshness": temporal_freshness,
        },
    }
