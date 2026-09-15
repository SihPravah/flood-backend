from fastapi import APIRouter, Depends, Query

from app.core.config import settings
from app.services.dependencies import (
    get_ml_intelligence_service,
    get_monitoring_state_store,
)
from app.services.intelligence import MLIntelligenceService
from app.services.state_store import MonitoringStateStore

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "pravaha-backend",
    }


@router.get("/ready")
def readiness():
    return {
        "status": "ready",
        "service": "pravaha-backend",
        "demo_mode": settings.demo_mode,
        "data_service_configured": bool(settings.data_service_url),
        "ml_service_configured": bool(settings.ml_service_url),
    }


@router.get("/system/health")
def system_health(
    scenario_stage: str = Query(default=settings.demo_stage),
    ml_service: MLIntelligenceService = Depends(get_ml_intelligence_service),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    source_health = ml_service.get_source_health(scenario_stage)
    unavailable = [
        source
        for source in source_health
        if source.status == "UNAVAILABLE"
    ]
    degraded = [
        source
        for source in source_health
        if source.status == "DEGRADED"
    ]
    status = "degraded" if unavailable or degraded else "healthy"
    return {
        "status": status,
        "service": "pravaha-backend",
        "demo_mode": settings.demo_mode,
        "scenario_id": "DEMO-001" if settings.demo_mode else None,
        "scenario_stage": scenario_stage.upper(),
        "backend": {"status": "HEALTHY"},
        "data": {"status": status.upper()},
        "ml": {
            "status": "SIMULATED" if settings.demo_mode else "HEALTHY",
            "model_state": "DEVELOPMENT_FALLBACK" if settings.demo_mode else "CONFIGURED",
        },
        "source_health": source_health,
        "state_store": state_store.summary(),
    }
