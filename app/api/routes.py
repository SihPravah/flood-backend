from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.schemas.routes import (
    NoSafeRouteResponse,
    SafeRouteRequest,
    SafeRouteSuccess,
)
from app.services.dependencies import (
    get_ml_intelligence_service,
    get_monitoring_state_store,
)
from app.services.errors import MLIntelligenceServiceError
from app.services.intelligence import MLIntelligenceService
from app.services.state_store import MonitoringStateStore

router = APIRouter()


@router.post(
    "/routes/safe",
    response_model=SafeRouteSuccess | NoSafeRouteResponse,
    summary="Plan a risk-aware route without claiming guaranteed safety.",
)
def safe_route(
    request: SafeRouteRequest,
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = scenario_stage or state_store.latest_scenario_stage(
        default=settings.demo_stage
    )
    try:
        route_result = ml_service.plan_safe_route(request, stage)
        state_store.record_route_evaluation(route_result)
        return route_result
    except MLIntelligenceServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
