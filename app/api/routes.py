from fastapi import APIRouter, Depends, HTTPException

from app.schemas.routes import (
    NoSafeRouteResponse,
    SafeRouteRequest,
    SafeRouteSuccess,
)
from app.services.dependencies import (
    get_ml_intelligence_service,
)
from app.services.errors import MLIntelligenceServiceError
from app.services.intelligence import MLIntelligenceService

router = APIRouter()


@router.post(
    "/routes/safe",
    response_model=SafeRouteSuccess | NoSafeRouteResponse,
    summary="Plan a risk-aware route without claiming guaranteed safety.",
)
def safe_route(
    request: SafeRouteRequest,
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    try:
        return ml_service.plan_safe_route(request)
    except MLIntelligenceServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

