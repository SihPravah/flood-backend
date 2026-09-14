from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.schemas.map import (
    Alert,
    CatchmentDetail,
    DrainDetail,
    MapIntelligenceResponse,
    RoadDetail,
    SensorDetail,
)
from app.services.data_state import DataStateService
from app.services.dependencies import (
    get_data_state_service,
    get_ml_intelligence_service,
)
from app.services.errors import (
    DataStateServiceError,
    MLIntelligenceServiceError,
)
from app.services.intelligence import MLIntelligenceService

router = APIRouter()


@router.get(
    "/map/intelligence",
    response_model=MapIntelligenceResponse,
    summary="Return the current PRAVAHA city map snapshot.",
)
def map_intelligence(
    data_service: DataStateService = Depends(
        get_data_state_service
    ),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    fused_state = _get_fused_state(
        data_service,
        settings.default_catchment_id,
    )
    try:
        return ml_service.build_map_intelligence(
            fused_state
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/catchments/{catchment_id}",
    response_model=CatchmentDetail,
    summary="Return catchment-level risk, confidence and reasons.",
)
def catchment_detail(
    catchment_id: str,
    data_service: DataStateService = Depends(
        get_data_state_service
    ),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    fused_state = _get_fused_state(
        data_service,
        catchment_id,
    )
    try:
        return ml_service.get_catchment_detail(
            fused_state,
            catchment_id,
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/drains/{drain_id}",
    response_model=DrainDetail,
)
def drain_detail(
    drain_id: str,
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    try:
        return ml_service.get_drain_detail(
            drain_id
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/roads/{road_id}",
    response_model=RoadDetail,
)
def road_detail(
    road_id: str,
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    try:
        return ml_service.get_road_detail(
            road_id
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/sensors/{device_id}",
    response_model=SensorDetail,
)
def sensor_detail(
    device_id: str,
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    try:
        return ml_service.get_sensor_detail(
            device_id
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/alerts",
    response_model=list[Alert],
)
def alerts(
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
):
    try:
        return ml_service.get_alerts()
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


def _get_fused_state(
    data_service: DataStateService,
    catchment_id: str,
) -> dict:
    try:
        return data_service.get_fused_catchment_state(
            catchment_id
        )
    except DataStateServiceError as exc:
        raise _service_unavailable(exc) from exc


def _service_unavailable(
    exc: Exception,
) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=str(exc),
    )

