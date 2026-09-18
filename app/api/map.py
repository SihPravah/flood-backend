from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.schemas.map import (
    Alert,
    CatchmentDetail,
    DrainDetail,
    LocationInspection,
    MapIntelligenceResponse,
    RoadDetail,
    SensorDetail,
    StructuredEvent,
)
from app.services.data_state import DataStateService
from app.services.dependencies import (
    get_data_state_service,
    get_ml_intelligence_service,
    get_monitoring_state_store,
)
from app.services.errors import (
    DataStateServiceError,
    MLIntelligenceServiceError,
)
from app.services.intelligence import MLIntelligenceService
from app.services.state_store import MonitoringStateStore

router = APIRouter()


@router.get(
    "/map/intelligence",
    response_model=MapIntelligenceResponse,
    summary="Return the current PRAVAHA city map snapshot.",
)
def map_intelligence(
    scenario_stage: str | None = Query(default=None),
    data_service: DataStateService = Depends(
        get_data_state_service
    ),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    if scenario_stage is None:
        latest_snapshot = state_store.latest_snapshot()
        if latest_snapshot is not None:
            return MapIntelligenceResponse.model_validate(latest_snapshot)

    stage = _active_stage(scenario_stage, state_store)
    fused_state = _get_fused_state(
        data_service,
        settings.default_catchment_id,
        stage,
    )
    try:
        snapshot = ml_service.build_map_intelligence(
            fused_state,
            stage,
        )
        state_store.record_snapshot(snapshot, scenario_stage=stage)
        state_store.record_events(snapshot.events)
        return snapshot
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/catchments/{catchment_id}",
    response_model=CatchmentDetail,
    summary="Return catchment-level risk, confidence and reasons.",
)
def catchment_detail(
    catchment_id: str,
    scenario_stage: str | None = Query(default=None),
    data_service: DataStateService = Depends(
        get_data_state_service
    ),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = _active_stage(scenario_stage, state_store)
    fused_state = _get_fused_state(
        data_service,
        catchment_id,
        stage,
    )
    try:
        return ml_service.get_catchment_detail(
            fused_state,
            catchment_id,
            stage,
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/drains/{drain_id}",
    response_model=DrainDetail,
)
def drain_detail(
    drain_id: str,
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = _active_stage(scenario_stage, state_store)
    try:
        return ml_service.get_drain_detail(
            drain_id,
            stage,
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/roads/{road_id}",
    response_model=RoadDetail,
)
def road_detail(
    road_id: str,
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = _active_stage(scenario_stage, state_store)
    try:
        return ml_service.get_road_detail(
            road_id,
            stage,
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/sensors/{device_id}",
    response_model=SensorDetail,
)
def sensor_detail(
    device_id: str,
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = _active_stage(scenario_stage, state_store)
    try:
        return ml_service.get_sensor_detail(
            device_id,
            stage,
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/alerts",
    response_model=list[Alert],
)
def alerts(
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = _active_stage(scenario_stage, state_store)
    try:
        return ml_service.get_alerts(stage)
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/map/inspect",
    response_model=LocationInspection,
    summary="Inspect real/static GIS context for a map coordinate.",
)
def inspect_location(
    longitude: float = Query(ge=-180.0, le=180.0),
    latitude: float = Query(ge=-90.0, le=90.0),
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    stage = _active_stage(scenario_stage, state_store)
    try:
        return ml_service.inspect_location(
            longitude=longitude,
            latitude=latitude,
            scenario_stage=stage,
        )
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


@router.get(
    "/events",
    response_model=list[StructuredEvent],
)
def events(
    scenario_stage: str | None = Query(default=None),
    ml_service: MLIntelligenceService = Depends(
        get_ml_intelligence_service
    ),
    state_store: MonitoringStateStore = Depends(get_monitoring_state_store),
):
    if scenario_stage is None and state_store.recent_events():
        return [
            StructuredEvent.model_validate(event)
            for event in state_store.recent_events()
        ]

    stage = _active_stage(scenario_stage, state_store)
    try:
        current_events = ml_service.get_events(stage)
        state_store.record_events(current_events)
        return current_events
    except MLIntelligenceServiceError as exc:
        raise _service_unavailable(exc) from exc


def _get_fused_state(
    data_service: DataStateService,
    catchment_id: str,
    scenario_stage: str,
) -> dict:
    try:
        return data_service.get_fused_catchment_state(
            catchment_id,
            scenario_stage,
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


def _active_stage(
    scenario_stage: str | None,
    state_store: MonitoringStateStore,
) -> str:
    if scenario_stage is not None:
        return scenario_stage
    return state_store.latest_scenario_stage(default=settings.demo_stage)
