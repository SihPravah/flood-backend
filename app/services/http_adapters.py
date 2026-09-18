from typing import Any

import httpx

from app.schemas.map import (
    Alert,
    CatchmentDetail,
    DrainDetail,
    LocationInspection,
    MapIntelligenceResponse,
    RoadDetail,
    SensorDetail,
    SourceHealth,
    StructuredEvent,
)
from app.schemas.routes import (
    NoSafeRouteResponse,
    SafeRouteRequest,
    SafeRouteSuccess,
)
from app.schemas.sensor import SensorIngestionPayload
from app.services.data_state import DataIngestionResult, DataStateService
from app.services.errors import DataStateServiceError, MLIntelligenceServiceError
from app.services.intelligence import MLIntelligenceService


class HTTPDataStateService(DataStateService):
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        retry_count: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count

    def ingest_sensor(
        self,
        payload: SensorIngestionPayload,
        *,
        received_at: Any,
    ) -> DataIngestionResult:
        request_payload = payload.model_dump(mode="json", by_alias=True)
        request_payload["received_at"] = received_at.isoformat()
        response = _request_json(
            "POST",
            f"{self.base_url}/api/v1/ingest/sensors",
            json=request_payload,
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count,
            error_type=DataStateServiceError,
        )
        return DataIngestionResult(
            catchment_id=response["catchment_id"],
            observed_at=payload.timestamp,
            received_at=received_at,
            canonical_location=response["canonical_location"],
            provenance=response.get("provenance", payload.provenance),
            fused_state=response["fused_state"],
            scenario_stage=_stage_from_fused_state(response["fused_state"]),
        )

    def get_fused_catchment_state(
        self,
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> dict[str, Any]:
        return _request_json(
            "GET",
            f"{self.base_url}/api/v1/catchments/{catchment_id}/state",
            params={"scenario_stage": scenario_stage},
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count,
            error_type=DataStateServiceError,
        )


class HTTPMLIntelligenceService(MLIntelligenceService):
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        retry_count: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count

    def build_map_intelligence(
        self,
        fused_state: dict[str, Any],
        scenario_stage: str = "WARNING",
    ) -> MapIntelligenceResponse:
        payload = self._request(
            "POST",
            "/api/v1/intelligence/map",
            json={"fused_state": fused_state, "scenario_stage": scenario_stage},
        )
        return MapIntelligenceResponse.model_validate(payload)

    def get_catchment_detail(
        self,
        fused_state: dict[str, Any],
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> CatchmentDetail:
        payload = self._request(
            "POST",
            f"/api/v1/intelligence/catchments/{catchment_id}",
            json={"fused_state": fused_state, "scenario_stage": scenario_stage},
        )
        return CatchmentDetail.model_validate(payload)

    def get_drain_detail(
        self,
        drain_id: str,
        scenario_stage: str = "WARNING",
    ) -> DrainDetail:
        payload = self._request(
            "GET",
            f"/api/v1/intelligence/drains/{drain_id}",
            params={"scenario_stage": scenario_stage},
        )
        return DrainDetail.model_validate(payload)

    def get_road_detail(
        self,
        road_id: str,
        scenario_stage: str = "WARNING",
    ) -> RoadDetail:
        payload = self._request(
            "GET",
            f"/api/v1/intelligence/roads/{road_id}",
            params={"scenario_stage": scenario_stage},
        )
        return RoadDetail.model_validate(payload)

    def get_sensor_detail(
        self,
        device_id: str,
        scenario_stage: str = "WARNING",
    ) -> SensorDetail:
        payload = self._request(
            "GET",
            f"/api/v1/intelligence/sensors/{device_id}",
            params={"scenario_stage": scenario_stage},
        )
        return SensorDetail.model_validate(payload)

    def inspect_location(
        self,
        *,
        longitude: float,
        latitude: float,
        scenario_stage: str = "WARNING",
    ) -> LocationInspection:
        payload = self._request(
            "GET",
            "/api/v1/intelligence/map/inspect",
            params={
                "longitude": longitude,
                "latitude": latitude,
                "scenario_stage": scenario_stage,
            },
        )
        return LocationInspection.model_validate(payload)

    def get_alerts(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[Alert]:
        payload = self._request(
            "GET",
            "/api/v1/intelligence/alerts",
            params={"scenario_stage": scenario_stage},
        )
        return [Alert.model_validate(item) for item in payload]

    def get_events(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[StructuredEvent]:
        payload = self._request(
            "GET",
            "/api/v1/intelligence/events",
            params={"scenario_stage": scenario_stage},
        )
        return [StructuredEvent.model_validate(item) for item in payload]

    def plan_safe_route(
        self,
        request: SafeRouteRequest,
        scenario_stage: str = "WARNING",
    ) -> SafeRouteSuccess | NoSafeRouteResponse:
        payload = self._request(
            "POST",
            "/api/v1/intelligence/routes/safe",
            params={"scenario_stage": scenario_stage},
            json=request.model_dump(mode="json", by_alias=True),
        )
        if payload.get("status") == "NO_SAFE_ROUTE":
            return NoSafeRouteResponse.model_validate(payload)
        return SafeRouteSuccess.model_validate(payload)

    def get_source_health(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[SourceHealth]:
        payload = self._request(
            "GET",
            "/api/v1/intelligence/source-health",
            params={"scenario_stage": scenario_stage},
        )
        return [SourceHealth.model_validate(item) for item in payload]

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        return _request_json(
            method,
            f"{self.base_url}{path}",
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count,
            error_type=MLIntelligenceServiceError,
            **kwargs,
        )


class UnavailableDataStateService(DataStateService):
    def ingest_sensor(
        self,
        payload: SensorIngestionPayload,
        *,
        received_at: Any,
    ) -> DataIngestionResult:
        raise DataStateServiceError(
            "Operational Data/IoT service is not configured; demo fallback is disabled."
        )

    def get_fused_catchment_state(
        self,
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> dict[str, Any]:
        raise DataStateServiceError(
            "Operational Data/IoT service is not configured; demo fallback is disabled."
        )


class UnavailableMLIntelligenceService(MLIntelligenceService):
    def build_map_intelligence(
        self,
        fused_state: dict[str, Any],
        scenario_stage: str = "WARNING",
    ) -> MapIntelligenceResponse:
        raise _unavailable()

    def get_catchment_detail(
        self,
        fused_state: dict[str, Any],
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> CatchmentDetail:
        raise _unavailable()

    def get_drain_detail(
        self,
        drain_id: str,
        scenario_stage: str = "WARNING",
    ) -> DrainDetail:
        raise _unavailable()

    def get_road_detail(
        self,
        road_id: str,
        scenario_stage: str = "WARNING",
    ) -> RoadDetail:
        raise _unavailable()

    def get_sensor_detail(
        self,
        device_id: str,
        scenario_stage: str = "WARNING",
    ) -> SensorDetail:
        raise _unavailable()

    def inspect_location(
        self,
        *,
        longitude: float,
        latitude: float,
        scenario_stage: str = "WARNING",
    ) -> LocationInspection:
        raise _unavailable()

    def get_alerts(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[Alert]:
        raise _unavailable()

    def get_events(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[StructuredEvent]:
        raise _unavailable()

    def plan_safe_route(
        self,
        request: SafeRouteRequest,
        scenario_stage: str = "WARNING",
    ) -> SafeRouteSuccess | NoSafeRouteResponse:
        raise _unavailable()

    def get_source_health(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[SourceHealth]:
        raise _unavailable()


def _request_json(
    method: str,
    url: str,
    *,
    timeout_seconds: float,
    retry_count: int,
    error_type: type[Exception],
    **kwargs: Any,
) -> Any:
    last_error: Exception | None = None
    attempts = max(retry_count + 1, 1)
    for _ in range(attempts):
        try:
            with httpx.Client(timeout=timeout_seconds) as client:
                response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc

    raise error_type(f"Service request failed for {url}: {last_error}") from last_error


def _unavailable() -> MLIntelligenceServiceError:
    return MLIntelligenceServiceError(
        "Operational ML intelligence service is not configured; demo fallback is disabled."
    )


def _stage_from_fused_state(
    fused_state: dict[str, Any],
) -> str:
    rainfall = (
        fused_state
        .get("rainfall", {})
        .get("intensity", {})
        .get("value")
    )
    soil = fused_state.get("soil", {}).get("saturation")
    rainfall_value = float(rainfall or 0.0)
    soil_percent = float(soil or 0.0) * 100.0
    if rainfall_value >= 70.0 or soil_percent >= 88.0:
        return "SEVERE"
    if rainfall_value >= 40.0 or soil_percent >= 75.0:
        return "WARNING"
    if rainfall_value >= 15.0 or soil_percent >= 55.0:
        return "WATCH"
    return "NORMAL"
