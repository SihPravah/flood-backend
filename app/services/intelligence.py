from datetime import datetime, timezone
from typing import Any

from app.schemas.map import (
    Alert,
    CatchmentDetail,
    CityStatus,
    DrainDetail,
    FeatureCollection,
    MapIntelligenceResponse,
    MapLayers,
    MapSummary,
    RoadDetail,
    SensorDetail,
    SourceMetadata,
)
from app.schemas.routes import (
    NoSafeRouteResponse,
    RouteAlternative,
    RouteSegment,
    SafeRouteRequest,
    SafeRouteSuccess,
)
from app.services.errors import MLIntelligenceServiceError


class MLIntelligenceService:
    def build_map_intelligence(
        self,
        fused_state: dict[str, Any],
    ) -> MapIntelligenceResponse:
        raise NotImplementedError

    def get_catchment_detail(
        self,
        fused_state: dict[str, Any],
        catchment_id: str,
    ) -> CatchmentDetail:
        raise NotImplementedError

    def get_drain_detail(
        self,
        drain_id: str,
    ) -> DrainDetail:
        raise NotImplementedError

    def get_road_detail(
        self,
        road_id: str,
    ) -> RoadDetail:
        raise NotImplementedError

    def get_sensor_detail(
        self,
        device_id: str,
    ) -> SensorDetail:
        raise NotImplementedError

    def get_alerts(self) -> list[Alert]:
        raise NotImplementedError

    def plan_safe_route(
        self,
        request: SafeRouteRequest,
    ) -> SafeRouteSuccess | NoSafeRouteResponse:
        raise NotImplementedError


class DemoMLIntelligenceService(MLIntelligenceService):
    def build_map_intelligence(
        self,
        fused_state: dict[str, Any],
    ) -> MapIntelligenceResponse:
        generated_at = _state_time(fused_state)
        return MapIntelligenceResponse(
            snapshot_id=_snapshot_id(generated_at),
            generated_at=generated_at,
            mode="DEMO",
            data_label="SIMULATED",
            city=CityStatus(
                city_id="UK-DEHRADUN",
                name="Dehradun",
                operational_status="ELEVATED",
                confidence=0.74,
                reasons=[
                    "simulated_drain_overload",
                    "model_avoidance_present",
                ],
                last_updated=generated_at,
            ),
            layers=_demo_layers(),
            summary=MapSummary(
                catchment_count=1,
                high_risk_catchments=0,
                overflowing_drains=1,
                roads_to_avoid=1,
                confirmed_road_closures=0,
                active_alerts=1,
            ),
        )

    def get_catchment_detail(
        self,
        fused_state: dict[str, Any],
        catchment_id: str,
    ) -> CatchmentDetail:
        generated_at = _state_time(fused_state)
        return CatchmentDetail(
            catchment_id=catchment_id,
            snapshot_id=_snapshot_id(generated_at),
            risk_score=0.64,
            risk_level="WARNING",
            confidence=0.76,
            reasons=[
                "rainfall_increasing",
                "soil_saturation_high",
            ],
            provenance=_simulated_sources(
                ["SIM_NODE_04"],
                static_verification_status="ESTIMATED",
            ),
            last_updated=generated_at,
            fused_state="FusedCatchmentState v2.1",
            hydrology={
                "runoff_mm": 18.2,
                "concentration_time_minutes": 26.0,
            },
            rainfall=fused_state["rainfall"],
            soil=fused_state["soil"],
            anticipation={
                "trend": "RISING",
                "threshold_window": {
                    "risk_level": "HIGH",
                    "earliest_minutes": 30,
                    "latest_minutes": 60,
                },
            },
            landslide={
                "risk_level": "WATCH",
                "susceptibility_score": 0.42,
                "reasons": [
                    "steep_terrain",
                    "soil_saturation_high",
                ],
            },
        )

    def get_drain_detail(
        self,
        drain_id: str,
    ) -> DrainDetail:
        generated_at = _demo_time()
        return DrainDetail(
            drain_id=drain_id,
            snapshot_id=_snapshot_id(generated_at),
            risk_score=0.72,
            risk_level="HIGH",
            confidence=0.70,
            reasons=[
                "estimated_inflow_exceeds_effective_capacity"
            ],
            provenance=_simulated_sources(
                ["UK-CHM-DEHRADUN-01"],
                capacity_verification_status="ESTIMATED",
            ),
            last_updated=generated_at,
            inflow_m3_per_s=3.2,
            capacity_m3_per_s=2.6,
            capacity_utilization=1.23,
            overflow_m3_per_s=0.6,
            condition="ESTIMATED",
            affected_roads=["ROAD-FAST"],
        )

    def get_road_detail(
        self,
        road_id: str,
    ) -> RoadDetail:
        generated_at = _demo_time()
        is_closed = road_id == "ROAD-CLOSED"
        return RoadDetail(
            road_id=road_id,
            snapshot_id=_snapshot_id(generated_at),
            risk_score=0.81 if not is_closed else 0.95,
            risk_level="HIGH" if not is_closed else "SEVERE",
            confidence=0.73,
            reasons=(
                ["authority_confirmed_closure"]
                if is_closed
                else [
                    "nearby_drain_over_capacity",
                    "flood_risk_requires_avoidance",
                ]
            ),
            provenance=_simulated_sources(
                ["DRAIN-01"],
                static_verification_status="ESTIMATED",
            ),
            last_updated=generated_at,
            recommendation="CLOSED" if is_closed else "AVOID",
            associated_drain_id="DRAIN-01",
            authority_closed=is_closed,
            terrain={
                "depression_score": 0.65,
                "slope_fraction": 0.12,
            },
            historical_waterlogging_score=0.76,
            landslide_exposure={
                "risk_level": "WATCH",
                "susceptibility_score": 0.42,
            },
        )

    def get_sensor_detail(
        self,
        device_id: str,
    ) -> SensorDetail:
        generated_at = _demo_time()
        return SensorDetail(
            device_id=device_id,
            snapshot_id=_snapshot_id(generated_at),
            measurements={
                "rainfall_mm_per_hr": 48.0,
                "soil_moisture_percentage": 82.0,
            },
            observed_at=generated_at,
            age_minutes=2.0,
            freshness="GOOD",
            provenance=_simulated_sources([device_id]),
            last_updated=generated_at,
        )

    def get_alerts(self) -> list[Alert]:
        generated_at = _demo_time()
        return [
            Alert(
                alert_id="ALERT-DEMO-01",
                alert_type="FLOOD_WATCH",
                risk_level="WARNING",
                confidence=0.74,
                message=(
                    "Demo warning: runoff and drain utilization "
                    "are increasing in the simulated scenario."
                ),
                affected_entity_ids=[
                    "UK-CHM-DEHRADUN-01",
                    "DRAIN-01",
                    "ROAD-FAST",
                ],
                reasons=[
                    "simulated_rainfall_increase",
                    "simulated_drain_overload",
                ],
                provenance=_simulated_sources(
                    ["SIM_NODE_04"]
                ),
                issued_at=generated_at,
                last_updated=generated_at,
            )
        ]

    def plan_safe_route(
        self,
        request: SafeRouteRequest,
    ) -> SafeRouteSuccess | NoSafeRouteResponse:
        generated_at = _demo_time()
        if (
            request.destination.place_id
            == "DEMO-NO-SAFE-ROUTE"
        ):
            return NoSafeRouteResponse(
                snapshot_id=_snapshot_id(generated_at),
                generated_at=generated_at,
                reason_code="NO_ROUTABLE_PATH",
                message=(
                    "No route satisfies current closure and "
                    "avoidance constraints in this demo state."
                ),
                blocked_by=[
                    _route_segment(
                        "ROAD-FAST",
                        "AVOID",
                        0.81,
                        "HIGH",
                    ),
                    _route_segment(
                        "ROAD-CLOSED",
                        "CLOSED",
                        0.95,
                        "SEVERE",
                    ),
                ],
                provenance=_simulated_sources(
                    ["SIM_NODE_04", "DRAIN-01"]
                ),
                safety_note=(
                    "PRAVAHA is decision support and does not "
                    "guarantee route safety."
                ),
            )

        selected = RouteAlternative(
            route_id="ROUTE-SAFE-DEMO",
            label="Safest available demo route",
            strategy=request.strategy,
            travel_time_minutes=18.0,
            distance_km=6.4,
            maximum_risk_score=0.38,
            minimum_confidence=0.71,
            additional_time_vs_fastest_minutes=7.0,
            unsafe_segments_avoided=1,
            closures_avoided=0,
            explanation=[
                "avoids_model_avoid_road",
                "keeps_risk_and_confidence_separate",
            ],
            geometry={
                "type": "LineString",
                "coordinates": [
                    [78.0300, 30.3200],
                    [78.0405, 30.3270],
                    [78.0520, 30.3350],
                ],
            },
            segments=[
                _route_segment(
                    "ROAD-BYPASS",
                    "CAUTION",
                    0.38,
                    "WATCH",
                )
            ],
        )
        return SafeRouteSuccess(
            snapshot_id=_snapshot_id(generated_at),
            generated_at=generated_at,
            selected_route=selected,
            alternatives=[selected],
            provenance=_simulated_sources(
                ["SIM_NODE_04", "DRAIN-01"]
            ),
            safety_note=(
                "PRAVAHA is decision support and does not "
                "guarantee route safety."
            ),
        )


def _route_segment(
    road_id: str,
    recommendation: str,
    risk_score: float,
    risk_level: str,
) -> RouteSegment:
    return RouteSegment(
        road_id=road_id,
        recommendation=recommendation,
        risk_score=risk_score,
        risk_level=risk_level,
        confidence=0.72,
        reasons=[
            "demo_risk_evidence"
        ],
    )


def _demo_layers() -> MapLayers:
    return MapLayers(
        catchments=FeatureCollection(
            features=[
                _feature(
                    "UK-CHM-DEHRADUN-01",
                    "Polygon",
                    [
                        [
                            [78.028, 30.318],
                            [78.056, 30.318],
                            [78.056, 30.342],
                            [78.028, 30.342],
                            [78.028, 30.318],
                        ]
                    ],
                    {
                        "risk_level": "WARNING",
                        "confidence": 0.76,
                    },
                )
            ]
        ),
        drains=FeatureCollection(
            features=[
                _feature(
                    "DRAIN-01",
                    "LineString",
                    [
                        [78.0300, 30.3200],
                        [78.0330, 30.3230],
                    ],
                    {"risk_level": "HIGH"},
                )
            ]
        ),
        roads=FeatureCollection(
            features=[
                _feature(
                    "ROAD-FAST",
                    "LineString",
                    [
                        [78.0300, 30.3202],
                        [78.0330, 30.3232],
                    ],
                    {"recommendation": "AVOID"},
                )
            ]
        ),
        sensors=FeatureCollection(
            features=[
                _feature(
                    "SIM_NODE_04",
                    "Point",
                    [78.039, 30.329],
                    {"data_label": "SIMULATED"},
                )
            ]
        ),
    )


def _feature(
    feature_id: str,
    geometry_type: str,
    coordinates: Any,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": geometry_type,
            "coordinates": coordinates,
        },
        "properties": {
            "id": feature_id,
            **properties,
        },
    }


def _simulated_sources(
    sources: list[str],
    *,
    static_verification_status: str | None = None,
    capacity_verification_status: str | None = None,
) -> SourceMetadata:
    return SourceMetadata(
        data_label="SIMULATED",
        sources=sources,
        static_verification_status=(
            static_verification_status
        ),
        capacity_verification_status=(
            capacity_verification_status
        ),
        provider="backend-demo-adapter",
    )


def _state_time(
    fused_state: dict[str, Any],
) -> datetime:
    value = fused_state.get("state_time")
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    raise MLIntelligenceServiceError(
        "Fused state is missing a valid state_time."
    )


def _demo_time() -> datetime:
    return datetime(
        2026,
        9,
        9,
        8,
        45,
        tzinfo=timezone.utc,
    )


def _snapshot_id(
    generated_at: datetime,
) -> str:
    utc = generated_at.astimezone(
        timezone.utc
    )
    return "snap_" + utc.strftime("%Y%m%dT%H%M%SZ")

