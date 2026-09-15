from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import heapq
from typing import Any, Literal

from app.schemas.map import (
    Alert,
    CatchmentDetail,
    CityStatus,
    DrainDetail,
    FeatureCollection,
    LocationInspection,
    MapIntelligenceResponse,
    MapLayers,
    MapSummary,
    ModelMetadata,
    RoadDetail,
    SensorDetail,
    SourceHealth,
    SourceMetadata,
    StructuredEvent,
)
from app.schemas.routes import (
    NoSafeRouteResponse,
    RouteAlternative,
    RouteSegment,
    SafeRouteRequest,
    SafeRouteSuccess,
)
from app.services.errors import MLIntelligenceServiceError
from app.services import gis_context as gis


ScenarioStage = Literal["NORMAL", "WATCH", "WARNING", "SEVERE"]

SCENARIO_ID = "DEMO-001"
CITY_ID = "UK-DEHRADUN"
CATCHMENT_ID = "UK-CHM-DEHRADUN-01"
VILLAGE_ID = "VILLAGE-CHANDRABANI"
DRAIN_ID = "D-22"
ROAD_DIRECT_ID = "ROAD-SHELTER-CORRIDOR"
ROAD_BYPASS_ID = "ROAD-HIGHER-GROUND-BYPASS"
ROAD_CLOSED_ID = "ROAD-BRIDGE-APPROACH"
ROAD_HILLSIDE_ID = "ROAD-TRANSPORT-NAGAR-CONNECTOR"
SENSOR_PRIMARY_ID = "SENSOR-SIM-RAIN-SOIL-01"
SENSOR_SECONDARY_ID = "SENSOR-SIM-RAIN-SOIL-02"
SHELTER_ID = "SHELTER-SCHOOL-01"
LANDSLIDE_ZONE_ID = "LANDSLIDE-ZONE-S-01"
ROUTE_ID = "ROUTE-DEMO-001"
NO_SAFE_DESTINATION_ID = "DEMO-NO-SAFE-ROUTE"

BASE_TIME = datetime(2026, 9, 9, 8, 0, tzinfo=timezone.utc)
STAGE_ORDER: tuple[ScenarioStage, ...] = (
    "NORMAL",
    "WATCH",
    "WARNING",
    "SEVERE",
)


@dataclass(frozen=True)
class StageProfile:
    offset_minutes: int
    city_status: str
    risk_score: float
    risk_level: str
    confidence: float
    rainfall_intensity_mm_per_hr: float
    rain_1h_mm: float
    rain_3h_mm: float
    rain_24h_mm: float
    soil_saturation: float
    runoff_mm: float
    discharge_m3_s: float
    drain_utilization: float
    predicted_drain_utilization_30m: float
    overflow_m3_s: float
    road_recommendation: str
    latest_threshold_crossing: str | None
    reasons: tuple[str, ...]


PROFILES: dict[ScenarioStage, StageProfile] = {
    "NORMAL": StageProfile(
        0,
        "NORMAL",
        0.18,
        "LOW",
        0.84,
        4.0,
        3.0,
        8.0,
        22.0,
        0.34,
        1.2,
        0.36,
        0.42,
        0.44,
        0.0,
        "PASSABLE",
        None,
        ("rainfall_light", "drains_below_capacity", "soil_not_saturated"),
    ),
    "WATCH": StageProfile(
        30,
        "ELEVATED",
        0.38,
        "WATCH",
        0.79,
        18.0,
        18.0,
        38.0,
        64.0,
        0.56,
        8.1,
        1.12,
        0.76,
        0.88,
        0.0,
        "CAUTION",
        "+60 min WARNING",
        ("rainfall_increasing", "drain_capacity_tightening", "soil_moisture_rising"),
    ),
    "WARNING": StageProfile(
        60,
        "WARNING",
        0.64,
        "WARNING",
        0.74,
        42.0,
        48.0,
        92.0,
        145.0,
        0.82,
        19.1,
        3.07,
        1.18,
        1.31,
        0.42,
        "AVOID",
        "+30 min HIGH",
        (
            "rainfall_increasing",
            "soil_saturation_high",
            "downstream_drain_over_capacity",
        ),
    ),
    "SEVERE": StageProfile(
        90,
        "EMERGENCY",
        0.88,
        "SEVERE",
        0.71,
        70.0,
        84.0,
        168.0,
        270.0,
        0.96,
        38.7,
        5.94,
        1.56,
        1.72,
        1.2,
        "AVOID",
        "NOW SEVERE",
        (
            "severe_runoff",
            "drain_overload",
            "route_viability_degraded",
            "landslide_susceptibility_high",
        ),
    ),
}


@dataclass(frozen=True)
class DemoEdge:
    start: str
    end: str
    road_id: str
    time_minutes: float
    distance_km: float
    risk_score: float
    risk_level: str
    recommendation: str
    confidence: float
    reasons: tuple[str, ...]
    coordinates: tuple[tuple[float, float], ...]


class MLIntelligenceService:
    def build_map_intelligence(
        self,
        fused_state: dict[str, Any],
        scenario_stage: str = "WARNING",
    ) -> MapIntelligenceResponse:
        raise NotImplementedError

    def get_catchment_detail(
        self,
        fused_state: dict[str, Any],
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> CatchmentDetail:
        raise NotImplementedError

    def get_drain_detail(
        self,
        drain_id: str,
        scenario_stage: str = "WARNING",
    ) -> DrainDetail:
        raise NotImplementedError

    def get_road_detail(
        self,
        road_id: str,
        scenario_stage: str = "WARNING",
    ) -> RoadDetail:
        raise NotImplementedError

    def get_sensor_detail(
        self,
        device_id: str,
        scenario_stage: str = "WARNING",
    ) -> SensorDetail:
        raise NotImplementedError

    def get_alerts(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[Alert]:
        raise NotImplementedError

    def inspect_location(
        self,
        *,
        longitude: float,
        latitude: float,
        scenario_stage: str = "WARNING",
    ) -> LocationInspection:
        raise NotImplementedError

    def get_events(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[StructuredEvent]:
        raise NotImplementedError

    def plan_safe_route(
        self,
        request: SafeRouteRequest,
        scenario_stage: str = "WARNING",
    ) -> SafeRouteSuccess | NoSafeRouteResponse:
        raise NotImplementedError

    def get_source_health(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[SourceHealth]:
        raise NotImplementedError


class DemoMLIntelligenceService(MLIntelligenceService):
    def build_map_intelligence(
        self,
        fused_state: dict[str, Any],
        scenario_stage: str = "WARNING",
    ) -> MapIntelligenceResponse:
        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = _state_time(fused_state)
        source_health = self.get_source_health(stage)
        events = self.get_events(stage)
        summary = MapSummary(
            catchment_count=1,
            high_risk_catchments=1 if profile.risk_score >= 0.70 else 0,
            overflowing_drains=1 if profile.overflow_m3_s > 0.0 else 0,
            roads_to_avoid=1 if profile.road_recommendation == "AVOID" else 0,
            confirmed_road_closures=1 if stage == "SEVERE" else 0,
            active_alerts=len(self.get_alerts(stage)),
            highest_risk_catchment="Chandrabani focused micro-catchment",
            highest_risk_ward="Chandrabani settlement point",
            shelters_available=1,
            exposed_population=850 if stage == "SEVERE" else None,
            source_health=source_health,
            latest_threshold_crossing=profile.latest_threshold_crossing,
        )
        return MapIntelligenceResponse(
            snapshot_id=_snapshot_id(stage, generated_at),
            generated_at=generated_at,
            state_time=generated_at,
            scenario_id=SCENARIO_ID,
            mode="DEMO",
            data_label="SIMULATED",
            city=CityStatus(
                city_id=CITY_ID,
                name="Dehradun",
                district="Chandrabani, Dehradun",
                operational_status=profile.city_status,
                confidence=profile.confidence,
                reasons=list(profile.reasons),
                last_updated=generated_at,
            ),
            layers=_demo_layers(stage),
            summary=summary,
            source_health=source_health,
            events=events,
            model_metadata=_model_metadata(stage, generated_at, profile),
            study_area=gis.STUDY_AREA,
        )

    def get_catchment_detail(
        self,
        fused_state: dict[str, Any],
        catchment_id: str,
        scenario_stage: str = "WARNING",
    ) -> CatchmentDetail:
        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = _state_time(fused_state)
        return CatchmentDetail(
            type="catchment",
            catchment_id=catchment_id,
            name="Chandrabani focused micro-catchment",
            ward_name="Chandrabani settlement context",
            snapshot_id=_snapshot_id(stage, generated_at),
            risk_score=profile.risk_score,
            risk_level=profile.risk_level,
            confidence=profile.confidence,
            reasons=list(profile.reasons),
            provenance=_simulated_sources(
                [SENSOR_PRIMARY_ID],
                static_verification_status="ESTIMATED",
                terrain_source_status=gis.TERRAIN["derived_status"],
                terrain_source=gis.TERRAIN["source_name"],
                catchment_geometry_status="ESTIMATED",
            ),
            last_updated=generated_at,
            fused_state="FusedCatchmentState v2.1",
            status=profile.city_status,
            trend="RISING" if stage != "NORMAL" else "STABLE",
            hydrology={
                "runoff_mm": profile.runoff_mm,
                "discharge_m3_per_s": profile.discharge_m3_s,
                "runoff_coefficient": round(
                    profile.runoff_mm / max(profile.rain_1h_mm, 1.0),
                    2,
                ),
                "concentration_time_minutes": 26.0,
                "response": (
                    "Fast response, drainage demand exceeds estimated capacity"
                    if profile.drain_utilization >= 1.0
                    else "Response within monitored reserve"
                ),
                "drainage_demand": (
                    "Exceeds D-22 capacity"
                    if profile.drain_utilization >= 1.0
                    else "Within D-22 reserve"
                ),
            },
            rainfall=fused_state["rainfall"],
            rainfall_windows=_rainfall_windows(profile),
            soil=fused_state["soil"],
            terrain={
                "elevation_m": gis.TERRAIN["mean_elevation_m"],
                "min_elevation_m": gis.TERRAIN["min_elevation_m"],
                "max_elevation_m": gis.TERRAIN["max_elevation_m"],
                "mean_slope_deg": gis.TERRAIN["mean_slope_deg"],
                "mean_slope_fraction": gis.TERRAIN["mean_slope_fraction"],
                "catchment_area_km2": gis.STUDY_AREA["approx_area_km2"],
                "curve_number": 79,
                "hand_m": None,
                "twi": None,
                "terrain_source": gis.TERRAIN["source_name"],
                "source_status": gis.TERRAIN["derived_status"],
            },
            anticipation={
                "trend": "RISING" if stage != "NORMAL" else "STABLE",
                "threshold_window": None
                if stage == "NORMAL"
                else {
                    "risk_level": "WARNING"
                    if stage == "WATCH"
                    else profile.risk_level,
                    "earliest_minutes": 0 if stage == "SEVERE" else 30,
                    "latest_minutes": 60 if stage == "WATCH" else 30,
                },
                "timeline": _timeline(stage),
            },
            landslide={
                "risk_level": "HIGH" if stage == "SEVERE" else "WATCH",
                "susceptibility_score": round(
                    min(0.18 + profile.soil_saturation * 0.62, 1.0),
                    2,
                ),
                "reasons": ["steep_terrain", "soil_saturation_increasing"],
            },
            cascade=_cascade(profile),
            impact={
                "affected_wards": [VILLAGE_ID],
                "exposed_roads": [ROAD_DIRECT_ID, ROAD_CLOSED_ID],
                "threatened_shelters": [SHELTER_ID]
                if stage == "SEVERE"
                else [],
                "exposed_population": 850 if stage == "SEVERE" else None,
                "evacuation_readiness": (
                    "Needs route review before movement"
                    if profile.road_recommendation == "AVOID"
                    else "Route monitoring sufficient in demo"
                ),
            },
            provenance_table=_provenance_rows(profile),
        )

    def get_drain_detail(
        self,
        drain_id: str,
        scenario_stage: str = "WARNING",
    ) -> DrainDetail:
        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = _demo_time(stage)
        return DrainDetail(
            type="drain",
            drain_id=drain_id,
            name="D-22 estimated municipal collector",
            drain_type="estimated municipal collector aligned to OSM stream corridor",
            snapshot_id=_snapshot_id(stage, generated_at),
            risk_score=round(min(profile.drain_utilization / 1.6, 1.0), 2),
            risk_level="HIGH" if profile.drain_utilization >= 1.0 else "WATCH",
            confidence=profile.confidence,
            reasons=[
                "estimated_inflow_exceeds_effective_capacity",
                "upstream_runoff_rising",
            ]
            if profile.overflow_m3_s > 0
            else ["drain_utilization_below_capacity"],
            provenance=_simulated_sources(
                [CATCHMENT_ID],
                capacity_verification_status="ESTIMATED",
                geometry_source_status="ESTIMATED",
                geometry_basis="OSM stream corridor; not verified municipal storm drain",
            ),
            last_updated=generated_at,
            upstream_nodes=["D-22-U1", "D-22-U2"],
            downstream_node="D-22-OUT",
            inflow_m3_per_s=round(profile.drain_utilization * 2.6, 2),
            capacity_m3_per_s=2.6,
            capacity_utilization=profile.drain_utilization,
            predicted_utilization_30m=profile.predicted_drain_utilization_30m,
            overflow_m3_per_s=profile.overflow_m3_s,
            overflow_margin_m3_per_s=round(
                2.6 - profile.drain_utilization * 2.6,
                2,
            ),
            condition="ESTIMATED",
            condition_factor=0.82,
            affected_roads=[ROAD_DIRECT_ID],
            contributing_catchments=[CATCHMENT_ID],
            nearby_settlements=[VILLAGE_ID],
            timeline=_timeline(stage),
            provenance_table=[
                row
                for row in _provenance_rows(profile)
                if row["variable"] in {"rainfall", "runoff", "drain capacity"}
            ],
        )

    def get_road_detail(
        self,
        road_id: str,
        scenario_stage: str = "WARNING",
    ) -> RoadDetail:
        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = _demo_time(stage)
        is_bypass = road_id == ROAD_BYPASS_ID
        is_closed = road_id == ROAD_CLOSED_ID and stage == "SEVERE"
        road_static = gis.ROADS.get(road_id, {})
        recommendation = (
            "CLOSED"
            if is_closed
            else "CAUTION"
            if is_bypass and stage != "NORMAL"
            else "PASSABLE"
            if is_bypass
            else profile.road_recommendation
        )
        risk_level = (
            "SEVERE" if is_closed else "WATCH" if is_bypass else profile.risk_level
        )
        risk_score = (
            0.95
            if is_closed
            else 0.34
            if is_bypass
            else min(profile.risk_score + 0.12, 1.0)
        )
        return RoadDetail(
            type="road",
            road_id=road_id,
            name=str(road_static.get("name") or _road_name(road_id)),
            road_class=str(road_static.get("road_class") or "mapped road"),
            segment_length_km=_road_length_km(road_id),
            jurisdiction="Chandrabani focused study area, Dehradun",
            snapshot_id=_snapshot_id(stage, generated_at),
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=profile.confidence,
            reasons=["authority_confirmed_closure"]
            if is_closed
            else [
                "nearby_drain_over_capacity",
                "flood_risk_requires_avoidance",
            ]
            if recommendation == "AVOID"
            else ["route_segment_requires_monitoring"],
            provenance=_simulated_sources(
                [DRAIN_ID],
                road_verification_status=road_static.get(
                    "source_status",
                    "ESTIMATED",
                ),
                road_source="OpenStreetMap",
                road_source_osm_id=road_static.get("source_osm_id"),
                hazard_status_basis=(
                    "DEMO_AUTHORITY_CLOSURE_ONLY"
                    if is_closed
                    else "MODEL_RECOMMENDATION"
                ),
            ),
            last_updated=generated_at,
            recommendation=recommendation,
            status_basis="AUTHORITY_CONFIRMED"
            if is_closed
            else "MODEL_RECOMMENDATION",
            associated_drain_id=DRAIN_ID,
            authority_closed=is_closed,
            terrain={
                "depression_score": 0.12 if is_bypass else 0.68,
                "stream_proximity_m": _nearest_stream_distance_m(road_id),
                "mean_slope_deg": gis.TERRAIN["mean_slope_deg"],
                "mean_slope_fraction": gis.TERRAIN["mean_slope_fraction"],
                "terrain_source_status": gis.TERRAIN["derived_status"],
            },
            historical_waterlogging_score=0.08 if is_bypass else 0.76,
            landslide_exposure={
                "risk_level": "HIGH" if stage == "SEVERE" else "WATCH",
                "susceptibility_score": 0.72 if stage == "SEVERE" else 0.38,
            },
            contributors=[
                _metric(
                    "Catchment flood risk",
                    round(profile.risk_score * 100),
                    "%",
                    profile.risk_level,
                ),
                _metric(
                    "Nearby drain utilization",
                    round(profile.drain_utilization * 100),
                    "%",
                    "HIGH" if profile.drain_utilization >= 1.0 else "WATCH",
                ),
                _metric(
                    "Stream proximity",
                    _nearest_stream_distance_m(road_id),
                    "m",
                    "DERIVED",
                ),
                _metric(
                    "Historical waterlogging",
                    8 if is_bypass else 76,
                    "%",
                    "ESTIMATED",
                ),
            ],
            related_infrastructure=[
                _metric("Associated drain", DRAIN_ID, None, "ESTIMATED"),
                _metric("Nearest shelter", SHELTER_ID, None, "SIMULATED"),
                _metric("Alternative road", ROAD_BYPASS_ID, None, "OPEN_REAL_DATA"),
            ],
            anticipation=[
                _metric(
                    "Possible degradation",
                    "Not available" if stage == "NORMAL" else "+30 min",
                    None,
                    "SIMULATED",
                ),
                _metric("Expected recovery", "Not available", None, "MISSING"),
            ],
            routing_effect=[
                _metric(
                    "Current route usage",
                    "Avoided" if recommendation == "AVOID" else "Eligible",
                    None,
                    recommendation,
                ),
                _metric(
                    "Added time caused",
                    7 if recommendation == "AVOID" else 0,
                    "min",
                    "SIMULATED",
                ),
            ],
        )

    def get_sensor_detail(
        self,
        device_id: str,
        scenario_stage: str = "WARNING",
    ) -> SensorDetail:
        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = _demo_time(stage)
        missing = device_id == SENSOR_SECONDARY_ID and stage == "SEVERE"
        return SensorDetail(
            type="sensor",
            device_id=device_id,
            sensor_type="rainfall_soil_tilt_node",
            snapshot_id=_snapshot_id(stage, generated_at),
            latitude=30.285029 if device_id == SENSOR_PRIMARY_ID else 30.2854,
            longitude=77.978689 if device_id == SENSOR_PRIMARY_ID else 77.9926,
            source="backend-demo-data-iot-adapter",
            measurements={
                "rainfall_mm_per_hr": None
                if missing
                else profile.rainfall_intensity_mm_per_hr,
                "soil_moisture_percentage": None
                if missing
                else round(profile.soil_saturation * 100),
                "slope_tilt_degrees": None
                if missing
                else round(1.2 + profile.soil_saturation * 2.4, 1),
            },
            observed_at=generated_at - timedelta(minutes=74 if missing else 2),
            age_minutes=74 if missing else 2,
            freshness="UNUSABLE" if missing else "GOOD",
            status="MISSING" if missing else "SIMULATED",
            missing_fields=[
                "rainfall_mm_per_hr",
                "soil_moisture_percentage",
                "slope_tilt_degrees",
            ]
            if missing
            else [],
            provenance=_simulated_sources([device_id]),
            last_updated=generated_at,
            history=_stage_history(stage),
        )

    def get_alerts(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[Alert]:
        stage = _stage(scenario_stage)
        if stage == "NORMAL":
            return []
        profile = PROFILES[stage]
        generated_at = _demo_time(stage)
        alerts = [
            Alert(
                alert_id=f"ALERT-{SCENARIO_ID}-{stage}-FLOOD",
                alert_type="flash_flood",
                risk_level=profile.risk_level,
                severity=profile.risk_level,
                confidence=profile.confidence,
                location="Chandrabani focused micro-catchment",
                message=(
                    f"{profile.risk_level} demo flood intelligence for the "
                    "Chandrabani settlement corridor."
                ),
                recommended_review=(
                    "Inspect AVOID roads and drainage overload before dispatch."
                    if stage in {"WARNING", "SEVERE"}
                    else "Review D-22 and prepare route monitoring."
                ),
                affected_entity_ids=[CATCHMENT_ID, DRAIN_ID, ROAD_DIRECT_ID],
                reasons=list(profile.reasons),
                provenance=_simulated_sources([SENSOR_PRIMARY_ID, DRAIN_ID]),
                issued_at=generated_at,
                last_updated=generated_at,
            )
        ]
        if profile.overflow_m3_s > 0.0:
            alerts.append(
                Alert(
                    alert_id=f"ALERT-{SCENARIO_ID}-{stage}-DRAIN",
                    alert_type="drainage_overload",
                    risk_level=profile.risk_level,
                    severity=profile.risk_level,
                    confidence=profile.confidence,
                    location="Drain D-22",
                    message="D-22 exceeds effective capacity in the deterministic demo scenario.",
                    recommended_review="Review roads influenced by D-22 before approving movement.",
                    affected_entity_ids=[DRAIN_ID, ROAD_DIRECT_ID],
                    reasons=["estimated_inflow_exceeds_effective_capacity"],
                    provenance=_simulated_sources(
                        [CATCHMENT_ID, SENSOR_PRIMARY_ID],
                        capacity_verification_status="ESTIMATED",
                    ),
                    issued_at=generated_at,
                    last_updated=generated_at,
                )
            )
        return alerts

    def inspect_location(
        self,
        *,
        longitude: float,
        latitude: float,
        scenario_stage: str = "WARNING",
    ) -> LocationInspection:
        stage = _stage(scenario_stage)
        profile = PROFILES[stage]
        generated_at = _demo_time(stage)
        nearest_road = gis.nearest_named(
            longitude,
            latitude,
            gis.ROADS,
        )
        nearest_stream = gis.nearest_named(
            longitude,
            latitude,
            gis.STREAMS,
        )
        nearest_settlement = gis.nearest_named(
            longitude,
            latitude,
            gis.SETTLEMENTS,
        )
        nearest_asset = gis.nearest_named(
            longitude,
            latitude,
            gis.CRITICAL_ASSETS,
        )
        elevation = (
            gis.nearest_elevation_m(longitude, latitude)
            if gis.within_study_area(longitude, latitude)
            else None
        )

        return LocationInspection(
            id="clicked-location",
            snapshot_id=_snapshot_id(stage, generated_at),
            latitude=latitude,
            longitude=longitude,
            jurisdiction=(
                "Chandrabani, Dehradun"
                if gis.within_study_area(longitude, latitude)
                else "Outside focused study area"
            ),
            ward_or_village=_entity_name(gis.SETTLEMENTS, nearest_settlement),
            catchment_id=CATCHMENT_ID
            if gis.within_study_area(longitude, latitude)
            else None,
            nearest_road=_entity_name(gis.ROADS, nearest_road),
            nearest_stream=_entity_name(gis.STREAMS, nearest_stream),
            nearest_drain=DRAIN_ID if nearest_stream is not None else None,
            nearest_shelter=gis.SHELTER["name"],
            terrain=[
                _metric(
                    "Nearest SRTM elevation",
                    elevation,
                    "m",
                    gis.TERRAIN["source_status"]
                    if elevation is not None
                    else "NOT_AVAILABLE",
                ),
                _metric(
                    "Mean catchment slope",
                    gis.TERRAIN["mean_slope_deg"],
                    "deg",
                    gis.TERRAIN["derived_status"],
                ),
                _metric("HAND", None, "m", "NOT_AVAILABLE"),
                _metric("TWI", None, None, "NOT_AVAILABLE"),
            ],
            hydrology=[
                _metric("Catchment", CATCHMENT_ID, None, "ESTIMATED"),
                _metric(
                    "Nearest stream distance",
                    _distance_value(nearest_stream),
                    "m",
                    "OPEN_REAL_DATA",
                ),
                _metric(
                    "Municipal drain capacity",
                    "Not measured",
                    None,
                    "ESTIMATED",
                ),
            ],
            hazard_context=[
                _metric(
                    "Current flood risk",
                    round(profile.risk_score * 100),
                    "%",
                    profile.risk_level,
                ),
                _metric(
                    "Landslide inventory",
                    "Not available",
                    None,
                    "NOT_AVAILABLE",
                ),
                _metric(
                    "Nearest critical asset",
                    _entity_name(gis.CRITICAL_ASSETS, nearest_asset),
                    None,
                    "OPEN_REAL_DATA",
                ),
            ],
            data_quality=[
                _metric(
                    "Terrain source",
                    gis.TERRAIN["source_name"],
                    None,
                    gis.TERRAIN["derived_status"],
                ),
                _metric(
                    "Road/stream source",
                    "OpenStreetMap",
                    None,
                    "OPEN_REAL_DATA",
                ),
                _metric(
                    "Shelter designation",
                    gis.SHELTER["designation_status"],
                    None,
                    "DEMO",
                ),
                _metric(
                    "Dynamic conditions",
                    SCENARIO_ID,
                    None,
                    "SIMULATED",
                ),
            ],
        )

    def get_events(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[StructuredEvent]:
        stage = _stage(scenario_stage)
        events: list[StructuredEvent] = []
        for index, item in enumerate(
            STAGE_ORDER[: STAGE_ORDER.index(stage) + 1],
            start=1,
        ):
            profile = PROFILES[item]
            timestamp = _demo_time(item)
            events.append(
                StructuredEvent(
                    event_id=f"EVENT-{SCENARIO_ID}-{index:03d}",
                    snapshot_id=_snapshot_id(stage, _demo_time(stage)),
                    generated_at=timestamp,
                    event_type="RISK_LEVEL_CHANGED",
                    entity_type="catchment",
                    entity_id=CATCHMENT_ID,
                    previous_value=None
                    if index == 1
                    else PROFILES[STAGE_ORDER[index - 2]].risk_level,
                    current_value=profile.risk_level,
                    severity=profile.risk_level,
                    title=f"{profile.risk_level} catchment risk",
                    message=f"Catchment risk is {profile.risk_level} in {SCENARIO_ID}.",
                    reasons=list(profile.reasons),
                    provenance=_simulated_sources([SENSOR_PRIMARY_ID, DRAIN_ID]),
                )
            )
        if stage in {"WARNING", "SEVERE"}:
            events.append(
                StructuredEvent(
                    event_id=f"EVENT-{SCENARIO_ID}-ROAD-AVOID",
                    snapshot_id=_snapshot_id(stage, _demo_time(stage)),
                    generated_at=_demo_time("WARNING"),
                    event_type="ROAD_RECOMMENDATION_CHANGED",
                    entity_type="road",
                    entity_id=ROAD_DIRECT_ID,
                    previous_value="CAUTION",
                    current_value="AVOID",
                    severity="WARNING",
                    title="Road marked AVOID by model",
                    message="Road recommendation changed to AVOID because D-22 exceeded estimated capacity.",
                    reasons=["estimated_inflow_exceeds_effective_capacity", "avoid_not_closed"],
                    provenance=_simulated_sources([SENSOR_PRIMARY_ID, DRAIN_ID]),
                )
            )
        if stage == "SEVERE":
            events.append(
                StructuredEvent(
                    event_id=f"EVENT-{SCENARIO_ID}-AUTHORITY-CLOSURE",
                    snapshot_id=_snapshot_id(stage, _demo_time(stage)),
                    generated_at=_demo_time("SEVERE"),
                    event_type="AUTHORITY_CLOSURE",
                    entity_type="road",
                    entity_id=ROAD_CLOSED_ID,
                    previous_value="CAUTION",
                    current_value="CLOSED",
                    severity="SEVERE",
                    title="Authority closure active",
                    message=(
                        "Mapped unnamed connector is explicitly CLOSED by a "
                        "demo authority-closure fixture."
                    ),
                    reasons=["authority_confirmed_closure"],
                    provenance=_simulated_sources([ROAD_CLOSED_ID]),
                )
            )
        return events

    def plan_safe_route(
        self,
        request: SafeRouteRequest,
        scenario_stage: str = "WARNING",
    ) -> SafeRouteSuccess | NoSafeRouteResponse:
        stage = _stage(scenario_stage)
        generated_at = _demo_time(stage)
        destination_node = (
            "ISOLATED"
            if request.destination.place_id == NO_SAFE_DESTINATION_ID
            else "SHELTER"
        )
        graph = _route_graph(stage, destination_node=destination_node)
        result = _shortest_route(
            edges=graph,
            start="ORIGIN",
            destination=destination_node,
            strategy=request.strategy,
            allow_avoid_segments=request.allow_avoid_segments,
        )
        if result is None:
            blocked = [
                _route_segment(edge)
                for edge in graph
                if edge.recommendation in {"AVOID", "CLOSED"}
            ]
            return NoSafeRouteResponse(
                snapshot_id=_snapshot_id(stage, generated_at),
                generated_at=generated_at,
                reason_code="NO_ROUTABLE_PATH",
                message="No route meeting current closure and avoidance policy is available.",
                blocked_by=blocked,
                provenance=_simulated_sources([SENSOR_PRIMARY_ID, DRAIN_ID]),
                safety_note="PRAVAHA is decision support and does not guarantee route safety.",
            )

        selected = _route_alternative(result, request.strategy, stage)
        return SafeRouteSuccess(
            snapshot_id=_snapshot_id(stage, generated_at),
            generated_at=generated_at,
            selected_route=selected,
            alternatives=[selected],
            provenance=_simulated_sources([SENSOR_PRIMARY_ID, DRAIN_ID]),
            safety_note="PRAVAHA is decision support and does not guarantee route safety.",
        )

    def get_source_health(
        self,
        scenario_stage: str = "WARNING",
    ) -> list[SourceHealth]:
        stage = _stage(scenario_stage)
        generated_at = _demo_time(stage)
        return [
            SourceHealth(
                source_id=SENSOR_PRIMARY_ID,
                name="Demo rainfall and soil node",
                category="IOT_SENSOR",
                status="SIMULATED",
                last_success_at=generated_at,
                last_observation_at=generated_at - timedelta(minutes=2),
                age_seconds=120,
                expected_interval_seconds=900,
                freshness="GOOD",
                provenance="SIMULATED",
                message="Deterministic DEMO-001 source",
            ),
            SourceHealth(
                source_id=SENSOR_SECONDARY_ID,
                name="Secondary hillside sensor",
                category="IOT_SENSOR",
                status="UNAVAILABLE" if stage == "SEVERE" else "SIMULATED",
                last_success_at=generated_at
                - timedelta(minutes=74 if stage == "SEVERE" else 11),
                last_observation_at=generated_at
                - timedelta(minutes=74 if stage == "SEVERE" else 11),
                age_seconds=4440 if stage == "SEVERE" else 660,
                expected_interval_seconds=900,
                freshness="UNUSABLE" if stage == "SEVERE" else "DEGRADED",
                provenance="MISSING" if stage == "SEVERE" else "SIMULATED",
                message="Demo source intentionally stale in SEVERE stage"
                if stage == "SEVERE"
                else "Deterministic DEMO-001 source",
            ),
            SourceHealth(
                source_id="STATIC-GIS-CHANDRABANI",
                name="Chandrabani static GIS context",
                category="STATIC_GIS",
                status="STATIC",
                last_success_at=None,
                last_observation_at=None,
                age_seconds=None,
                expected_interval_seconds=None,
                freshness="GOOD",
                provenance="DERIVED",
                message=(
                    "SRTM elevation plus OSM roads, streams, settlements and POIs; "
                    "municipal drain capacity and shelter designation remain demo/estimated."
                ),
            ),
            SourceHealth(
                source_id="MODEL-DEVELOPMENT-FALLBACK",
                name="Synthetic development model",
                category="DEMO_FIXTURE",
                status="SIMULATED",
                last_success_at=generated_at,
                last_observation_at=generated_at,
                age_seconds=0,
                expected_interval_seconds=None,
                freshness="GOOD",
                provenance="SIMULATED",
                message="Development fallback, not operationally validated.",
            ),
        ]


def _demo_layers(stage: ScenarioStage) -> MapLayers:
    profile = PROFILES[stage]
    return MapLayers(
        catchments=_collection(
            [
                _feature(
                    CATCHMENT_ID,
                    "catchment",
                    "Polygon",
                    gis.bbox_polygon_coordinates(),
                    {
                        "name": "Chandrabani focused micro-catchment",
                        "risk_level": profile.risk_level,
                        "risk_score": profile.risk_score,
                        "confidence": profile.confidence,
                        "source_status": "ESTIMATED",
                        "mean_elevation_m": gis.TERRAIN["mean_elevation_m"],
                        "mean_slope_deg": gis.TERRAIN["mean_slope_deg"],
                    },
                )
            ]
        ),
        wards=_collection(
            [
                gis.point_feature(
                    settlement_id,
                    "ward",
                    settlement["coordinates"],
                    {
                        "name": settlement["name"],
                        "admin_level": settlement["admin_level"],
                        "risk_level": profile.risk_level,
                        "confidence": profile.confidence,
                        "source": settlement["source"],
                        "source_status": settlement["source_status"],
                        "source_osm_id": settlement["source_osm_id"],
                    },
                )
                for settlement_id, settlement in gis.SETTLEMENTS.items()
            ]
        ),
        rainfall=_collection(
            [
                _feature(
                    "RAIN-DEMO-001-CELL",
                    "rainfall",
                    "Polygon",
                    gis.bbox_polygon_coordinates(),
                    {
                        "intensity_mm_per_hr": profile.rainfall_intensity_mm_per_hr,
                        "risk_level": profile.risk_level,
                        "source_status": "SIMULATED",
                    },
                )
            ]
        ),
        sensors=_collection(
            [
                _feature(
                    SENSOR_PRIMARY_ID,
                    "sensor",
                    "Point",
                    [77.978689, 30.285029],
                    {
                        "name": "Demo rainfall and soil node",
                        "freshness": "GOOD",
                        "status": "SIMULATED",
                        "rainfall_mm_per_hr": profile.rainfall_intensity_mm_per_hr,
                    },
                ),
                _feature(
                    SENSOR_SECONDARY_ID,
                    "sensor",
                    "Point",
                    [77.9926, 30.2854],
                    {
                        "name": "Secondary hillside sensor",
                        "freshness": "UNUSABLE"
                        if stage == "SEVERE"
                        else "DEGRADED",
                        "status": "MISSING" if stage == "SEVERE" else "SIMULATED",
                        "rainfall_mm_per_hr": None
                        if stage == "SEVERE"
                        else profile.rainfall_intensity_mm_per_hr * 0.8,
                    },
                ),
            ]
        ),
        rivers=_collection(
            [
                gis.line_feature(
                    stream_id,
                    "river",
                    stream["coordinates"],
                    {
                        "name": stream["name"],
                        "stream_order": 2,
                        "source": stream["source"],
                        "source_status": stream["source_status"],
                        "source_osm_id": stream["source_osm_id"],
                    },
                )
                for stream_id, stream in gis.STREAMS.items()
            ]
        ),
        drains=_collection(
            [
                gis.line_feature(
                    DRAIN_ID,
                    "drain",
                    gis.STREAM_COORDS,
                    {
                        "name": "D-22 estimated municipal collector",
                        "risk_level": "HIGH"
                        if profile.drain_utilization >= 1.0
                        else "WATCH",
                        "utilization": profile.drain_utilization,
                        "source_status": "ESTIMATED",
                        "capacity_verification_status": "ESTIMATED",
                        "geometry_basis": "OSM stream geometry; not verified municipal drain",
                    },
                )
            ]
        ),
        roads=_collection(
            [
                gis.line_feature(
                    road_id,
                    "road",
                    road["coordinates"],
                    {
                        "name": road["name"],
                        "road_class": road["road_class"],
                        "recommendation": _road_recommendation(road_id, stage, profile),
                        "risk_level": _road_risk_level(road_id, stage, profile),
                        "source": road["source"],
                        "source_status": road["source_status"],
                        "source_osm_id": road["source_osm_id"],
                    },
                )
                for road_id, road in gis.ROADS.items()
            ]
        ),
        landslide=_collection(
            [
                _feature(
                    LANDSLIDE_ZONE_ID,
                    "landslide",
                    "Polygon",
                    [
                        [
                            [77.971, 30.290],
                            [77.980, 30.290],
                            [77.980, 30.298],
                            [77.971, 30.298],
                            [77.971, 30.290],
                        ]
                    ],
                    {
                        "name": "DEMO steep-slope susceptibility zone",
                        "susceptibility": "HIGH" if stage == "SEVERE" else "WATCH",
                        "risk_level": "HIGH" if stage == "SEVERE" else "WATCH",
                        "source_status": "DEMO",
                        "historical_inventory_status": "NOT_AVAILABLE",
                    },
                )
            ]
        ),
        closures=_collection(
            [
                _feature(
                    ROAD_CLOSED_ID,
                    "road",
                    "LineString",
                    [list(point) for point in gis.ROAD_CLOSED_COORDS],
                    {
                        "name": "DEMO authority closure on mapped unnamed road",
                        "recommendation": "CLOSED",
                        "authority_closed": True,
                        "risk_level": "SEVERE",
                        "source_status": "OPEN_REAL_DATA",
                        "status_basis": "DEMO_AUTHORITY_CLOSURE",
                    },
                )
            ]
            if stage == "SEVERE"
            else []
        ),
        shelters=_collection(
            [
                _feature(
                    SHELTER_ID,
                    "shelter",
                    "Point",
                    list(gis.SHELTER["coordinates"]),
                    {
                        "name": gis.SHELTER["name"],
                        "status": "AVAILABLE",
                        "source_status": gis.SHELTER["source_status"],
                        "poi_source_status": gis.SHELTER["poi_source_status"],
                        "designation_status": gis.SHELTER["designation_status"],
                    },
                )
            ]
        ),
        routes=_collection(
            [
                _feature(
                    ROUTE_ID,
                    "route",
                    "LineString",
                    _route_coordinates(stage),
                    {
                        "strategy": "safest",
                        "recommendation": "BYPASS"
                        if profile.road_recommendation == "AVOID"
                        else "DIRECT",
                    },
                )
            ]
        ),
    )


def _road_recommendation(
    road_id: str,
    stage: ScenarioStage,
    profile: StageProfile,
) -> str:
    if road_id == ROAD_CLOSED_ID and stage == "SEVERE":
        return "CLOSED"
    if road_id == ROAD_BYPASS_ID:
        return "PASSABLE" if stage == "NORMAL" else "CAUTION"
    return profile.road_recommendation


def _road_risk_level(
    road_id: str,
    stage: ScenarioStage,
    profile: StageProfile,
) -> str:
    if road_id == ROAD_CLOSED_ID and stage == "SEVERE":
        return "SEVERE"
    if road_id == ROAD_BYPASS_ID:
        return "LOW" if stage == "NORMAL" else "WATCH"
    return profile.risk_level


def _road_length_km(road_id: str) -> float | None:
    road = gis.ROADS.get(road_id)
    if road is None:
        return None
    return round(_line_length_km(road["coordinates"]), 2)


def _nearest_stream_distance_m(road_id: str) -> int | None:
    road = gis.ROADS.get(road_id)
    if road is None:
        return None
    nearest = gis.nearest_named(
        longitude=road["coordinates"][0][0],
        latitude=road["coordinates"][0][1],
        entities=gis.STREAMS,
    )
    if nearest is None:
        return None
    return int(round(nearest[1]))


def _line_length_km(
    coordinates: tuple[tuple[float, float], ...],
) -> float:
    distance_m = 0.0
    for start, end in zip(coordinates, coordinates[1:]):
        distance_m += gis.haversine_m(
            start[1],
            start[0],
            end[1],
            end[0],
        )
    return distance_m / 1000.0


def _shortest_route(
    *,
    edges: list[DemoEdge],
    start: str,
    destination: str,
    strategy: str,
    allow_avoid_segments: bool,
) -> tuple[DemoEdge, ...] | None:
    adjacency: dict[str, list[DemoEdge]] = {}
    for edge in edges:
        adjacency.setdefault(edge.start, []).append(edge)
        adjacency.setdefault(edge.end, [])
    if start not in adjacency or destination not in adjacency:
        return None

    distances = {node: float("inf") for node in adjacency}
    previous: dict[str, tuple[str, DemoEdge]] = {}
    distances[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distances[node]:
            continue
        if node == destination:
            break
        for edge in adjacency[node]:
            edge_cost = _edge_cost(edge, strategy, allow_avoid_segments)
            if edge_cost == float("inf"):
                continue
            candidate = cost + edge_cost
            if candidate < distances[edge.end]:
                distances[edge.end] = candidate
                previous[edge.end] = (node, edge)
                heapq.heappush(queue, (candidate, edge.end))
    if distances[destination] == float("inf"):
        return None
    route: list[DemoEdge] = []
    cursor = destination
    while cursor != start:
        if cursor not in previous:
            return None
        prior, edge = previous[cursor]
        route.append(edge)
        cursor = prior
    return tuple(reversed(route))


def _edge_cost(
    edge: DemoEdge,
    strategy: str,
    allow_avoid_segments: bool,
) -> float:
    if edge.recommendation == "CLOSED":
        return float("inf")
    if edge.recommendation == "AVOID" and not allow_avoid_segments:
        return float("inf")
    risk_weight = {
        "safest": 36.0,
        "balanced": 18.0,
        "fastest_available": 4.0,
    }[strategy]
    confidence_penalty = (1.0 - edge.confidence) * (
        10.0 if strategy != "fastest_available" else 3.0
    )
    avoid_penalty = 40.0 if edge.recommendation == "AVOID" else 0.0
    return edge.time_minutes + edge.risk_score * risk_weight + confidence_penalty + avoid_penalty


def _route_graph(
    stage: ScenarioStage,
    *,
    destination_node: str,
) -> list[DemoEdge]:
    profile = PROFILES[stage]
    bypass_recommendation = "PASSABLE" if stage == "NORMAL" else "CAUTION"
    if destination_node == "ISOLATED":
        return [
            _edge(
                "ORIGIN",
                "ISOLATED",
                ROAD_DIRECT_ID,
                11,
                4.8,
                min(profile.risk_score + 0.12, 1.0),
                profile.risk_level,
                "AVOID",
                _coordinates_list(gis.ROAD_DIRECT_COORDS),
            ),
            _edge(
                "ORIGIN",
                "RIDGE",
                ROAD_CLOSED_ID,
                9,
                3.7,
                0.95,
                "SEVERE",
                "CLOSED",
                _coordinates_list(gis.ROAD_CLOSED_COORDS),
            ),
        ]
    return [
        _edge(
            "ORIGIN",
            "SHELTER",
            ROAD_DIRECT_ID,
            11,
            4.8,
            min(profile.risk_score + 0.12, 1.0),
            profile.risk_level,
            profile.road_recommendation,
            _coordinates_list(gis.ROAD_DIRECT_COORDS),
        ),
        _edge(
            "ORIGIN",
            "BYPASS",
            ROAD_BYPASS_ID,
            8,
            3.0,
            0.34,
            "WATCH",
            bypass_recommendation,
            _coordinates_list(gis.ROAD_BYPASS_COORDS[:3]),
        ),
        _edge(
            "BYPASS",
            "SHELTER",
            ROAD_BYPASS_ID,
            10,
            3.4,
            0.38,
            "WATCH",
            bypass_recommendation,
            _coordinates_list(gis.ROAD_BYPASS_COORDS[2:]),
        ),
        _edge(
            "ORIGIN",
            "BRIDGE",
            ROAD_CLOSED_ID,
            6,
            2.0,
            0.95 if stage == "SEVERE" else 0.62,
            "SEVERE" if stage == "SEVERE" else "WARNING",
            "CLOSED" if stage == "SEVERE" else "CAUTION",
            _coordinates_list(gis.ROAD_CLOSED_COORDS),
        ),
        _edge(
            "BRIDGE",
            "SHELTER",
            ROAD_HILLSIDE_ID,
            7,
            2.0,
            0.82 if stage == "SEVERE" else 0.58,
            "HIGH" if stage == "SEVERE" else "WARNING",
            "AVOID" if stage == "SEVERE" else "CAUTION",
            _coordinates_list(gis.ROAD_DIRECT_COORDS[-3:]),
        ),
    ]


def _edge(
    start: str,
    end: str,
    road_id: str,
    time_minutes: float,
    distance_km: float,
    risk_score: float,
    risk_level: str,
    recommendation: str,
    coordinates: list[list[float]],
) -> DemoEdge:
    return DemoEdge(
        start,
        end,
        road_id,
        time_minutes,
        distance_km,
        risk_score,
        risk_level,
        recommendation,
        0.72,
        ("demo_graph_routing_evidence",),
        tuple((float(lon), float(lat)) for lon, lat in coordinates),
    )


def _coordinates_list(
    coordinates: tuple[tuple[float, float], ...],
) -> list[list[float]]:
    return [[lon, lat] for lon, lat in coordinates]


def _route_alternative(
    route: tuple[DemoEdge, ...],
    strategy: str,
    stage: ScenarioStage,
) -> RouteAlternative:
    coordinates: list[list[float]] = []
    for edge in route:
        for coordinate in edge.coordinates:
            item = [coordinate[0], coordinate[1]]
            if not coordinates or coordinates[-1] != item:
                coordinates.append(item)
    max_risk = max(edge.risk_score for edge in route)
    min_confidence = min(edge.confidence for edge in route)
    travel_time = sum(edge.time_minutes for edge in route)
    distance_km = sum(edge.distance_km for edge in route)
    graph = _route_graph(stage, destination_node="SHELTER")
    avoided_avoid = sum(edge.recommendation == "AVOID" for edge in graph)
    avoided_closed = sum(edge.recommendation == "CLOSED" for edge in graph)
    route_id = ROUTE_ID if route[0].road_id != ROAD_BYPASS_ID else "ROUTE-DEMO-001-BYPASS"
    return RouteAlternative(
        route_id=route_id,
        label="Bypass via higher ground"
        if route[0].road_id == ROAD_BYPASS_ID
        else "Direct monitored corridor",
        strategy=strategy,
        travel_time_minutes=travel_time,
        distance_km=round(distance_km, 2),
        maximum_risk_score=max_risk,
        minimum_confidence=min_confidence,
        additional_time_vs_fastest_minutes=max(0.0, travel_time - 11.0),
        unsafe_segments_avoided=avoided_avoid,
        closures_avoided=avoided_closed,
        explanation=[
            "Graph route excludes authority CLOSED roads",
            "Graph route excludes model AVOID roads unless explicitly allowed",
            "PRAVAHA does not claim guaranteed safety",
        ],
        geometry={"type": "LineString", "coordinates": coordinates},
        segments=[_route_segment(edge) for edge in route],
    )


def _route_segment(edge: DemoEdge) -> RouteSegment:
    return RouteSegment(
        road_id=edge.road_id,
        recommendation=edge.recommendation,
        risk_score=edge.risk_score,
        risk_level=edge.risk_level,
        confidence=edge.confidence,
        reasons=list(edge.reasons),
    )


def _timeline(stage: ScenarioStage) -> list[dict[str, Any]]:
    profile = PROFILES[stage]
    if stage == "NORMAL":
        points = [
            (0, 0.18, "LOW", "STABLE", "42%", "PASSABLE", "NORMAL", "Stable low-risk demo state"),
            (15, 0.18, "LOW", "STABLE", "43%", "PASSABLE", "NORMAL", "No threshold crossing"),
            (30, 0.19, "LOW", "STABLE", "44%", "PASSABLE", "NORMAL", "Drain reserve remains available"),
            (60, 0.20, "LOW", "STABLE", "46%", "PASSABLE", "NORMAL", "Monitor only"),
        ]
    elif stage == "WATCH":
        points = [
            (0, 0.38, "WATCH", "RISING", "76%", "CAUTION", "ELEVATED", "Rainfall trend increasing"),
            (15, 0.43, "WATCH", "RISING", "81%", "CAUTION", "ELEVATED", "Drain reserve narrowing"),
            (30, 0.48, "WATCH", "RISING", "88%", "CAUTION", "ELEVATED", "Approaching warning threshold"),
            (60, 0.56, "WARNING", "RISING", "96%", "CAUTION", "WARNING", "Possible WARNING threshold"),
        ]
    elif stage == "WARNING":
        points = [
            (0, 0.64, "WARNING", "RISING", "118%", "AVOID", "WARNING", "D-22 over capacity"),
            (15, 0.70, "HIGH", "RISING", "124%", "AVOID", "WARNING", "Runoff continues rising"),
            (30, 0.76, "HIGH", "RISING", "131%", "AVOID", "WARNING", "Transport Nagar Road remains AVOID"),
            (60, 0.84, "HIGH", "RISING", "143%", "AVOID", "EMERGENCY", "Possible corridor degradation"),
        ]
    else:
        points = [
            (0, 0.88, "SEVERE", "RISING", "156%", "AVOID", "EMERGENCY", "Severe hazard state active"),
            (15, 0.90, "SEVERE", "RISING", "163%", "AVOID", "EMERGENCY", "Drain overflow expanding"),
            (30, 0.92, "SEVERE", "RISING", "172%", "AVOID", "EMERGENCY", "Fastest corridor unavailable"),
            (60, 0.95, "SEVERE", "RISING", "184%", "AVOID", "EMERGENCY", "Shelter routing requires review"),
        ]
    return [
        {
            "label": "NOW" if minutes == 0 else f"+{minutes} min",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": max(profile.confidence - minutes / 600.0, 0.0),
            "trend": trend,
            "drainage_status": drainage,
            "road_status": road,
            "city_status": city,
            "note": note,
        }
        for minutes, risk_score, risk_level, trend, drainage, road, city, note in points
    ]


def _rainfall_windows(profile: StageProfile) -> list[dict[str, Any]]:
    values = [
        ("15 min", round(profile.rain_1h_mm / 4), 0.92, 8, 2, 7),
        ("30 min", round(profile.rain_1h_mm / 2), 0.88, 14, 2, 9),
        ("1 h", profile.rain_1h_mm, 0.84, 24, 2, 11),
        ("3 h", profile.rain_3h_mm, 0.76, 52, 2, 18),
        ("6 h", profile.rain_3h_mm + 26, 0.68, 77, 2, 31),
        ("24 h", profile.rain_24h_mm, 0.61, 164, 2, 46),
    ]
    return [
        {
            "label": label,
            "value_mm": value,
            "status": "DERIVED",
            "coverage_fraction": coverage,
            "observation_count": count,
            "latest_observation_age_minutes": age,
            "largest_gap_minutes": gap,
            "quality": "GOOD"
            if coverage >= 0.8
            else "DEGRADED"
            if coverage >= 0.65
            else "UNUSABLE",
        }
        for label, value, coverage, count, age, gap in values
    ]


def _provenance_rows(profile: StageProfile) -> list[dict[str, Any]]:
    return [
        {
            "variable": "rainfall",
            "source": SENSOR_PRIMARY_ID,
            "status": "SIMULATED",
            "age_minutes": 2,
            "confidence": profile.confidence,
        },
        {
            "variable": "soil saturation",
            "source": SENSOR_PRIMARY_ID,
            "status": "SIMULATED",
            "age_minutes": 2,
            "confidence": profile.confidence,
        },
        {
            "variable": "runoff",
            "source": "ML hydrology development output",
            "status": "DERIVED",
            "age_minutes": 1,
            "confidence": max(profile.confidence - 0.04, 0.0),
        },
        {
            "variable": "drain capacity",
            "source": "estimated D-22 profile aligned to OSM stream",
            "status": "ESTIMATED",
            "age_minutes": None,
            "confidence": 0.66,
        },
        {
            "variable": "DEM terrain",
            "source": gis.TERRAIN["source_name"],
            "status": gis.TERRAIN["derived_status"],
            "age_minutes": None,
            "confidence": 0.64,
        },
    ]


def _cascade(profile: StageProfile) -> list[dict[str, str]]:
    return [
        {
            "label": "Heavy rainfall",
            "state": "OBSERVED",
            "detail": f"{profile.rainfall_intensity_mm_per_hr} mm/hr SIMULATED fixture",
        },
        {
            "label": "Soil saturation",
            "state": "OBSERVED",
            "detail": f"{round(profile.soil_saturation * 100)}% saturation",
        },
        {
            "label": "Runoff increase",
            "state": "PREDICTED",
            "detail": f"{profile.runoff_mm} mm runoff estimate",
        },
        {
            "label": "Drain overload",
            "state": "PREDICTED" if profile.overflow_m3_s > 0 else "POSSIBLE",
            "detail": f"D-22 overflow {profile.overflow_m3_s} m3/s"
            if profile.overflow_m3_s > 0
            else "No overflow in this scenario",
        },
        {
            "label": "Road flooding",
            "state": "PREDICTED"
            if profile.road_recommendation == "AVOID"
            else "POSSIBLE",
            "detail": f"{ROAD_DIRECT_ID} is {profile.road_recommendation}",
        },
    ]


def _stage_history(stage: ScenarioStage) -> list[dict[str, Any]]:
    index = STAGE_ORDER.index(stage)
    return [
        {
            "label": item,
            "rainfall_mm_per_hr": PROFILES[item].rainfall_intensity_mm_per_hr,
            "soil_moisture_percentage": round(PROFILES[item].soil_saturation * 100),
        }
        for item in STAGE_ORDER[: index + 1]
    ]


def _model_metadata(
    stage: ScenarioStage,
    generated_at: datetime,
    profile: StageProfile,
) -> ModelMetadata:
    return ModelMetadata(
        prediction_id=f"PRED-{SCENARIO_ID}-{stage}",
        model_version="synthetic-development-v1",
        generated_at=generated_at,
        input_state_time=generated_at,
        risk_score=profile.risk_score,
        risk_level=profile.risk_level,
        confidence=profile.confidence,
        data_quality_score=0.90 if stage != "SEVERE" else 0.74,
        top_factors=list(profile.reasons[:3]),
        runtime_status="DEVELOPMENT_FALLBACK",
        operationally_validated=False,
    )


def _feature(
    feature_id: str,
    entity_type: str,
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
            "entityType": entity_type,
            "data_label": "SIMULATED",
            **properties,
        },
    }


def _collection(features: list[dict[str, Any]]) -> FeatureCollection:
    return FeatureCollection(features=features)


def _simulated_sources(
    sources: list[str],
    **extras: Any,
) -> SourceMetadata:
    return SourceMetadata(
        data_label="SIMULATED",
        sources=sources,
        provider="backend-demo-adapter",
        **extras,
    )


def _state_time(fused_state: dict[str, Any]) -> datetime:
    value = fused_state.get("state_time")
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise MLIntelligenceServiceError("Fused state is missing a valid state_time.")


def _demo_time(stage: ScenarioStage) -> datetime:
    return BASE_TIME + timedelta(minutes=PROFILES[stage].offset_minutes)


def _snapshot_id(stage: ScenarioStage, generated_at: datetime) -> str:
    utc = generated_at.astimezone(timezone.utc)
    return (
        f"snap_{SCENARIO_ID.lower()}_{stage.lower()}_"
        + utc.strftime("%Y%m%dT%H%M%SZ")
    )


def _stage(value: str) -> ScenarioStage:
    normalized = value.upper()
    if normalized not in PROFILES:
        raise MLIntelligenceServiceError(f"Unknown demo scenario stage: {value}")
    return normalized  # type: ignore[return-value]


def _metric(
    label: str,
    value: Any,
    unit: str | None,
    status: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "unit": unit,
        "status": status,
    }


def _entity_name(
    entities: dict[str, dict[str, Any]],
    nearest: tuple[str, float] | None,
) -> str | None:
    if nearest is None:
        return None
    entity_id, _ = nearest
    entity = entities.get(entity_id)
    if entity is None:
        return entity_id
    return str(entity.get("name") or entity_id)


def _distance_value(
    nearest: tuple[str, float] | None,
) -> int | None:
    if nearest is None:
        return None
    return int(round(nearest[1]))


def _road_name(road_id: str) -> str:
    names = {
        ROAD_DIRECT_ID: "Transport Nagar Road",
        ROAD_BYPASS_ID: "Post Office Road",
        ROAD_CLOSED_ID: "DEMO authority closure on mapped unnamed road",
        ROAD_HILLSIDE_ID: "Transport Nagar Road connector",
    }
    return names.get(road_id, road_id)


def _route_coordinates(stage: ScenarioStage) -> list[list[float]]:
    if stage in {"WARNING", "SEVERE"}:
        return _coordinates_list(gis.ROAD_BYPASS_COORDS)
    return _coordinates_list(gis.ROAD_DIRECT_COORDS)
