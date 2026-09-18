from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RiskLevel = Literal[
    "LOW",
    "WATCH",
    "WARNING",
    "HIGH",
    "SEVERE",
]
DataLabel = Literal[
    "OBSERVED",
    "DERIVED",
    "ESTIMATED",
    "SIMULATED",
    "MISSING",
]
RoadRecommendation = Literal[
    "PASSABLE",
    "CAUTION",
    "AVOID",
    "CLOSED",
]


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    data_label: DataLabel
    sources: list[str] = Field(
        default_factory=list
    )
    static_verification_status: str | None = None
    capacity_verification_status: str | None = None
    road_verification_status: str | None = None
    provider: str | None = None


class SourceHealth(BaseModel):
    source_id: str
    name: str
    category: str
    status: Literal[
        "HEALTHY",
        "DEGRADED",
        "UNAVAILABLE",
        "STATIC",
        "SIMULATED",
    ]
    last_success_at: datetime | None = None
    last_observation_at: datetime | None = None
    age_seconds: int | None = Field(default=None, ge=0)
    expected_interval_seconds: int | None = Field(default=None, ge=0)
    freshness: Literal["GOOD", "DEGRADED", "UNUSABLE"]
    provenance: DataLabel
    message: str


class StructuredEvent(BaseModel):
    event_id: str
    snapshot_id: str
    generated_at: datetime
    event_type: str
    entity_type: str
    entity_id: str
    previous_value: str | None = None
    current_value: str | None = None
    severity: RiskLevel
    title: str
    message: str
    reasons: list[str] = Field(default_factory=list)
    provenance: SourceMetadata


class ModelMetadata(BaseModel):
    prediction_id: str
    model_version: str
    generated_at: datetime
    input_state_time: datetime
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    data_quality_score: float = Field(ge=0.0, le=1.0)
    top_factors: list[str] = Field(default_factory=list)
    runtime_status: str = "DEVELOPMENT_FALLBACK"
    operationally_validated: bool = False


class DataMetric(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    value: Any | None
    unit: str | None = None
    status: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    age_minutes: float | None = Field(default=None, ge=0.0)
    source: str | None = None
    quality: str | None = None


class RiskBearing(BaseModel):
    model_config = ConfigDict(extra="allow")

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    risk_level: RiskLevel
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reasons: list[str] = Field(
        default_factory=list
    )
    provenance: SourceMetadata
    last_updated: datetime


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[dict[str, Any]] = Field(
        default_factory=list
    )


class MapLayers(BaseModel):
    catchments: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    wards: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    rainfall: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    drains: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    roads: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    rivers: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    landslide: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    closures: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    sensors: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    shelters: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    routes: FeatureCollection = Field(
        default_factory=FeatureCollection
    )


class CityStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    city_id: str
    name: str
    operational_status: Literal[
        "NORMAL",
        "ELEVATED",
        "WARNING",
        "EMERGENCY",
        "INSUFFICIENT_DATA",
    ]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reasons: list[str] = Field(
        default_factory=list
    )
    last_updated: datetime


class MapSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    catchment_count: int
    high_risk_catchments: int
    overflowing_drains: int
    roads_to_avoid: int
    confirmed_road_closures: int
    active_alerts: int
    highest_risk_catchment: str | None = None
    highest_risk_ward: str | None = None
    shelters_available: int | None = None
    exposed_population: int | None = None
    source_health: list[SourceHealth] = Field(default_factory=list)
    latest_threshold_crossing: str | None = None


class MapIntelligenceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    snapshot_id: str
    generated_at: datetime
    state_time: datetime
    scenario_id: str
    mode: Literal["DEMO", "OPERATIONAL"]
    data_label: DataLabel
    city: CityStatus
    layers: MapLayers
    summary: MapSummary
    source_health: list[SourceHealth] = Field(default_factory=list)
    events: list[StructuredEvent] = Field(default_factory=list)
    model_metadata: ModelMetadata


class CatchmentDetail(RiskBearing):
    catchment_id: str
    snapshot_id: str
    fused_state: str
    hydrology: dict[str, Any]
    rainfall: dict[str, Any]
    soil: dict[str, Any]
    anticipation: dict[str, Any]
    landslide: dict[str, Any]


class DrainDetail(RiskBearing):
    drain_id: str
    snapshot_id: str
    inflow_m3_per_s: float
    capacity_m3_per_s: float
    capacity_utilization: float
    overflow_m3_per_s: float
    condition: str
    affected_roads: list[str] = Field(
        default_factory=list
    )


class RoadDetail(RiskBearing):
    road_id: str
    snapshot_id: str
    recommendation: RoadRecommendation
    associated_drain_id: str | None = None
    authority_closed: bool = False
    terrain: dict[str, Any] = Field(
        default_factory=dict
    )
    historical_waterlogging_score: float | None = None
    landslide_exposure: dict[str, Any] = Field(
        default_factory=dict
    )


class SensorDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    device_id: str
    snapshot_id: str
    measurements: dict[str, Any]
    observed_at: datetime
    age_minutes: float
    freshness: str
    provenance: SourceMetadata
    last_updated: datetime


class LocationInspection(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["location"] = "location"
    id: str
    snapshot_id: str
    latitude: float
    longitude: float
    jurisdiction: str | None = None
    ward_or_village: str | None = None
    catchment_id: str | None = None
    nearest_road: str | None = None
    nearest_stream: str | None = None
    nearest_drain: str | None = None
    nearest_shelter: str | None = None
    terrain: list[DataMetric] = Field(default_factory=list)
    hydrology: list[DataMetric] = Field(default_factory=list)
    hazard_context: list[DataMetric] = Field(default_factory=list)
    data_quality: list[DataMetric] = Field(default_factory=list)


class Alert(BaseModel):
    model_config = ConfigDict(extra="allow")

    alert_id: str
    alert_type: str
    risk_level: RiskLevel
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    message: str
    affected_entity_ids: list[str]
    reasons: list[str]
    provenance: SourceMetadata
    issued_at: datetime
    last_updated: datetime
