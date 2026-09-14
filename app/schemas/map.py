from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    data_label: DataLabel
    sources: list[str] = Field(
        default_factory=list
    )
    static_verification_status: str | None = None
    capacity_verification_status: str | None = None
    provider: str | None = None


class RiskBearing(BaseModel):
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
    drains: FeatureCollection = Field(
        default_factory=FeatureCollection
    )
    roads: FeatureCollection = Field(
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
    city_id: str
    name: str
    operational_status: Literal[
        "NORMAL",
        "ELEVATED",
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
    catchment_count: int
    high_risk_catchments: int
    overflowing_drains: int
    roads_to_avoid: int
    confirmed_road_closures: int
    active_alerts: int


class MapIntelligenceResponse(BaseModel):
    snapshot_id: str
    generated_at: datetime
    mode: Literal["DEMO", "OPERATIONAL"]
    data_label: DataLabel
    city: CityStatus
    layers: MapLayers
    summary: MapSummary


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
    device_id: str
    snapshot_id: str
    measurements: dict[str, Any]
    observed_at: datetime
    age_minutes: float
    freshness: str
    provenance: SourceMetadata
    last_updated: datetime


class Alert(BaseModel):
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

