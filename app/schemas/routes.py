from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.map import (
    RoadRecommendation,
    SourceMetadata,
)


class RoutePoint(BaseModel):
    lon: float = Field(
        ge=-180.0,
        le=180.0,
    )
    lat: float = Field(
        ge=-90.0,
        le=90.0,
    )
    label: str | None = None
    place_id: str | None = None


class SafeRouteRequest(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    strategy: Literal[
        "safest",
        "balanced",
        "fastest_available",
    ] = "safest"
    allow_avoid_segments: bool = False


class RouteSegment(BaseModel):
    road_id: str
    recommendation: RoadRecommendation
    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    risk_level: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reasons: list[str]


class RouteAlternative(BaseModel):
    route_id: str
    label: str
    strategy: str
    travel_time_minutes: float
    distance_km: float
    maximum_risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )
    minimum_confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    additional_time_vs_fastest_minutes: float
    unsafe_segments_avoided: int
    closures_avoided: int
    explanation: list[str]
    geometry: dict
    segments: list[RouteSegment]


class SafeRouteSuccess(BaseModel):
    status: Literal["ROUTE_FOUND"] = "ROUTE_FOUND"
    snapshot_id: str
    generated_at: datetime
    selected_route: RouteAlternative
    alternatives: list[RouteAlternative]
    provenance: SourceMetadata
    safety_note: str


class NoSafeRouteResponse(BaseModel):
    status: Literal["NO_SAFE_ROUTE"] = "NO_SAFE_ROUTE"
    snapshot_id: str
    generated_at: datetime
    reason_code: str
    message: str
    blocked_by: list[RouteSegment]
    provenance: SourceMetadata
    safety_note: str

