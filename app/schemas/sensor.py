from datetime import datetime

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
)


class Location(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )

    village: str
    ward: str
    lat: float = Field(
        ge=-90.0,
        le=90.0,
        validation_alias=AliasChoices(
            "lat",
            "latitude",
        ),
        serialization_alias="lat",
    )
    lon: float = Field(
        ge=-180.0,
        le=180.0,
        validation_alias=AliasChoices(
            "lon",
            "longitude",
        ),
        serialization_alias="lon",
    )


class SensorMetrics(BaseModel):
    rainfall_mm_per_hr: float = Field(ge=0.0)
    soil_moisture_percentage: float = Field(ge=0.0, le=100.0)
    slope_tilt_degrees: float


class SensorIngestionPayload(BaseModel):
    device_id: str
    timestamp: datetime
    location: Location
    sensor_metrics: SensorMetrics

    @property
    def canonical_location(self) -> dict[str, float | str]:
        return {
            "village": self.location.village,
            "ward": self.location.ward,
            "latitude": self.location.lat,
            "longitude": self.location.lon,
        }
