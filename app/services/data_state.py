from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.services.errors import DataStateServiceError


class DataStateService:
    def get_fused_catchment_state(
        self,
        catchment_id: str,
    ) -> dict[str, Any]:
        raise NotImplementedError


class DemoDataStateService(DataStateService):
    def get_fused_catchment_state(
        self,
        catchment_id: str,
    ) -> dict[str, Any]:
        if not settings.demo_mode:
            raise DataStateServiceError(
                "Demo fused-state fallback is disabled in operational mode."
            )

        generated_at = datetime(
            2026,
            9,
            9,
            8,
            45,
            tzinfo=timezone.utc,
        )

        return {
            "catchment_id": catchment_id,
            "state_time": generated_at.isoformat(),
            "rainfall": {
                "intensity": {
                    "value": 48.0,
                    "status": "SIMULATED",
                    "confidence": 0.90,
                    "age_minutes": 2.0,
                },
                "rain_15m": _window(12.0),
                "rain_30m": _window(24.0),
                "rain_1h": _window(48.0),
                "rain_3h": _window(92.0),
                "rain_6h": _window(118.0),
                "rain_24h": _window(145.0),
            },
            "soil": {
                "saturation": 0.82,
                "status": "SIMULATED",
                "confidence": 0.86,
                "age_minutes": 3.0,
            },
            "data_quality": {
                "overall_score": 0.90,
                "missing_sources": [],
                "temporal_freshness": "GOOD",
            },
        }


def _window(
    value_mm: float,
) -> dict[str, Any]:
    return {
        "value_mm": value_mm,
        "status": "DERIVED",
        "coverage_fraction": 1.0,
        "largest_gap_minutes": 5.0,
        "latest_observation_age_minutes": 2.0,
        "observation_count": 6,
        "quality": "GOOD",
    }

