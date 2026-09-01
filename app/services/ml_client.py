from app.schemas.prediction import PredictionResponse
from app.schemas.sensor import SensorIngestionPayload


class MLClient:
    def predict(
        self,
        payload: SensorIngestionPayload,
    ) -> PredictionResponse:
        """
        Temporary mock prediction.

        This will later be replaced by a real HTTP call
        to the flood-ml service while preserving the
        backend-facing interface.
        """

        rainfall = payload.sensor_metrics.rainfall_mm_per_hr
        soil = payload.sensor_metrics.soil_moisture_percentage / 100.0
        slope = payload.sensor_metrics.slope_tilt_degrees

        # Temporary development-only heuristic.
        score = (
            rainfall / 100.0
            + soil * 0.30
            + min(abs(slope) / 90.0, 1.0) * 0.10
        )

        score = max(0.0, min(score, 1.0))

        if score >= 0.85:
            level = "SEVERE"
        elif score >= 0.70:
            level = "HIGH"
        elif score >= 0.50:
            level = "WARNING"
        elif score >= 0.30:
            level = "WATCH"
        else:
            level = "LOW"

        return PredictionResponse(
            device_id=payload.device_id,
            timestamp=payload.timestamp,
            risk_score=round(score, 4),
            risk_level=level,
            confidence=0.50,
            prediction_mode="MOCK",
            model_version="mock-v1",
        )


ml_client = MLClient()