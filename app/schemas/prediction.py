from datetime import datetime

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    device_id: str
    timestamp: datetime

    risk_score: float
    risk_level: str

    confidence: float

    prediction_mode: str
    model_version: str