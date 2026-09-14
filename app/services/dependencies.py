from app.services.data_state import (
    DataStateService,
    DemoDataStateService,
)
from app.services.intelligence import (
    DemoMLIntelligenceService,
    MLIntelligenceService,
)


_data_state_service = DemoDataStateService()
_ml_intelligence_service = DemoMLIntelligenceService()


def get_data_state_service() -> DataStateService:
    return _data_state_service


def get_ml_intelligence_service() -> MLIntelligenceService:
    return _ml_intelligence_service

