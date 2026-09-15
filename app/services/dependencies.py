from app.services.data_state import (
    DataStateService,
    DemoDataStateService,
)
from app.services.http_adapters import (
    HTTPDataStateService,
    HTTPMLIntelligenceService,
    UnavailableDataStateService,
    UnavailableMLIntelligenceService,
)
from app.services.intelligence import (
    DemoMLIntelligenceService,
    MLIntelligenceService,
)
from app.services.state_store import MonitoringStateStore
from app.core.config import Settings, settings


def build_data_state_service(config: Settings) -> DataStateService:
    if config.demo_mode:
        return DemoDataStateService()
    if config.data_service_url:
        return HTTPDataStateService(
            config.data_service_url,
            timeout_seconds=config.request_timeout_seconds,
            retry_count=config.request_retry_count,
        )
    return UnavailableDataStateService()


def build_ml_intelligence_service(config: Settings) -> MLIntelligenceService:
    if config.demo_mode:
        return DemoMLIntelligenceService()
    if config.ml_service_url:
        return HTTPMLIntelligenceService(
            config.ml_service_url,
            timeout_seconds=config.request_timeout_seconds,
            retry_count=config.request_retry_count,
        )
    return UnavailableMLIntelligenceService()


_data_state_service = build_data_state_service(settings)
_ml_intelligence_service = build_ml_intelligence_service(settings)
_monitoring_state_store = MonitoringStateStore()


def get_data_state_service() -> DataStateService:
    return _data_state_service


def get_ml_intelligence_service() -> MLIntelligenceService:
    return _ml_intelligence_service


def get_monitoring_state_store() -> MonitoringStateStore:
    return _monitoring_state_store
