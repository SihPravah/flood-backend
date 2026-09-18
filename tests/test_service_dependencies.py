from app.core.config import Settings
from app.services.data_state import DemoDataStateService
from app.services.dependencies import (
    build_data_state_service,
    build_ml_intelligence_service,
)
from app.services.http_adapters import (
    HTTPDataStateService,
    HTTPMLIntelligenceService,
    UnavailableDataStateService,
    UnavailableMLIntelligenceService,
)
from app.services.intelligence import DemoMLIntelligenceService


def test_dependency_factories_choose_demo_services():
    config = Settings(demo_mode=True)

    assert isinstance(build_data_state_service(config), DemoDataStateService)
    assert isinstance(build_ml_intelligence_service(config), DemoMLIntelligenceService)


def test_dependency_factories_choose_http_services_for_operational_mode():
    config = Settings(
        demo_mode=False,
        data_service_url="http://data-service",
        ml_service_url="http://ml-service",
    )

    assert isinstance(build_data_state_service(config), HTTPDataStateService)
    assert isinstance(build_ml_intelligence_service(config), HTTPMLIntelligenceService)


def test_dependency_factories_fail_closed_without_operational_urls():
    config = Settings(
        demo_mode=False,
        data_service_url="",
        ml_service_url="",
    )

    assert isinstance(build_data_state_service(config), UnavailableDataStateService)
    assert isinstance(build_ml_intelligence_service(config), UnavailableMLIntelligenceService)
