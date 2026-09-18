import pytest

from app.main import app
from app.services.dependencies import get_monitoring_state_store


@pytest.fixture(autouse=True)
def isolated_backend_state():
    app.dependency_overrides.clear()
    get_monitoring_state_store().clear()
    yield
    app.dependency_overrides.clear()
    get_monitoring_state_store().clear()
