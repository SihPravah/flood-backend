from fastapi.testclient import TestClient

from app.main import app
from app.services.dependencies import (
    get_data_state_service,
    get_ml_intelligence_service,
)
from app.services.errors import (
    DataStateServiceError,
    MLIntelligenceServiceError,
)


client = TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_map_snapshot_exposes_demo_label_and_layers():
    response = client.get(
        "/api/v1/map/intelligence?scenario_stage=WARNING"
    )

    assert response.status_code == 200
    body = response.json()

    assert body["mode"] == "DEMO"
    assert body["data_label"] == "SIMULATED"
    assert body["scenario_id"] == "DEMO-001"
    assert body["city"]["operational_status"] == "WARNING"
    assert body["summary"]["overflowing_drains"] == 1
    assert body["layers"]["catchments"]["type"] == "FeatureCollection"
    assert body["source_health"][0]["provenance"] == "SIMULATED"
    assert body["model_metadata"]["operationally_validated"] is False
    assert body["events"]

    first_coordinate = body["layers"]["sensors"]["features"][0]["geometry"][
        "coordinates"
    ]
    assert first_coordinate == [78.039, 30.329]


def test_catchment_detail_exposes_confidence_provenance_and_fused_state():
    response = client.get(
        "/api/v1/map/catchments/UK-CHM-DEHRADUN-01?scenario_stage=WARNING"
    )

    assert response.status_code == 200
    body = response.json()

    assert body["risk_score"] == 0.64
    assert body["risk_level"] == "WARNING"
    assert body["confidence"] == 0.74
    assert body["provenance"]["data_label"] == "SIMULATED"
    assert body["fused_state"] == "FusedCatchmentState v2.1"
    assert "rainfall" in body


def test_detail_endpoints_exist():
    endpoints = [
        "/api/v1/map/drains/D-22",
        "/api/v1/map/roads/ROAD-SHELTER-CORRIDOR",
        "/api/v1/map/sensors/SENSOR-SIM-RAIN-SOIL-01",
        "/api/v1/map/alerts",
    ]

    for endpoint in endpoints:
        response = client.get(endpoint)
        assert response.status_code == 200


def test_avoid_and_closed_are_distinct():
    avoid = client.get(
        "/api/v1/map/roads/ROAD-SHELTER-CORRIDOR?scenario_stage=SEVERE"
    ).json()
    closed = client.get(
        "/api/v1/map/roads/ROAD-BRIDGE-APPROACH?scenario_stage=SEVERE"
    ).json()

    assert avoid["recommendation"] == "AVOID"
    assert avoid["authority_closed"] is False
    assert closed["recommendation"] == "CLOSED"
    assert closed["authority_closed"] is True


def test_events_endpoint_reports_state_changes():
    response = client.get("/api/v1/events?scenario_stage=WARNING")

    assert response.status_code == 200
    body = response.json()

    assert any(event["entity_id"] == "ROAD-SHELTER-CORRIDOR" for event in body)
    assert all(event["provenance"]["data_label"] == "SIMULATED" for event in body)
    assert all("title" in event for event in body)
    assert all("reasons" in event for event in body)


def test_data_service_failure_returns_503():
    class FailingDataService:
        def get_fused_catchment_state(self, catchment_id, scenario_stage="WARNING"):
            raise DataStateServiceError("data unavailable")

    app.dependency_overrides[get_data_state_service] = (
        lambda: FailingDataService()
    )

    response = client.get(
        "/api/v1/map/intelligence"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "data unavailable"


def test_ml_service_failure_returns_503():
    class FailingMLService:
        def build_map_intelligence(self, fused_state, scenario_stage="WARNING"):
            raise MLIntelligenceServiceError("ml unavailable")

    app.dependency_overrides[get_ml_intelligence_service] = (
        lambda: FailingMLService()
    )

    response = client.get(
        "/api/v1/map/intelligence"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "ml unavailable"
