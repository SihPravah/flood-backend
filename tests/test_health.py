from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["service"] == "pravaha-backend"


def test_system_health_exposes_source_health():
    response = client.get("/api/v1/system/health?scenario_stage=SEVERE")

    assert response.status_code == 200
    body = response.json()

    assert body["scenario_id"] == "DEMO-001"
    assert body["status"] == "degraded"
    assert any(
        source["source_id"] == "SENSOR-SIM-RAIN-SOIL-02"
        and source["status"] == "UNAVAILABLE"
        for source in body["source_health"]
    )
    assert "state_store" in body
    assert "snapshot_count" in body["state_store"]
    assert "event_count" in body["state_store"]
    assert "route_evaluation_count" in body["state_store"]
