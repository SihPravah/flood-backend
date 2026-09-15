from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


ROUTE_REQUEST = {
    "origin": {
        "lon": 78.0300,
        "lat": 30.3200,
        "label": "Demo origin",
    },
        "destination": {
            "lon": 78.0520,
            "lat": 30.3350,
            "label": "Demo shelter",
            "place_id": "SHELTER-SCHOOL-01",
        },
    "strategy": "safest",
}


def test_safe_route_success_shape():
    response = client.post(
        "/api/v1/routes/safe?scenario_stage=WARNING",
        json=ROUTE_REQUEST,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ROUTE_FOUND"
    assert body["provenance"]["data_label"] == "SIMULATED"
    assert body["selected_route"]["maximum_risk_score"] < 1.0
    assert body["selected_route"]["segments"][0]["road_id"] == (
        "ROAD-HIGHER-GROUND-BYPASS"
    )
    assert body["selected_route"]["minimum_confidence"] <= 1.0
    assert "does not guarantee route safety" in body["safety_note"]


def test_no_safe_route_is_explicit_not_exception():
    payload = {
        **ROUTE_REQUEST,
        "destination": {
            **ROUTE_REQUEST["destination"],
            "place_id": "DEMO-NO-SAFE-ROUTE",
        },
    }

    response = client.post(
        "/api/v1/routes/safe?scenario_stage=SEVERE",
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "NO_SAFE_ROUTE"
    assert body["reason_code"] == "NO_ROUTABLE_PATH"
    assert any(
        segment["recommendation"] == "AVOID"
        for segment in body["blocked_by"]
    )
    assert any(
        segment["recommendation"] == "CLOSED"
        for segment in body["blocked_by"]
    )


def test_route_request_accepts_contract_longitude_latitude():
    payload = {
        "origin": {
            "longitude": 78.0300,
            "latitude": 30.3200,
        },
        "destination": {
            "longitude": 78.0520,
            "latitude": 30.3350,
            "place_id": "SHELTER-SCHOOL-01",
        },
        "strategy": "balanced",
    }

    response = client.post(
        "/api/v1/routes/safe?scenario_stage=NORMAL",
        json=payload,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ROUTE_FOUND"
