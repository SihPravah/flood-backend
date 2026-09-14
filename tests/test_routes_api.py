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
        "place_id": "SHELTER-01",
    },
    "strategy": "safest",
}


def test_safe_route_success_shape():
    response = client.post(
        "/api/v1/routes/safe",
        json=ROUTE_REQUEST,
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ROUTE_FOUND"
    assert body["provenance"]["data_label"] == "SIMULATED"
    assert body["selected_route"]["maximum_risk_score"] < 1.0
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
        "/api/v1/routes/safe",
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
