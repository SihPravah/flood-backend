from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


VALID_PAYLOAD = {
    "device_id": "SIM_NODE_04",
    "timestamp": "2026-08-30T14:30:00Z",
    "location": {
        "village": "Munnar",
        "ward": "Ward_3",
        "latitude": 10.0889,
        "longitude": 77.0595,
    },
    "sensor_metrics": {
        "rainfall_mm_per_hr": 45.5,
        "soil_moisture_percentage": 82.0,
        "slope_tilt_degrees": 12.2,
    },
}


def test_sensor_ingestion():
    response = client.post(
        "/api/v1/ingest/sensors",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 202

    body = response.json()

    # Basic ingestion response
    assert body["status"] == "accepted"
    assert body["device_id"] == "SIM_NODE_04"

    # Backend-added metadata
    assert "observed_at" in body
    assert "received_at" in body
    assert "age_seconds" in body

    assert isinstance(body["age_seconds"], int)
    assert body["age_seconds"] >= 0

    # Mock ML prediction should be attached
    assert "prediction" in body

    prediction = body["prediction"]

    assert prediction["device_id"] == "SIM_NODE_04"

    assert "timestamp" in prediction

    assert "risk_score" in prediction
    assert 0.0 <= prediction["risk_score"] <= 1.0

    assert "risk_level" in prediction
    assert prediction["risk_level"] in {
        "LOW",
        "WATCH",
        "WARNING",
        "HIGH",
        "SEVERE",
    }

    assert "confidence" in prediction
    assert 0.0 <= prediction["confidence"] <= 1.0

    assert prediction["prediction_mode"] == "MOCK"
    assert prediction["model_version"] == "mock-v1"


def test_invalid_soil_moisture_rejected():
    payload = {
        **VALID_PAYLOAD,
        "sensor_metrics": {
            **VALID_PAYLOAD["sensor_metrics"],
            "soil_moisture_percentage": 150.0,
        },
    }

    response = client.post(
        "/api/v1/ingest/sensors",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_rainfall_rejected():
    payload = {
        **VALID_PAYLOAD,
        "sensor_metrics": {
            **VALID_PAYLOAD["sensor_metrics"],
            "rainfall_mm_per_hr": -5.0,
        },
    }

    response = client.post(
        "/api/v1/ingest/sensors",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_latitude_rejected():
    payload = {
        **VALID_PAYLOAD,
        "location": {
            **VALID_PAYLOAD["location"],
            "latitude": 120.0,
        },
    }

    response = client.post(
        "/api/v1/ingest/sensors",
        json=payload,
    )

    assert response.status_code == 422


def test_mock_prediction_is_deterministic():
    first_response = client.post(
        "/api/v1/ingest/sensors",
        json=VALID_PAYLOAD,
    )

    second_response = client.post(
        "/api/v1/ingest/sensors",
        json=VALID_PAYLOAD,
    )

    assert first_response.status_code == 202
    assert second_response.status_code == 202

    first_prediction = first_response.json()["prediction"]
    second_prediction = second_response.json()["prediction"]

    assert first_prediction["risk_score"] == second_prediction["risk_score"]
    assert first_prediction["risk_level"] == second_prediction["risk_level"]

    assert first_prediction["prediction_mode"] == "MOCK"
    assert second_prediction["prediction_mode"] == "MOCK"