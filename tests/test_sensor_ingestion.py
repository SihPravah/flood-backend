from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


VALID_PAYLOAD = {
    "device_id": "SENSOR-SIM-RAIN-SOIL-01",
    "timestamp": "2026-08-30T14:30:00Z",
    "location": {
        "village": "Munnar",
        "ward": "Ward_3",
        "lat": 10.0889,
        "lon": 77.0595,
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
    assert body["device_id"] == "SENSOR-SIM-RAIN-SOIL-01"

    # Backend-added metadata
    assert "observed_at" in body
    assert "received_at" in body
    assert "age_seconds" in body

    assert isinstance(body["age_seconds"], int)
    assert body["age_seconds"] >= 0

    # Snapshot-derived prediction summary should be attached.
    assert "prediction" in body
    assert body["canonical_location"]["latitude"] == 10.0889
    assert body["canonical_location"]["longitude"] == 77.0595
    assert body["fused_state"]["rainfall"]["intensity"]["value"] == 45.5
    assert body["fused_state"]["soil"]["saturation"] == 0.82
    assert body["snapshot_id"]

    prediction = body["prediction"]

    assert prediction["device_id"] == "SENSOR-SIM-RAIN-SOIL-01"

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

    assert prediction["prediction_mode"] == "DEVELOPMENT_FALLBACK"
    assert prediction["model_version"] == "synthetic-development-v1"


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
            "lat": 120.0,
        },
    }

    response = client.post(
        "/api/v1/ingest/sensors",
        json=payload,
    )

    assert response.status_code == 422


def test_legacy_latitude_longitude_payload_is_still_normalized():
    payload = {
        **VALID_PAYLOAD,
        "location": {
            "village": "Munnar",
            "ward": "Ward_3",
            "latitude": 10.0889,
            "longitude": 77.0595,
        },
    }

    response = client.post(
        "/api/v1/ingest/sensors",
        json=payload,
    )

    assert response.status_code == 202
    assert response.json()["canonical_location"] == {
        "village": "Munnar",
        "ward": "Ward_3",
        "latitude": 10.0889,
        "longitude": 77.0595,
    }


def test_observed_at_alias_and_received_at_are_accepted():
    payload = {
        **VALID_PAYLOAD,
        "observed_at": "2026-09-09T08:30:00Z",
        "received_at": "2026-09-09T08:30:05Z",
    }
    payload.pop("timestamp")

    response = client.post(
        "/api/v1/ingest/sensors",
        json=payload,
    )

    assert response.status_code == 202
    body = response.json()
    assert body["observed_at"] == "2026-09-09T08:30:00+00:00"
    assert body["received_at"] == "2026-09-09T08:30:05+00:00"
    assert body["provenance"] == "OBSERVED"


def test_sensor_event_changes_monitoring_snapshot():
    normal_payload = {
        **VALID_PAYLOAD,
        "timestamp": "2026-09-09T08:00:00Z",
        "sensor_metrics": {
            **VALID_PAYLOAD["sensor_metrics"],
            "rainfall_mm_per_hr": 4.0,
            "soil_moisture_percentage": 34.0,
        },
    }
    warning_payload = {
        **VALID_PAYLOAD,
        "timestamp": "2026-09-09T08:30:00Z",
        "sensor_metrics": {
            **VALID_PAYLOAD["sensor_metrics"],
            "rainfall_mm_per_hr": 48.0,
            "soil_moisture_percentage": 82.0,
        },
    }

    normal = client.post(
        "/api/v1/ingest/sensors",
        json=normal_payload,
    ).json()
    warning = client.post(
        "/api/v1/ingest/sensors",
        json=warning_payload,
    ).json()

    assert normal["snapshot_id"] != warning["snapshot_id"]
    assert normal["prediction"]["risk_score"] < warning["prediction"]["risk_score"]
    assert normal["city"]["operational_status"] == "NORMAL"
    assert warning["city"]["operational_status"] == "WARNING"

    latest_snapshot = client.get("/api/v1/map/intelligence").json()
    assert latest_snapshot["snapshot_id"] == warning["snapshot_id"]
    assert latest_snapshot["city"]["operational_status"] == "WARNING"

    route = client.post(
        "/api/v1/routes/safe",
        json={
            "origin": {"lon": 78.0300, "lat": 30.3200},
            "destination": {
                "lon": 78.0520,
                "lat": 30.3350,
                "place_id": "SHELTER-SCHOOL-01",
            },
            "strategy": "safest",
        },
    ).json()
    assert route["status"] == "ROUTE_FOUND"
    assert route["selected_route"]["segments"][0]["road_id"] == (
        "ROAD-HIGHER-GROUND-BYPASS"
    )


def test_snapshot_prediction_summary_is_deterministic():
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

    assert first_prediction["prediction_mode"] == "DEVELOPMENT_FALLBACK"
    assert second_prediction["prediction_mode"] == "DEVELOPMENT_FALLBACK"
