from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_golden_sensor_event_updates_state_intelligence_and_routing():
    normal = post_sensor_event(
        "2026-09-09T08:00:00Z",
        rainfall=4.0,
        soil=34.0,
    )
    warning = post_sensor_event(
        "2026-09-09T09:00:00Z",
        rainfall=48.0,
        soil=82.0,
    )
    severe = post_sensor_event(
        "2026-09-09T09:30:00Z",
        rainfall=72.0,
        soil=96.0,
    )

    assert normal["status"] == "accepted"
    assert normal["canonical_location"]["latitude"] == 30.329
    assert normal["canonical_location"]["longitude"] == 78.039
    assert normal["provenance"] == "OBSERVED"
    assert normal["fused_state"]["rainfall"]["intensity"]["value"] == 4.0
    assert normal["fused_state"]["soil"]["saturation"] == 0.34

    assert warning["snapshot_id"] != normal["snapshot_id"]
    assert severe["snapshot_id"] != warning["snapshot_id"]
    assert warning["prediction"]["risk_score"] > normal["prediction"]["risk_score"]
    assert severe["prediction"]["risk_score"] > warning["prediction"]["risk_score"]
    assert severe["prediction"]["risk_level"] == "SEVERE"

    latest = client.get("/api/v1/map/intelligence")
    assert latest.status_code == 200
    latest_snapshot = latest.json()
    assert latest_snapshot["snapshot_id"] == severe["snapshot_id"]
    assert latest_snapshot["city"]["operational_status"] == "EMERGENCY"
    assert latest_snapshot["model_metadata"]["confidence"] < normal["prediction"]["confidence"]
    assert any(
        source["status"] == "UNAVAILABLE"
        and source["provenance"] == "MISSING"
        for source in latest_snapshot["source_health"]
    )
    assert latest_snapshot["summary"]["latest_threshold_crossing"] == "NOW SEVERE"

    catchment = client.get(
        "/api/v1/map/catchments/UK-CHM-DEHRADUN-01"
    ).json()
    assert catchment["fused_state"] == "FusedCatchmentState v2.1"
    assert catchment["hydrology"]["runoff_mm"] > 0
    assert catchment["anticipation"]["threshold_window"]["earliest_minutes"] == 0
    assert catchment["landslide"]["risk_level"] == "HIGH"

    drain = client.get("/api/v1/map/drains/D-22").json()
    assert drain["capacity_utilization"] > 1.0
    assert drain["overflow_m3_per_s"] > 0

    road = client.get(
        "/api/v1/map/roads/ROAD-SHELTER-CORRIDOR"
    ).json()
    closed = client.get(
        "/api/v1/map/roads/ROAD-BRIDGE-APPROACH"
    ).json()
    assert road["recommendation"] == "AVOID"
    assert road["authority_closed"] is False
    assert closed["recommendation"] == "CLOSED"
    assert closed["authority_closed"] is True

    route = client.post(
        "/api/v1/routes/safe",
        json=route_payload("SHELTER-SCHOOL-01"),
    ).json()
    assert route["status"] == "ROUTE_FOUND"
    assert route["selected_route"]["route_id"] == "ROUTE-DEMO-001-BYPASS"
    assert route["selected_route"]["unsafe_segments_avoided"] >= 1

    no_route = client.post(
        "/api/v1/routes/safe",
        json=route_payload("DEMO-NO-SAFE-ROUTE"),
    ).json()
    assert no_route["status"] == "NO_SAFE_ROUTE"
    assert any(
        segment["recommendation"] == "AVOID"
        for segment in no_route["blocked_by"]
    )
    assert any(
        segment["recommendation"] == "CLOSED"
        for segment in no_route["blocked_by"]
    )


def test_publisher_payload_uses_external_lat_lon_and_observed_at():
    from scripts.publish_sensor_event import build_payload, normalize_soil

    class Args:
        url = "http://127.0.0.1:8000/api/v1/ingest/sensors"
        device = "UK-SNS-00127"
        rainfall = 48.0
        soil = 0.82
        tilt = 0.0
        lat = 30.329
        lon = 78.039
        village = "Chandrabani"
        ward = "Ward 7"
        observed_at = "2026-09-09T09:00:00Z"
        received_at = None
        provenance = "OBSERVED"

    payload = build_payload(Args())

    assert payload["observed_at"] == "2026-09-09T09:00:00Z"
    assert payload["location"] == {
        "village": "Chandrabani",
        "ward": "Ward 7",
        "lat": 30.329,
        "lon": 78.039,
    }
    assert payload["sensor_metrics"]["rainfall_mm_per_hr"] == 48.0
    assert payload["sensor_metrics"]["soil_moisture_percentage"] == 82.0
    assert payload["sensor_metrics"]["slope_tilt_degrees"] == 0.0
    assert payload["provenance"] == "OBSERVED"
    assert normalize_soil(82.0) == 82.0


def post_sensor_event(
    observed_at: str,
    *,
    rainfall: float,
    soil: float,
) -> dict:
    response = client.post(
        "/api/v1/ingest/sensors",
        json={
            "device_id": "SENSOR-SIM-RAIN-SOIL-01",
            "observed_at": observed_at,
            "received_at": observed_at,
            "location": {
                "village": "Chandrabani",
                "ward": "Ward 7",
                "lat": 30.329,
                "lon": 78.039,
            },
            "sensor_metrics": {
                "rainfall_mm_per_hr": rainfall,
                "soil_moisture_percentage": soil,
                "slope_tilt_degrees": 2.1,
            },
            "provenance": "OBSERVED",
        },
    )
    assert response.status_code == 202
    return response.json()


def route_payload(destination_id: str) -> dict:
    destination = {
        "lon": 78.056,
        "lat": 30.338,
        "place_id": destination_id,
    }
    if destination_id == "DEMO-NO-SAFE-ROUTE":
        destination = {
            "lon": 78.055,
            "lat": 30.342,
            "place_id": destination_id,
        }

    return {
        "origin": {
            "lon": 78.03,
            "lat": 30.32,
        },
        "destination": destination,
        "strategy": "safest",
    }
