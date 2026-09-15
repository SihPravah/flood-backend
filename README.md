# PRAVAHA Backend

FastAPI orchestration layer for PRAVAHA. The backend exposes stable API
surfaces to the frontend and delegates data fusion and intelligence to
service boundaries instead of duplicating ML logic.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\uvicorn app.main:app --reload
```

OpenAPI is available at `/docs`.

## Current Endpoints

- `GET /api/v1/health`
- `GET /api/v1/ready`
- `POST /api/v1/ingest/sensors`
- `GET /api/v1/map/intelligence`
- `GET /api/v1/map/catchments/{catchment_id}`
- `GET /api/v1/map/drains/{drain_id}`
- `GET /api/v1/map/roads/{road_id}`
- `GET /api/v1/map/sensors/{device_id}`
- `GET /api/v1/map/alerts`
- `GET /api/v1/events`
- `GET /api/v1/system/health`
- `POST /api/v1/routes/safe`

Map, detail, alert, event, route and system-health endpoints accept
`scenario_stage=NORMAL|WATCH|WARNING|SEVERE` in demo mode.

## Sensor Contract

Public raw sensor ingestion accepts `location.lat` and `location.lon`.
The backend normalizes internally to `latitude` and `longitude` where
needed.

## Demo Mode

The default local service uses deterministic demo adapters marked
`SIMULATED`. Operational mode must not silently substitute simulated
data for unavailable observed services.

The deterministic demo namespace is `DEMO-001`, with shared IDs such as
`UK-CHM-DEHRADUN-01`, `D-22`, `ROAD-SHELTER-CORRIDOR`,
`ROAD-BRIDGE-APPROACH`, `SENSOR-SIM-RAIN-SOIL-01`, and
`SHELTER-SCHOOL-01`.

## Sensor Event Demo

With the backend running, publish a canonical IoT-style event:

```powershell
.\.venv\Scripts\python.exe scripts\publish_sensor_event.py --device UK-SNS-00127 --rainfall 48 --soil 0.82 --lat 30.329 --lon 78.039
```

The script sends `observed_at`, external `location.lat/location.lon`, rainfall
intensity, soil moisture, and provenance to `POST /api/v1/ingest/sensors`.
Backend normalizes the event, builds a fused catchment state through its Data
service boundary, calls ML intelligence, records a new monitoring snapshot, and
returns the updated prediction summary. `GET /api/v1/map/intelligence` then
returns the latest snapshot when no `scenario_stage` query is supplied.

## Recent State Retention

The backend keeps a small in-process store of recent map snapshots,
structured events, and route evaluations. This supports local monitoring
and demo review, but it is not durable operational persistence.
