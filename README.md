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
- `POST /api/v1/routes/safe`

## Sensor Contract

Public raw sensor ingestion accepts `location.lat` and `location.lon`.
The backend normalizes internally to `latitude` and `longitude` where
needed.

## Demo Mode

The default local service uses deterministic demo adapters marked
`SIMULATED`. Operational mode must not silently substitute simulated
data for unavailable observed services.
