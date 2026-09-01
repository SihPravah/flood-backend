from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.sensors import router as sensors_router

app = FastAPI(
    title="PRAVAHA Backend",
    version="0.1.0",
)

app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["System"],
)

app.include_router(
    sensors_router,
    prefix="/api/v1",
    tags=["Sensors"],
)