from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.map import router as map_router
from app.api.routes import router as routes_router
from app.api.sensors import router as sensors_router
from app.core.config import settings
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Backend orchestration API for PRAVAHA decision support. "
        "Risk, confidence, provenance and authority closures are "
        "kept separate."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

app.include_router(
    map_router,
    prefix="/api/v1",
    tags=["Map Intelligence"],
)

app.include_router(
    routes_router,
    prefix="/api/v1",
    tags=["Routes"],
)
