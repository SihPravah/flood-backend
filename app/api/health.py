from fastapi import APIRouter

from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "pravaha-backend",
    }


@router.get("/ready")
def readiness():
    return {
        "status": "ready",
        "service": "pravaha-backend",
        "demo_mode": settings.demo_mode,
        "data_service_configured": bool(settings.data_service_url),
        "ml_service_configured": bool(settings.ml_service_url),
    }
