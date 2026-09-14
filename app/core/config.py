from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PRAVAHA Backend"
    environment: str = "development"

    demo_mode: bool = True
    default_catchment_id: str = "UK-CHM-DEHRADUN-01"

    use_real_ml: bool = False
    ml_service_url: str = "http://127.0.0.1:8001"
    data_service_url: str = "http://127.0.0.1:8002"

    request_timeout_seconds: float = 5.0
    request_retry_count: int = 2

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"


settings = Settings()
