from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "PRAVAHA Backend"
    environment: str = "development"

    use_real_ml: bool = False
    ml_service_url: str = "http://127.0.0.1:8001"

    class Config:
        env_file = ".env"


settings = Settings()