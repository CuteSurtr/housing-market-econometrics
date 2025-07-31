"""
Configuration settings for the Housing Market Econometrics API.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator


class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    app_name: str = "Housing Market Econometrics API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API settings
    api_v1_str: str = "/api/v1"
    docs_url: Optional[str] = "/docs"
    redoc_url: Optional[str] = "/redoc"
    
    # Database settings
    database_url: str = "postgresql://user:password@localhost:5432/housing_econometrics"
    async_database_url: str = "postgresql+asyncpg://user:password@localhost:5432/housing_econometrics"
    
    # Security settings
    secret_key: str = "your-secret-key-here-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS settings
    backend_cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Redis settings
    redis_url: str = "redis://localhost:6379/0"
    
    # Monitoring settings
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # External API settings
    fred_api_key: Optional[str] = None
    zillow_api_key: Optional[str] = None
    
    # File paths
    data_dir: str = "./data"
    models_dir: str = "./models"
    results_dir: str = "./results"
    
    @validator("backend_cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_cors_config() -> dict:
    """Get CORS configuration."""
    return {
        "allow_origins": settings.backend_cors_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


def get_database_url() -> str:
    """Get database URL from environment or settings."""
    return os.getenv("DATABASE_URL", settings.database_url)


def get_async_database_url() -> str:
    """Get async database URL from environment or settings."""
    return os.getenv("ASYNC_DATABASE_URL", settings.async_database_url)


def get_redis_url() -> str:
    """Get Redis URL from environment or settings."""
    return os.getenv("REDIS_URL", settings.redis_url)


def get_secret_key() -> str:
    """Get secret key from environment or settings."""
    return os.getenv("SECRET_KEY", settings.secret_key)


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.environment.lower() == "development"


def is_production() -> bool:
    """Check if running in production mode."""
    return settings.environment.lower() == "production"


def get_log_level() -> str:
    """Get log level from environment or settings."""
    return os.getenv("LOG_LEVEL", settings.log_level)
