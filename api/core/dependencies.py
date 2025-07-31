"""
Dependencies for the Housing Market Econometrics API.
"""

import os
import logging
from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis
from prometheus_client import Counter, Histogram, Gauge
from contextlib import contextmanager

from .config import get_database_url, get_redis_url, settings

logger = logging.getLogger(__name__)

# Database setup
SQLALCHEMY_DATABASE_URL = get_database_url()

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Redis setup
redis_client: Optional[redis.Redis] = None


def init_database():
    """Initialize database connection."""
    global engine, SessionLocal
    
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def init_redis():
    """Initialize Redis connection."""
    global redis_client
    
    try:
        redis_url = get_redis_url()
        redis_client = redis.from_url(redis_url)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Get database session as context manager."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> Optional[redis.Redis]:
    """Get Redis client."""
    global redis_client
    if redis_client is None:
        init_redis()
    return redis_client


def cleanup_connections():
    """Clean up database and Redis connections."""
    global redis_client
    
    try:
        if redis_client:
            redis_client.close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.error(f"Error closing Redis connection: {e}")
    
    try:
        engine.dispose()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database connection: {e}")


# Prometheus metrics
class Metrics:
    """Prometheus metrics for the API."""
    
    def __init__(self):
        # HTTP request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'path', 'status']
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'path']
        )
        
        # Model execution metrics
        self.econometric_model_execution_seconds = Histogram(
            'econometric_model_execution_seconds',
            'Econometric model execution time in seconds',
            ['model_type']
        )
        
        self.econometric_model_accuracy = Gauge(
            'econometric_model_accuracy',
            'Econometric model accuracy',
            ['model_type']
        )
        
        # Forecast metrics
        self.forecast_generation_total = Counter(
            'forecast_generation_total',
            'Total number of forecasts generated',
            ['model_type']
        )
        
        # Data processing metrics
        self.data_processing_records_total = Counter(
            'data_processing_records_total',
            'Total number of records processed',
            ['operation']
        )
        
        # Database metrics
        self.database_connections = Gauge(
            'database_connections',
            'Number of active database connections'
        )
        
        # Redis metrics
        self.redis_memory_used_bytes = Gauge(
            'redis_memory_used_bytes',
            'Redis memory usage in bytes'
        )


# Global metrics instance
_metrics: Optional[Metrics] = None


def get_metrics() -> Optional[Metrics]:
    """Get metrics instance."""
    global _metrics
    if _metrics is None and settings.enable_metrics:
        _metrics = Metrics()
    return _metrics


def update_model_metrics(model_type: str, execution_time: float, accuracy: Optional[float] = None):
    """Update model execution metrics."""
    metrics = get_metrics()
    if metrics:
        metrics.econometric_model_execution_seconds.labels(model_type=model_type).observe(execution_time)
        if accuracy is not None:
            metrics.econometric_model_accuracy.labels(model_type=model_type).set(accuracy)


def update_forecast_metrics(model_type: str):
    """Update forecast generation metrics."""
    metrics = get_metrics()
    if metrics:
        metrics.forecast_generation_total.labels(model_type=model_type).inc()


def update_data_processing_metrics(operation: str, record_count: int):
    """Update data processing metrics."""
    metrics = get_metrics()
    if metrics:
        metrics.data_processing_records_total.labels(operation=operation).inc(record_count)


def update_database_metrics(connection_count: int):
    """Update database connection metrics."""
    metrics = get_metrics()
    if metrics:
        metrics.database_connections.set(connection_count)


def update_redis_metrics():
    """Update Redis metrics."""
    metrics = get_metrics()
    if metrics and redis_client:
        try:
            info = redis_client.info('memory')
            memory_used = info.get('used_memory', 0)
            metrics.redis_memory_used_bytes.set(memory_used)
        except Exception as e:
            logger.error(f"Error updating Redis metrics: {e}")


# Health check functions
def check_database_health() -> bool:
    """Check database health."""
    try:
        with get_db_context() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def check_redis_health() -> bool:
    """Check Redis health."""
    try:
        redis_client = get_redis()
        if redis_client:
            redis_client.ping()
            return True
        return False
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


def get_system_health() -> dict:
    """Get overall system health."""
    return {
        "database": check_database_health(),
        "redis": check_redis_health(),
        "metrics_enabled": settings.enable_metrics
    }
