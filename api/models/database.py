"""
Database models for the Housing Market Econometrics API.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from ..core.dependencies import Base


class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    model_runs = relationship("ModelRun", back_populates="user")
    api_keys = relationship("APIKey", back_populates="user")


class APIKey(Base):
    """API key model for external access."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key_hash = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")


class ModelRun(Base):
    """Model run tracking."""
    __tablename__ = "model_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(50), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_type = Column(String(50), nullable=False)  # gjr_garch, regime_switching, etc.
    status = Column(String(20), default="running")  # running, completed, failed
    parameters = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="model_runs")
    forecasts = relationship("Forecast", back_populates="model_run")


class Forecast(Base):
    """Forecast results."""
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False)
    forecast_date = Column(DateTime(timezone=True), nullable=False)
    horizon = Column(Integer, nullable=False)  # Forecast horizon in periods
    predicted_value = Column(Float, nullable=False)
    confidence_lower = Column(Float, nullable=True)
    confidence_upper = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    model_run = relationship("ModelRun", back_populates="forecasts")


class HousingData(Base):
    """Housing market data."""
    __tablename__ = "housing_data"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime(timezone=True), nullable=False, index=True)
    region = Column(String(100), nullable=True)
    shiller_index = Column(Float, nullable=True)
    shiller_return = Column(Float, nullable=True)
    zillow_index = Column(Float, nullable=True)
    zillow_return = Column(Float, nullable=True)
    fed_rate = Column(Float, nullable=True)
    fed_change = Column(Float, nullable=True)
    fed_level = Column(Float, nullable=True)
    fed_vol = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ModelResult(Base):
    """Detailed model results."""
    __tablename__ = "model_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=False)
    result_type = Column(String(50), nullable=False)  # parameters, diagnostics, etc.
    result_data = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DataSource(Base):
    """Data source tracking."""
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    source_type = Column(String(50), nullable=False)  # csv, api, database
    url = Column(String(500), nullable=True)
    last_updated = Column(DateTime(timezone=True), nullable=True)
    record_count = Column(Integer, nullable=True)
    status = Column(String(20), default="active")  # active, inactive, error
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class SystemLog(Base):
    """System logging."""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    level = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, DEBUG
    message = Column(Text, nullable=False)
    module = Column(String(100), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    request_id = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelPerformance(Base):
    """Model performance tracking."""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String(50), nullable=False)
    metric_name = Column(String(50), nullable=False)  # accuracy, rmse, mae, etc.
    metric_value = Column(Float, nullable=False)
    test_date = Column(DateTime(timezone=True), nullable=False)
    parameters = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DataProcessingJob(Base):
    """Data processing job tracking."""
    __tablename__ = "data_processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(50), unique=True, index=True, nullable=False)
    job_type = Column(String(50), nullable=False)  # data_loading, feature_engineering, etc.
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    input_files = Column(JSON, nullable=True)
    output_files = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Indexes for better performance
from sqlalchemy import Index

# Create indexes for common queries
Index('idx_housing_data_date', HousingData.date)
Index('idx_housing_data_region', HousingData.region)
Index('idx_model_runs_user_status', ModelRun.user_id, ModelRun.status)
Index('idx_forecasts_model_run_date', Forecast.model_run_id, Forecast.forecast_date)
Index('idx_system_logs_level_created', SystemLog.level, SystemLog.created_at)
Index('idx_model_performance_type_date', ModelPerformance.model_type, ModelPerformance.test_date)
