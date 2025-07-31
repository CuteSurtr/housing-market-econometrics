"""
Pydantic schemas for the Housing Market Econometrics API.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from enum import Enum


# Enums
class ModelType(str, Enum):
    """Supported econometric model types."""
    GJR_GARCH = "gjr_garch"
    REGIME_SWITCHING = "regime_switching"
    JUMP_DIFFUSION = "jump_diffusion"
    TRANSFER_FUNCTION = "transfer_function"


class ModelStatus(str, Enum):
    """Model run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Data processing job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Base schemas
class BaseResponse(BaseModel):
    """Base response schema."""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# User schemas
class UserBase(BaseModel):
    """Base user schema."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r"^[^@]+@[^@]+\.[^@]+$")


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """User update schema."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, regex=r"^[^@]+@[^@]+\.[^@]+$")
    is_active: Optional[bool] = None


class User(UserBase):
    """User response schema."""
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Authentication schemas
class Token(BaseModel):
    """Token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str
    password: str


# API Key schemas
class APIKeyCreate(BaseModel):
    """API key creation schema."""
    name: str = Field(..., min_length=1, max_length=100)


class APIKey(APIKeyCreate):
    """API key response schema."""
    id: int
    user_id: int
    is_active: bool
    created_at: datetime
    last_used: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Model Run schemas
class ModelRunCreate(BaseModel):
    """Model run creation schema."""
    model_type: ModelType
    parameters: Optional[Dict[str, Any]] = None


class ModelRunUpdate(BaseModel):
    """Model run update schema."""
    status: Optional[ModelStatus] = None
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    completed_at: Optional[datetime] = None


class ModelRun(ModelRunCreate):
    """Model run response schema."""
    id: int
    run_id: str
    user_id: int
    status: ModelStatus
    execution_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Forecast schemas
class ForecastCreate(BaseModel):
    """Forecast creation schema."""
    model_run_id: int
    forecast_date: date
    horizon: int = Field(..., ge=1, le=60)
    predicted_value: float
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None


class Forecast(ForecastCreate):
    """Forecast response schema."""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Housing Data schemas
class HousingDataCreate(BaseModel):
    """Housing data creation schema."""
    date: date
    region: Optional[str] = None
    shiller_index: Optional[float] = None
    shiller_return: Optional[float] = None
    zillow_index: Optional[float] = None
    zillow_return: Optional[float] = None
    fed_rate: Optional[float] = None
    fed_change: Optional[float] = None
    fed_level: Optional[float] = None
    fed_vol: Optional[float] = None


class HousingData(HousingDataCreate):
    """Housing data response schema."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


# Model Analysis schemas
class ModelAnalysisRequest(BaseModel):
    """Model analysis request schema."""
    model_type: ModelType
    data_source: str = "housing_data"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    parameters: Optional[Dict[str, Any]] = None
    include_forecasts: bool = True
    forecast_horizon: int = Field(12, ge=1, le=60)


class ModelAnalysisResponse(BaseModel):
    """Model analysis response schema."""
    run_id: str
    model_type: ModelType
    status: ModelStatus
    execution_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    forecasts: Optional[List[Forecast]] = None
    created_at: datetime


# Forecasting schemas
class ForecastRequest(BaseModel):
    """Forecast request schema."""
    model_type: ModelType
    horizon: int = Field(12, ge=1, le=60)
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    parameters: Optional[Dict[str, Any]] = None


class ForecastResponse(BaseModel):
    """Forecast response schema."""
    run_id: str
    model_type: ModelType
    forecasts: List[Forecast]
    summary: Dict[str, Any]


# Data Processing schemas
class DataProcessingRequest(BaseModel):
    """Data processing request schema."""
    operation: str = Field(..., description="Data processing operation")
    input_files: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None


class DataProcessingJob(BaseModel):
    """Data processing job schema."""
    id: int
    job_id: str
    job_type: str
    status: JobStatus
    input_files: Optional[List[str]] = None
    output_files: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# Model Performance schemas
class ModelPerformanceCreate(BaseModel):
    """Model performance creation schema."""
    model_type: ModelType
    metric_name: str
    metric_value: float
    test_date: date
    parameters: Optional[Dict[str, Any]] = None


class ModelPerformance(ModelPerformanceCreate):
    """Model performance response schema."""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# System schemas
class SystemLogCreate(BaseModel):
    """System log creation schema."""
    level: str
    message: str
    module: Optional[str] = None
    user_id: Optional[int] = None
    request_id: Optional[str] = None


class SystemLog(SystemLogCreate):
    """System log response schema."""
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


# Health check schemas
class HealthCheck(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: datetime
    version: str
    database: str
    redis: Optional[str] = None
    services: Dict[str, bool]


# Metrics schemas
class MetricsResponse(BaseModel):
    """Metrics response schema."""
    http_requests_total: Dict[str, int]
    http_request_duration_seconds: Dict[str, float]
    econometric_model_execution_seconds: Dict[str, float]
    forecast_generation_total: Dict[str, int]
    data_processing_records_total: Dict[str, int]
    database_connections: int
    redis_memory_used_bytes: Optional[int] = None


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1)
    size: int = Field(10, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response schema."""
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int


# Validation schemas
class DataValidationRequest(BaseModel):
    """Data validation request schema."""
    data_source: str
    validation_rules: Dict[str, Any]


class DataValidationResponse(BaseModel):
    """Data validation response schema."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    record_count: int
    valid_records: int


# Export schemas
class ExportRequest(BaseModel):
    """Export request schema."""
    data_type: str  # housing_data, forecasts, model_results, etc.
    format: str = "csv"  # csv, json, excel
    filters: Optional[Dict[str, Any]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class ExportResponse(BaseModel):
    """Export response schema."""
    file_url: str
    file_size: int
    record_count: int
    expires_at: datetime
