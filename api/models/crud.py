"""
CRUD operations for the Housing Market Econometrics API.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, date

from .database import (
    User, APIKey, ModelRun, Forecast, HousingData, ModelResult,
    DataSource, SystemLog, ModelPerformance, DataProcessingJob
)
from .schemas import (
    UserCreate, UserUpdate, APIKeyCreate, ModelRunCreate, ModelRunUpdate,
    ForecastCreate, HousingDataCreate, ModelPerformanceCreate,
    SystemLogCreate, DataProcessingJob
)

logger = logging.getLogger(__name__)


# User CRUD operations
class UserCRUD:
    """CRUD operations for User model."""
    
    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """Get all users with pagination."""
        return db.query(User).offset(skip).limit(limit).all()
    
    @staticmethod
    def create(db: Session, user: UserCreate, hashed_password: str) -> User:
        """Create a new user."""
        db_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def update(db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
        """Update user."""
        db_user = UserCRUD.get_by_id(db, user_id)
        if not db_user:
            return None
        
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_user, field, value)
        
        db.commit()
        db.refresh(db_user)
        return db_user
    
    @staticmethod
    def delete(db: Session, user_id: int) -> bool:
        """Delete user."""
        db_user = UserCRUD.get_by_id(db, user_id)
        if not db_user:
            return False
        
        db.delete(db_user)
        db.commit()
        return True


# API Key CRUD operations
class APIKeyCRUD:
    """CRUD operations for APIKey model."""
    
    @staticmethod
    def get_by_id(db: Session, key_id: int) -> Optional[APIKey]:
        """Get API key by ID."""
        return db.query(APIKey).filter(APIKey.id == key_id).first()
    
    @staticmethod
    def get_by_user(db: Session, user_id: int) -> List[APIKey]:
        """Get all API keys for a user."""
        return db.query(APIKey).filter(APIKey.user_id == user_id).all()
    
    @staticmethod
    def create(db: Session, user_id: int, api_key: APIKeyCreate, key_hash: str) -> APIKey:
        """Create a new API key."""
        db_api_key = APIKey(
            user_id=user_id,
            name=api_key.name,
            key_hash=key_hash
        )
        db.add(db_api_key)
        db.commit()
        db.refresh(db_api_key)
        return db_api_key
    
    @staticmethod
    def update_last_used(db: Session, key_id: int) -> None:
        """Update last used timestamp."""
        db_api_key = APIKeyCRUD.get_by_id(db, key_id)
        if db_api_key:
            db_api_key.last_used = datetime.utcnow()
            db.commit()
    
    @staticmethod
    def delete(db: Session, key_id: int) -> bool:
        """Delete API key."""
        db_api_key = APIKeyCRUD.get_by_id(db, key_id)
        if not db_api_key:
            return False
        
        db.delete(db_api_key)
        db.commit()
        return True


# Model Run CRUD operations
class ModelRunCRUD:
    """CRUD operations for ModelRun model."""
    
    @staticmethod
    def get_by_id(db: Session, run_id: int) -> Optional[ModelRun]:
        """Get model run by ID."""
        return db.query(ModelRun).filter(ModelRun.id == run_id).first()
    
    @staticmethod
    def get_by_run_id(db: Session, run_id: str) -> Optional[ModelRun]:
        """Get model run by run_id."""
        return db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
    
    @staticmethod
    def get_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[ModelRun]:
        """Get model runs for a user."""
        return db.query(ModelRun).filter(ModelRun.user_id == user_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_by_status(db: Session, status: str) -> List[ModelRun]:
        """Get model runs by status."""
        return db.query(ModelRun).filter(ModelRun.status == status).all()
    
    @staticmethod
    def create(db: Session, user_id: int, model_run: ModelRunCreate) -> ModelRun:
        """Create a new model run."""
        run_id = str(uuid.uuid4())
        db_model_run = ModelRun(
            run_id=run_id,
            user_id=user_id,
            model_type=model_run.model_type.value,
            parameters=model_run.parameters
        )
        db.add(db_model_run)
        db.commit()
        db.refresh(db_model_run)
        return db_model_run
    
    @staticmethod
    def update(db: Session, run_id: int, model_run_update: ModelRunUpdate) -> Optional[ModelRun]:
        """Update model run."""
        db_model_run = ModelRunCRUD.get_by_id(db, run_id)
        if not db_model_run:
            return None
        
        update_data = model_run_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_model_run, field, value)
        
        db.commit()
        db.refresh(db_model_run)
        return db_model_run
    
    @staticmethod
    def delete(db: Session, run_id: int) -> bool:
        """Delete model run."""
        db_model_run = ModelRunCRUD.get_by_id(db, run_id)
        if not db_model_run:
            return False
        
        db.delete(db_model_run)
        db.commit()
        return True


# Forecast CRUD operations
class ForecastCRUD:
    """CRUD operations for Forecast model."""
    
    @staticmethod
    def get_by_id(db: Session, forecast_id: int) -> Optional[Forecast]:
        """Get forecast by ID."""
        return db.query(Forecast).filter(Forecast.id == forecast_id).first()
    
    @staticmethod
    def get_by_model_run(db: Session, model_run_id: int) -> List[Forecast]:
        """Get forecasts for a model run."""
        return db.query(Forecast).filter(Forecast.model_run_id == model_run_id).all()
    
    @staticmethod
    def get_by_date_range(db: Session, start_date: date, end_date: date) -> List[Forecast]:
        """Get forecasts by date range."""
        return db.query(Forecast).filter(
            and_(
                Forecast.forecast_date >= start_date,
                Forecast.forecast_date <= end_date
            )
        ).all()
    
    @staticmethod
    def create(db: Session, forecast: ForecastCreate) -> Forecast:
        """Create a new forecast."""
        db_forecast = Forecast(**forecast.dict())
        db.add(db_forecast)
        db.commit()
        db.refresh(db_forecast)
        return db_forecast
    
    @staticmethod
    def create_bulk(db: Session, forecasts: List[ForecastCreate]) -> List[Forecast]:
        """Create multiple forecasts."""
        db_forecasts = [Forecast(**forecast.dict()) for forecast in forecasts]
        db.add_all(db_forecasts)
        db.commit()
        for forecast in db_forecasts:
            db.refresh(forecast)
        return db_forecasts
    
    @staticmethod
    def delete_by_model_run(db: Session, model_run_id: int) -> int:
        """Delete all forecasts for a model run."""
        result = db.query(Forecast).filter(Forecast.model_run_id == model_run_id).delete()
        db.commit()
        return result


# Housing Data CRUD operations
class HousingDataCRUD:
    """CRUD operations for HousingData model."""
    
    @staticmethod
    def get_by_id(db: Session, data_id: int) -> Optional[HousingData]:
        """Get housing data by ID."""
        return db.query(HousingData).filter(HousingData.id == data_id).first()
    
    @staticmethod
    def get_by_date_range(db: Session, start_date: date, end_date: date, region: Optional[str] = None) -> List[HousingData]:
        """Get housing data by date range."""
        query = db.query(HousingData).filter(
            and_(
                HousingData.date >= start_date,
                HousingData.date <= end_date
            )
        )
        if region:
            query = query.filter(HousingData.region == region)
        return query.order_by(HousingData.date).all()
    
    @staticmethod
    def get_latest(db: Session, region: Optional[str] = None, limit: int = 100) -> List[HousingData]:
        """Get latest housing data."""
        query = db.query(HousingData)
        if region:
            query = query.filter(HousingData.region == region)
        return query.order_by(desc(HousingData.date)).limit(limit).all()
    
    @staticmethod
    def create(db: Session, housing_data: HousingDataCreate) -> HousingData:
        """Create new housing data."""
        db_housing_data = HousingData(**housing_data.dict())
        db.add(db_housing_data)
        db.commit()
        db.refresh(db_housing_data)
        return db_housing_data
    
    @staticmethod
    def create_bulk(db: Session, housing_data_list: List[HousingDataCreate]) -> List[HousingData]:
        """Create multiple housing data records."""
        db_housing_data_list = [HousingData(**data.dict()) for data in housing_data_list]
        db.add_all(db_housing_data_list)
        db.commit()
        for data in db_housing_data_list:
            db.refresh(data)
        return db_housing_data_list
    
    @staticmethod
    def update(db: Session, data_id: int, housing_data: HousingDataCreate) -> Optional[HousingData]:
        """Update housing data."""
        db_housing_data = HousingDataCRUD.get_by_id(db, data_id)
        if not db_housing_data:
            return None
        
        update_data = housing_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_housing_data, field, value)
        
        db.commit()
        db.refresh(db_housing_data)
        return db_housing_data
    
    @staticmethod
    def delete(db: Session, data_id: int) -> bool:
        """Delete housing data."""
        db_housing_data = HousingDataCRUD.get_by_id(db, data_id)
        if not db_housing_data:
            return False
        
        db.delete(db_housing_data)
        db.commit()
        return True


# System Log CRUD operations
class SystemLogCRUD:
    """CRUD operations for SystemLog model."""
    
    @staticmethod
    def create(db: Session, log: SystemLogCreate) -> SystemLog:
        """Create a new system log."""
        db_log = SystemLog(**log.dict())
        db.add(db_log)
        db.commit()
        db.refresh(db_log)
        return db_log
    
    @staticmethod
    def get_by_level(db: Session, level: str, limit: int = 100) -> List[SystemLog]:
        """Get logs by level."""
        return db.query(SystemLog).filter(SystemLog.level == level).order_by(desc(SystemLog.created_at)).limit(limit).all()
    
    @staticmethod
    def get_recent(db: Session, limit: int = 100) -> List[SystemLog]:
        """Get recent logs."""
        return db.query(SystemLog).order_by(desc(SystemLog.created_at)).limit(limit).all()


# Model Performance CRUD operations
class ModelPerformanceCRUD:
    """CRUD operations for ModelPerformance model."""
    
    @staticmethod
    def create(db: Session, performance: ModelPerformanceCreate) -> ModelPerformance:
        """Create a new model performance record."""
        db_performance = ModelPerformance(**performance.dict())
        db.add(db_performance)
        db.commit()
        db.refresh(db_performance)
        return db_performance
    
    @staticmethod
    def get_by_model_type(db: Session, model_type: str, limit: int = 100) -> List[ModelPerformance]:
        """Get performance records by model type."""
        return db.query(ModelPerformance).filter(ModelPerformance.model_type == model_type).order_by(desc(ModelPerformance.test_date)).limit(limit).all()
    
    @staticmethod
    def get_latest_by_model_type(db: Session, model_type: str) -> Optional[ModelPerformance]:
        """Get latest performance record for a model type."""
        return db.query(ModelPerformance).filter(ModelPerformance.model_type == model_type).order_by(desc(ModelPerformance.test_date)).first()


# Model Result CRUD operations
class ModelResultCRUD:
    """CRUD operations for ModelResult model."""
    
    @staticmethod
    def get_by_id(db: Session, result_id: int) -> Optional[ModelResult]:
        """Get model result by ID."""
        return db.query(ModelResult).filter(ModelResult.id == result_id).first()
    
    @staticmethod
    def get_by_model_run(db: Session, model_run_id: int) -> List[ModelResult]:
        """Get model results for a model run."""
        return db.query(ModelResult).filter(ModelResult.model_run_id == model_run_id).all()
    
    @staticmethod
    def create(db: Session, result_data: Dict[str, Any]) -> ModelResult:
        """Create a new model result."""
        db_result = ModelResult(**result_data)
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        return db_result
    
    @staticmethod
    def delete_by_model_run(db: Session, model_run_id: int) -> int:
        """Delete all model results for a model run."""
        result = db.query(ModelResult).filter(ModelResult.model_run_id == model_run_id).delete()
        db.commit()
        return result


# Data Processing Job CRUD operations
class DataProcessingJobCRUD:
    """CRUD operations for DataProcessingJob model."""
    
    @staticmethod
    def get_by_id(db: Session, job_id: int) -> Optional[DataProcessingJob]:
        """Get job by ID."""
        return db.query(DataProcessingJob).filter(DataProcessingJob.id == job_id).first()
    
    @staticmethod
    def get_by_job_id(db: Session, job_id: str) -> Optional[DataProcessingJob]:
        """Get job by job_id."""
        return db.query(DataProcessingJob).filter(DataProcessingJob.job_id == job_id).first()
    
    @staticmethod
    def get_by_status(db: Session, status: str) -> List[DataProcessingJob]:
        """Get jobs by status."""
        return db.query(DataProcessingJob).filter(DataProcessingJob.status == status).all()
    
    @staticmethod
    def create(db: Session, job_type: str, parameters: Optional[Dict[str, Any]] = None) -> DataProcessingJob:
        """Create a new data processing job."""
        job_id = str(uuid.uuid4())
        db_job = DataProcessingJob(
            job_id=job_id,
            job_type=job_type,
            parameters=parameters
        )
        db.add(db_job)
        db.commit()
        db.refresh(db_job)
        return db_job
    
    @staticmethod
    def update_status(db: Session, job_id: str, status: str, error_message: Optional[str] = None) -> Optional[DataProcessingJob]:
        """Update job status."""
        db_job = DataProcessingJobCRUD.get_by_job_id(db, job_id)
        if not db_job:
            return None
        
        db_job.status = status
        if status == "running" and not db_job.started_at:
            db_job.started_at = datetime.utcnow()
        elif status in ["completed", "failed"]:
            db_job.completed_at = datetime.utcnow()
        
        if error_message:
            db_job.error_message = error_message
        
        db.commit()
        db.refresh(db_job)
        return db_job
