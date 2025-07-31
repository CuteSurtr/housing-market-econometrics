"""
Housing data router for the Housing Market Econometrics API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import logging
import pandas as pd
import io
import json

from ..core.dependencies import get_db, get_metrics, update_data_processing_metrics
from ..core.security import get_current_active_user
from ..models.crud import HousingDataCRUD, DataProcessingJobCRUD, SystemLogCRUD, UserCRUD
from ..models.schemas import (
    HousingData, HousingDataCreate, BaseResponse, ErrorResponse, 
    PaginationParams, PaginatedResponse, DataValidationRequest, 
    DataValidationResponse, ExportRequest, ExportResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/housing", tags=["housing"])


@router.post("/upload", response_model=BaseResponse)
async def upload_housing_data(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload housing market data from CSV file.
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are supported"
            )
        
        # Read file content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        # Validate required columns
        required_columns = ['date', 'shiller_index']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate returns if not present
        if 'shiller_return' not in df.columns and 'shiller_index' in df.columns:
            df['shiller_return'] = df['shiller_index'].pct_change()
        
        # Prepare data for database
        housing_data_list = []
        for _, row in df.iterrows():
            data = HousingDataCreate(
                date=row['date'].date(),
                region=row.get('region'),
                shiller_index=row.get('shiller_index'),
                shiller_return=row.get('shiller_return'),
                zillow_index=row.get('zillow_index'),
                zillow_return=row.get('zillow_return'),
                fed_rate=row.get('fed_rate'),
                fed_change=row.get('fed_change'),
                fed_level=row.get('fed_level'),
                fed_vol=row.get('fed_vol')
            )
            housing_data_list.append(data)
        
        # Save to database
        saved_data = HousingDataCRUD.create_bulk(db, housing_data_list)
        
        # Update metrics
        update_data_processing_metrics("data_upload", len(saved_data))
        
        logger.info(f"Housing data uploaded by user {current_user}: {len(saved_data)} records")
        
        return {
            'success': True,
            'message': f'Successfully uploaded {len(saved_data)} housing data records'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload housing data"
        )


@router.get("/data", response_model=PaginatedResponse)
async def get_housing_data(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    region: Optional[str] = None,
    page: int = 1,
    size: int = 100,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get housing market data with filtering and pagination.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Set default date range if not provided
        if not start_date:
            start_date = date(2020, 1, 1)
        if not end_date:
            end_date = date.today()
        
        # Get data
        housing_data = HousingDataCRUD.get_by_date_range(db, start_date, end_date, region)
        
        # Apply pagination
        total_count = len(housing_data)
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        paginated_data = housing_data[start_idx:end_idx]
        
        return {
            'items': paginated_data,
            'total': total_count,
            'page': page,
            'size': size,
            'pages': (total_count + size - 1) // size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get housing data"
        )


@router.get("/data/latest", response_model=List[HousingData])
async def get_latest_housing_data(
    limit: int = 100,
    region: Optional[str] = None,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the latest housing market data.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get latest data
        housing_data = HousingDataCRUD.get_latest(db, region, limit)
        return housing_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get latest housing data"
        )


@router.get("/data/{data_id}", response_model=HousingData)
async def get_housing_data_by_id(
    data_id: int,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get specific housing data record by ID.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get data
        housing_data = HousingDataCRUD.get_by_id(db, data_id)
        if not housing_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Housing data not found"
            )
        
        return housing_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting housing data by ID: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get housing data"
        )


@router.post("/data", response_model=HousingData)
async def create_housing_data(
    housing_data: HousingDataCreate,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new housing data record.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create data
        created_data = HousingDataCRUD.create(db, housing_data)
        
        logger.info(f"Housing data created by user {current_user}: {created_data.id}")
        return created_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create housing data"
        )


@router.put("/data/{data_id}", response_model=HousingData)
async def update_housing_data(
    data_id: int,
    housing_data: HousingDataCreate,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing housing data record.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update data
        updated_data = HousingDataCRUD.update(db, data_id, housing_data)
        if not updated_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Housing data not found"
            )
        
        logger.info(f"Housing data updated by user {current_user}: {data_id}")
        return updated_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update housing data"
        )


@router.delete("/data/{data_id}", response_model=BaseResponse)
async def delete_housing_data(
    data_id: int,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a housing data record.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete data
        success = HousingDataCRUD.delete(db, data_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Housing data not found"
            )
        
        logger.info(f"Housing data deleted by user {current_user}: {data_id}")
        return {
            'success': True,
            'message': 'Housing data deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete housing data"
        )


@router.post("/validate", response_model=DataValidationResponse)
async def validate_housing_data(
    validation_request: DataValidationRequest,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Validate housing data for quality and consistency.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get data for validation
        housing_data = HousingDataCRUD.get_latest(db, limit=1000)
        
        errors = []
        warnings = []
        valid_records = 0
        
        for data in housing_data:
            record_valid = True
            
            # Check for missing values
            if data.shiller_index is None:
                errors.append(f"Missing Shiller index for date {data.date}")
                record_valid = False
            
            # Check for negative values
            if data.shiller_index is not None and data.shiller_index < 0:
                errors.append(f"Negative Shiller index for date {data.date}")
                record_valid = False
            
            # Check for extreme values
            if data.shiller_index is not None and data.shiller_index > 1000:
                warnings.append(f"Unusually high Shiller index for date {data.date}: {data.shiller_index}")
            
            # Check for missing returns
            if data.shiller_return is None and data.shiller_index is not None:
                warnings.append(f"Missing return calculation for date {data.date}")
            
            if record_valid:
                valid_records += 1
        
        total_records = len(housing_data)
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'record_count': total_records,
            'valid_records': valid_records
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate housing data"
        )


@router.post("/export", response_model=ExportResponse)
async def export_housing_data(
    export_request: ExportRequest,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Export housing data in various formats.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get data based on filters
        start_date = export_request.start_date or date(2020, 1, 1)
        end_date = export_request.end_date or date.today()
        
        housing_data = HousingDataCRUD.get_by_date_range(db, start_date, end_date)
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'date': data.date,
            'region': data.region,
            'shiller_index': data.shiller_index,
            'shiller_return': data.shiller_return,
            'zillow_index': data.zillow_index,
            'zillow_return': data.zillow_return,
            'fed_rate': data.fed_rate,
            'fed_change': data.fed_change,
            'fed_level': data.fed_level,
            'fed_vol': data.fed_vol
        } for data in housing_data])
        
        # Generate file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"housing_data_{timestamp}"
        
        if export_request.format.lower() == "csv":
            file_path = f"exports/{filename}.csv"
            df.to_csv(file_path, index=False)
        elif export_request.format.lower() == "json":
            file_path = f"exports/{filename}.json"
            df.to_json(file_path, orient='records', date_format='iso')
        elif export_request.format.lower() == "excel":
            file_path = f"exports/{filename}.xlsx"
            df.to_excel(file_path, index=False)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported export format"
            )
        
        # Calculate file size
        import os
        file_size = os.path.getsize(file_path)
        
        # Set expiration (24 hours from now)
        expires_at = datetime.now() + timedelta(hours=24)
        
        logger.info(f"Housing data exported by user {current_user}: {len(housing_data)} records")
        
        return {
            'file_url': f"/downloads/{filename}.{export_request.format.lower()}",
            'file_size': file_size,
            'record_count': len(housing_data),
            'expires_at': expires_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting housing data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export housing data"
        )


@router.get("/summary")
async def get_housing_data_summary(
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics for housing data.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get latest data
        housing_data = HousingDataCRUD.get_latest(db, limit=1000)
        
        if not housing_data:
            return {
                'total_records': 0,
                'date_range': None,
                'regions': [],
                'summary_stats': {}
            }
        
        # Calculate summary statistics
        df = pd.DataFrame([{
            'date': data.date,
            'shiller_index': data.shiller_index,
            'shiller_return': data.shiller_return,
            'zillow_index': data.zillow_index,
            'fed_rate': data.fed_rate
        } for data in housing_data])
        
        # Date range
        date_range = {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        }
        
        # Regions
        regions = list(set([data.region for data in housing_data if data.region]))
        
        # Summary statistics
        summary_stats = {}
        for column in ['shiller_index', 'shiller_return', 'zillow_index', 'fed_rate']:
            if column in df.columns and df[column].notna().any():
                summary_stats[column] = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'count': int(df[column].count())
                }
        
        return {
            'total_records': len(housing_data),
            'date_range': date_range,
            'regions': regions,
            'summary_stats': summary_stats
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting housing data summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get housing data summary"
        )


@router.get("/regions")
async def get_available_regions(
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of available regions in the housing data.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get all data to extract regions
        housing_data = HousingDataCRUD.get_latest(db, limit=10000)
        
        # Extract unique regions
        regions = list(set([data.region for data in housing_data if data.region]))
        regions.sort()
        
        return {
            'regions': regions,
            'count': len(regions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting available regions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get available regions"
        )


@router.get("/health")
async def housing_health_check():
    """
    Housing data service health check.
    """
    return {
        'status': 'healthy',
        'service': 'housing_data',
        'timestamp': datetime.utcnow().isoformat()
    }
