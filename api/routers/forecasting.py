"""
Forecasting router for the Housing Market Econometrics API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import logging
import uuid
import time
import pandas as pd

from ..core.dependencies import get_db, get_metrics, update_forecast_metrics
from ..core.security import get_current_active_user
from ..models.crud import ModelRunCRUD, ForecastCRUD, HousingDataCRUD, SystemLogCRUD, UserCRUD
from ..models.schemas import (
    ForecastRequest, ForecastResponse, Forecast, ModelAnalysisRequest, 
    ModelAnalysisResponse, BaseResponse, ErrorResponse, PaginationParams,
    PaginatedResponse
)
from econometric.models.gjr_garch import GJRGARCHModel
from econometric.models.regime_switching import RegimeSwitchingModel
from econometric.models.jump_diffusion import JumpDiffusionModel
from econometric.models.transfer_function import TransferFunctionModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecasting", tags=["forecasting"])


def get_model_instance(model_type: str):
    """Get model instance based on type."""
    models = {
        "gjr_garch": GJRGARCHModel(),
        "regime_switching": RegimeSwitchingModel(),
        "jump_diffusion": JumpDiffusionModel(),
        "transfer_function": TransferFunctionModel()
    }
    return models.get(model_type)


async def run_forecast_background(
    run_id: str,
    model_type: str,
    horizon: int,
    confidence_level: float,
    parameters: Optional[Dict[str, Any]],
    user_id: int,
    db: Session
):
    """Background task to run forecast."""
    try:
        start_time = time.time()
        
        # Update model run status to running
        ModelRunCRUD.update(db, run_id, {"status": "running"})
        
        # Get latest housing data
        housing_data = HousingDataCRUD.get_latest(db, limit=1000)
        if not housing_data:
            raise Exception("No housing data available for forecasting")
        
        # Prepare data
        data_df = pd.DataFrame([{
            'date': data.date,
            'shiller_return': data.shiller_return,
            'shiller_index': data.shiller_index,
            'fed_change': data.fed_change,
            'fed_level': data.fed_level
        } for data in housing_data])
        
        # Get model instance
        model = get_model_instance(model_type)
        if not model:
            raise Exception(f"Model type {model_type} not supported")
        
        # Fit model and generate forecast
        if model_type == "transfer_function":
            # For transfer function, we need both input and output series
            model.fit(data_df['shiller_return'], data_df['fed_change'])
            # Generate future input (assuming fed_change continues at recent average)
            future_input = pd.Series([data_df['fed_change'].mean()] * horizon)
            forecast_values = model.forecast(future_input)
        else:
            # For other models, use the main series
            series = data_df['shiller_return'] if model_type != "jump_diffusion" else data_df['shiller_index']
            model.fit(series)
            forecast_values = model.forecast(horizon=horizon)
        
        # Calculate confidence intervals
        if hasattr(model, 'get_confidence_intervals'):
            confidence_intervals = model.get_confidence_intervals(horizon, confidence_level)
        else:
            # Simple confidence intervals based on historical volatility
            std_dev = series.std() if model_type != "jump_diffusion" else series.pct_change().std()
            confidence_intervals = []
            for i in range(horizon):
                z_score = 1.96  # 95% confidence
                margin = z_score * std_dev * (i + 1) ** 0.5
                confidence_intervals.append({
                    'lower': forecast_values[i] - margin,
                    'upper': forecast_values[i] + margin
                })
        
        # Create forecast records
        forecast_date = datetime.now().date()
        forecasts = []
        for i, (value, ci) in enumerate(zip(forecast_values, confidence_intervals)):
            forecast_data = {
                'model_run_id': run_id,
                'forecast_date': forecast_date + timedelta(days=i*30),  # Monthly forecasts
                'horizon': i + 1,
                'predicted_value': float(value),
                'confidence_lower': float(ci['lower']),
                'confidence_upper': float(ci['upper'])
            }
            forecasts.append(forecast_data)
        
        # Save forecasts to database
        ForecastCRUD.create_bulk(db, forecasts)
        
        # Update model run with results
        execution_time = time.time() - start_time
        results = {
            'forecast_count': len(forecasts),
            'horizon': horizon,
            'confidence_level': confidence_level,
            'model_parameters': parameters,
            'data_points_used': len(data_df)
        }
        
        ModelRunCRUD.update(db, run_id, {
            'status': 'completed',
            'results': results,
            'execution_time': execution_time,
            'completed_at': datetime.utcnow()
        })
        
        # Update metrics
        update_forecast_metrics(model_type)
        
        logger.info(f"Forecast completed for run {run_id}")
        
    except Exception as e:
        logger.error(f"Forecast failed for run {run_id}: {e}")
        ModelRunCRUD.update(db, run_id, {
            'status': 'failed',
            'results': {'error': str(e)}
        })
        
        # Log error
        SystemLogCRUD.create(db, {
            'level': 'ERROR',
            'message': f'Forecast failed: {str(e)}',
            'module': 'forecasting',
            'user_id': user_id
        })


@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(
    forecast_request: ForecastRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Generate a new forecast using the specified model.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate model type
        if forecast_request.model_type not in ["gjr_garch", "regime_switching", "jump_diffusion", "transfer_function"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model type"
            )
        
        # Create model run record
        model_run = ModelRunCRUD.create(db, user.id, {
            'model_type': forecast_request.model_type,
            'parameters': forecast_request.parameters
        })
        
        # Start background forecast task
        background_tasks.add_task(
            run_forecast_background,
            model_run.id,
            forecast_request.model_type,
            forecast_request.horizon,
            forecast_request.confidence_level,
            forecast_request.parameters,
            user.id,
            db
        )
        
        logger.info(f"Forecast generation started for user {current_user}, run_id: {model_run.run_id}")
        
        return {
            'run_id': model_run.run_id,
            'model_type': forecast_request.model_type,
            'status': 'pending',
            'message': 'Forecast generation started'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting forecast generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start forecast generation"
        )


@router.get("/status/{run_id}", response_model=ModelAnalysisResponse)
async def get_forecast_status(
    run_id: str,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the status of a forecast generation job.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get model run
        model_run = ModelRunCRUD.get_by_run_id(db, run_id)
        if not model_run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Forecast run not found"
            )
        
        # Check if user owns this run
        if model_run.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this forecast"
            )
        
        # Get forecasts if completed
        forecasts = None
        if model_run.status == "completed":
            forecasts = ForecastCRUD.get_by_model_run(db, model_run.id)
        
        return {
            'run_id': model_run.run_id,
            'model_type': model_run.model_type,
            'status': model_run.status,
            'execution_time': model_run.execution_time,
            'results': model_run.results,
            'forecasts': forecasts,
            'created_at': model_run.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting forecast status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get forecast status"
        )


@router.get("/history", response_model=PaginatedResponse)
async def get_forecast_history(
    page: int = 1,
    size: int = 10,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get forecast history for the current user.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Calculate pagination
        skip = (page - 1) * size
        
        # Get model runs
        model_runs = ModelRunCRUD.get_by_user(db, user.id, skip=skip, limit=size)
        
        # Apply filters if provided
        if model_type:
            model_runs = [run for run in model_runs if run.model_type == model_type]
        if status:
            model_runs = [run for run in model_runs if run.status == status]
        
        # Get total count
        total_runs = len(ModelRunCRUD.get_by_user(db, user.id))
        
        return {
            'items': model_runs,
            'total': total_runs,
            'page': page,
            'size': size,
            'pages': (total_runs + size - 1) // size
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting forecast history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get forecast history"
        )


@router.get("/{run_id}/forecasts", response_model=List[Forecast])
async def get_forecasts_by_run(
    run_id: str,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get all forecasts for a specific model run.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get model run
        model_run = ModelRunCRUD.get_by_run_id(db, run_id)
        if not model_run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Forecast run not found"
            )
        
        # Check if user owns this run
        if model_run.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this forecast"
            )
        
        # Get forecasts
        forecasts = ForecastCRUD.get_by_model_run(db, model_run.id)
        return forecasts
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting forecasts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get forecasts"
        )


@router.delete("/{run_id}", response_model=BaseResponse)
async def delete_forecast_run(
    run_id: str,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a forecast run and all associated forecasts.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get model run
        model_run = ModelRunCRUD.get_by_run_id(db, run_id)
        if not model_run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Forecast run not found"
            )
        
        # Check if user owns this run
        if model_run.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this forecast"
            )
        
        # Delete associated forecasts first
        ForecastCRUD.delete_by_model_run(db, model_run.id)
        
        # Delete model run
        success = ModelRunCRUD.delete(db, model_run.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete forecast run"
            )
        
        logger.info(f"Forecast run deleted: {run_id}")
        return {
            'success': True,
            'message': 'Forecast run deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting forecast run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete forecast run"
        )


@router.get("/models/available")
async def get_available_models():
    """
    Get list of available forecasting models.
    """
    return {
        'models': [
            {
                'type': 'gjr_garch',
                'name': 'GJR-GARCH Model',
                'description': 'Glosten-Jagannathan-Runkle GARCH model for volatility modeling',
                'suitable_for': ['Volatility forecasting', 'Risk modeling'],
                'parameters': ['p', 'q', 'leverage']
            },
            {
                'type': 'regime_switching',
                'name': 'Regime Switching Model',
                'description': 'Markov regime switching model for structural breaks',
                'suitable_for': ['Structural break detection', 'Regime identification'],
                'parameters': ['regimes', 'transition_probabilities']
            },
            {
                'type': 'jump_diffusion',
                'name': 'Jump Diffusion Model',
                'description': 'Merton jump diffusion model for sudden price movements',
                'suitable_for': ['Jump risk modeling', 'Event analysis'],
                'parameters': ['jump_intensity', 'jump_size_mean', 'jump_size_std']
            },
            {
                'type': 'transfer_function',
                'name': 'Transfer Function Model',
                'description': 'Transfer function model for input-output relationships',
                'suitable_for': ['Policy impact analysis', 'Causal modeling'],
                'parameters': ['input_lags', 'output_lags', 'transfer_function_order']
            }
        ]
    }


@router.get("/health")
async def forecasting_health_check():
    """
    Forecasting service health check.
    """
    return {
        'status': 'healthy',
        'service': 'forecasting',
        'available_models': 4,
        'timestamp': datetime.utcnow().isoformat()
    }
