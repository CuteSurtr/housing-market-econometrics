"""
Models router for the Housing Market Econometrics API.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import logging
import time
import json
import pandas as pd

from ..core.dependencies import get_db, get_metrics, update_model_metrics
from ..core.security import get_current_active_user
from ..models.crud import (
    ModelRunCRUD, ModelResultCRUD, ModelPerformanceCRUD, 
    HousingDataCRUD, SystemLogCRUD, UserCRUD
)
from ..models.schemas import (
    ModelAnalysisRequest, ModelAnalysisResponse, ModelRun, ModelRunCreate,
    ModelPerformance, ModelPerformanceCreate, BaseResponse, ErrorResponse,
    PaginationParams, PaginatedResponse
)
from econometric.models.gjr_garch import GJRGARCHModel
from econometric.models.regime_switching import RegimeSwitchingModel
from econometric.models.jump_diffusion import JumpDiffusionModel
from econometric.models.transfer_function import TransferFunctionModel
from econometric.analytics.diagnostics import ModelDiagnostics
from econometric.analytics.risk_metrics import RiskMetricsCalculator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


def get_model_instance(model_type: str):
    """Get model instance based on type."""
    models = {
        "gjr_garch": GJRGARCHModel(),
        "regime_switching": RegimeSwitchingModel(),
        "jump_diffusion": JumpDiffusionModel(),
        "transfer_function": TransferFunctionModel()
    }
    return models.get(model_type)


async def run_model_analysis_background(
    run_id: str,
    model_type: str,
    data_source: str,
    start_date: Optional[date],
    end_date: Optional[date],
    parameters: Optional[Dict[str, Any]],
    include_forecasts: bool,
    forecast_horizon: int,
    user_id: int,
    db: Session
):
    """Background task to run model analysis."""
    try:
        start_time = time.time()
        
        # Update model run status to running
        ModelRunCRUD.update(db, run_id, {"status": "running"})
        
        # Get housing data
        if not start_date:
            start_date = date(2020, 1, 1)
        if not end_date:
            end_date = date.today()
        
        housing_data = HousingDataCRUD.get_by_date_range(db, start_date, end_date)
        if not housing_data:
            raise Exception("No housing data available for analysis")
        
        # Prepare data
        data_df = pd.DataFrame([{
            'date': data.date,
            'shiller_return': data.shiller_return,
            'shiller_index': data.shiller_index,
            'fed_change': data.fed_change,
            'fed_level': data.fed_level,
            'fed_vol': data.fed_vol
        } for data in housing_data])
        
        # Get model instance
        model = get_model_instance(model_type)
        if not model:
            raise Exception(f"Model type {model_type} not supported")
        
        # Fit model
        if model_type == "transfer_function":
            # For transfer function, we need both input and output series
            model.fit(data_df['shiller_return'], data_df['fed_change'])
            series = data_df['shiller_return']
        else:
            # For other models, use the main series
            series = data_df['shiller_return'] if model_type != "jump_diffusion" else data_df['shiller_index']
            model.fit(series)
        
        # Generate forecasts if requested
        forecasts = None
        if include_forecasts:
            if model_type == "transfer_function":
                # Generate future input (assuming fed_change continues at recent average)
                future_input = pd.Series([data_df['fed_change'].mean()] * forecast_horizon)
                forecast_values = model.forecast(future_input)
            else:
                forecast_values = model.forecast(horizon=forecast_horizon)
            
            # Create forecast records
            forecast_date = datetime.now().date()
            forecasts = []
            for i, value in enumerate(forecast_values):
                forecast_data = {
                    'model_run_id': run_id,
                    'forecast_date': forecast_date + timedelta(days=i*30),  # Monthly forecasts
                    'horizon': i + 1,
                    'predicted_value': float(value),
                    'confidence_lower': None,  # Would need model-specific confidence intervals
                    'confidence_upper': None
                }
                forecasts.append(forecast_data)
        
        # Calculate model diagnostics
        diagnostics = ModelDiagnostics()
        diagnostic_results = diagnostics.run_diagnostics(model, series)
        
        # Calculate risk metrics
        risk_calculator = RiskMetricsCalculator()
        risk_metrics = risk_calculator.calculate_risk_metrics(series)
        
        # Prepare results
        results = {
            'model_type': model_type,
            'data_points': len(data_df),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'parameters': parameters,
            'diagnostics': diagnostic_results,
            'risk_metrics': risk_metrics,
            'model_summary': {
                'aic': getattr(model, 'aic', None),
                'bic': getattr(model, 'bic', None),
                'log_likelihood': getattr(model, 'log_likelihood', None)
            }
        }
        
        # Save model results
        ModelResultCRUD.create(db, {
            'model_run_id': run_id,
            'result_type': 'full_analysis',
            'result_data': results
        })
        
        # Calculate model performance metrics
        if hasattr(model, 'aic') and model.aic is not None:
            performance = ModelPerformanceCreate(
                model_type=model_type,
                metric_name='aic',
                metric_value=float(model.aic),
                test_date=date.today(),
                parameters=parameters
            )
            ModelPerformanceCRUD.create(db, performance)
        
        # Update model run with results
        execution_time = time.time() - start_time
        ModelRunCRUD.update(db, run_id, {
            'status': 'completed',
            'results': results,
            'execution_time': execution_time,
            'completed_at': datetime.utcnow()
        })
        
        # Update metrics
        update_model_metrics(model_type, execution_time)
        
        logger.info(f"Model analysis completed for run {run_id}")
        
    except Exception as e:
        logger.error(f"Model analysis failed for run {run_id}: {e}")
        ModelRunCRUD.update(db, run_id, {
            'status': 'failed',
            'results': {'error': str(e)}
        })
        
        # Log error
        SystemLogCRUD.create(db, {
            'level': 'ERROR',
            'message': f'Model analysis failed: {str(e)}',
            'module': 'models',
            'user_id': user_id
        })


@router.post("/analyze", response_model=ModelAnalysisResponse)
async def run_model_analysis(
    analysis_request: ModelAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Run econometric model analysis.
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
        if analysis_request.model_type not in ["gjr_garch", "regime_switching", "jump_diffusion", "transfer_function"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model type"
            )
        
        # Create model run record
        model_run = ModelRunCRUD.create(db, user.id, {
            'model_type': analysis_request.model_type,
            'parameters': analysis_request.parameters
        })
        
        # Start background analysis task
        background_tasks.add_task(
            run_model_analysis_background,
            model_run.id,
            analysis_request.model_type,
            analysis_request.data_source,
            analysis_request.start_date,
            analysis_request.end_date,
            analysis_request.parameters,
            analysis_request.include_forecasts,
            analysis_request.forecast_horizon,
            user.id,
            db
        )
        
        logger.info(f"Model analysis started for user {current_user}, run_id: {model_run.run_id}")
        
        return {
            'run_id': model_run.run_id,
            'model_type': analysis_request.model_type,
            'status': 'pending',
            'message': 'Model analysis started'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting model analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start model analysis"
        )


@router.get("/status/{run_id}", response_model=ModelAnalysisResponse)
async def get_model_analysis_status(
    run_id: str,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get the status of a model analysis job.
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
                detail="Model run not found"
            )
        
        # Check if user owns this run
        if model_run.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this model run"
            )
        
        # Get model results if completed
        model_results = None
        if model_run.status == "completed":
            model_results = ModelResultCRUD.get_by_model_run(db, model_run.id)
        
        return {
            'run_id': model_run.run_id,
            'model_type': model_run.model_type,
            'status': model_run.status,
            'execution_time': model_run.execution_time,
            'results': model_run.results,
            'created_at': model_run.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model analysis status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model analysis status"
        )


@router.get("/history", response_model=PaginatedResponse)
async def get_model_analysis_history(
    page: int = 1,
    size: int = 10,
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get model analysis history for the current user.
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
        logger.error(f"Error getting model analysis history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model analysis history"
        )


@router.get("/performance", response_model=List[ModelPerformance])
async def get_model_performance(
    model_type: Optional[str] = None,
    metric_name: Optional[str] = None,
    limit: int = 100,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get model performance metrics.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get performance records
        if model_type:
            performance_records = ModelPerformanceCRUD.get_by_model_type(db, model_type, limit)
        else:
            # Get all performance records (limited)
            performance_records = []
            for mt in ["gjr_garch", "regime_switching", "jump_diffusion", "transfer_function"]:
                records = ModelPerformanceCRUD.get_by_model_type(db, mt, limit // 4)
                performance_records.extend(records)
        
        # Filter by metric if specified
        if metric_name:
            performance_records = [p for p in performance_records if p.metric_name == metric_name]
        
        return performance_records
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model performance"
        )


@router.post("/compare", response_model=Dict[str, Any])
async def compare_models(
    model_types: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Compare multiple econometric models.
    """
    try:
        # Get user
        user = UserCRUD.get_by_username(db, current_user)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Validate model types
        valid_models = ["gjr_garch", "regime_switching", "jump_diffusion", "transfer_function"]
        invalid_models = [mt for mt in model_types if mt not in valid_models]
        if invalid_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model types: {invalid_models}"
            )
        
        # Get housing data
        if not start_date:
            start_date = date(2020, 1, 1)
        if not end_date:
            end_date = date.today()
        
        housing_data = HousingDataCRUD.get_by_date_range(db, start_date, end_date)
        if not housing_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No housing data available for comparison"
            )
        
        # Prepare data
        data_df = pd.DataFrame([{
            'date': data.date,
            'shiller_return': data.shiller_return,
            'shiller_index': data.shiller_index,
            'fed_change': data.fed_change,
            'fed_level': data.fed_level
        } for data in housing_data])
        
        # Run models and collect results
        comparison_results = {}
        
        for model_type in model_types:
            try:
                model = get_model_instance(model_type)
                if not model:
                    continue
                
                # Fit model
                if model_type == "transfer_function":
                    model.fit(data_df['shiller_return'], data_df['fed_change'])
                    series = data_df['shiller_return']
                else:
                    series = data_df['shiller_return'] if model_type != "jump_diffusion" else data_df['shiller_index']
                    model.fit(series)
                
                # Collect model metrics
                model_results = {
                    'aic': getattr(model, 'aic', None),
                    'bic': getattr(model, 'bic', None),
                    'log_likelihood': getattr(model, 'log_likelihood', None),
                    'data_points': len(data_df)
                }
                
                # Calculate diagnostics
                diagnostics = ModelDiagnostics()
                diagnostic_results = diagnostics.run_diagnostics(model, series)
                model_results['diagnostics'] = diagnostic_results
                
                comparison_results[model_type] = model_results
                
            except Exception as e:
                logger.error(f"Error running model {model_type}: {e}")
                comparison_results[model_type] = {'error': str(e)}
        
        return {
            'comparison_date': datetime.now().isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'models_compared': model_types,
            'results': comparison_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare models"
        )


@router.delete("/{run_id}", response_model=BaseResponse)
async def delete_model_run(
    run_id: str,
    current_user: str = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete a model run and all associated results.
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
                detail="Model run not found"
            )
        
        # Check if user owns this run
        if model_run.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this model run"
            )
        
        # Delete associated results first
        ModelResultCRUD.delete_by_model_run(db, model_run.id)
        
        # Delete model run
        success = ModelRunCRUD.delete(db, model_run.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete model run"
            )
        
        logger.info(f"Model run deleted: {run_id}")
        return {
            'success': True,
            'message': 'Model run deleted successfully'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model run: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model run"
        )


@router.get("/available")
async def get_available_models():
    """
    Get list of available econometric models with descriptions.
    """
    return {
        'models': [
            {
                'type': 'gjr_garch',
                'name': 'GJR-GARCH Model',
                'description': 'Glosten-Jagannathan-Runkle GARCH model for volatility modeling with leverage effects',
                'suitable_for': ['Volatility forecasting', 'Risk modeling', 'Financial time series'],
                'parameters': ['p', 'q', 'leverage'],
                'advantages': ['Captures leverage effects', 'Good for volatility clustering'],
                'limitations': ['Assumes normal distribution', 'May not capture structural breaks']
            },
            {
                'type': 'regime_switching',
                'name': 'Regime Switching Model',
                'description': 'Markov regime switching model for structural breaks and regime changes',
                'suitable_for': ['Structural break detection', 'Regime identification', 'Policy analysis'],
                'parameters': ['regimes', 'transition_probabilities', 'regime_means'],
                'advantages': ['Captures structural breaks', 'Flexible for different regimes'],
                'limitations': ['Computationally intensive', 'Requires sufficient data per regime']
            },
            {
                'type': 'jump_diffusion',
                'name': 'Jump Diffusion Model',
                'description': 'Merton jump diffusion model for sudden price movements and rare events',
                'suitable_for': ['Jump risk modeling', 'Event analysis', 'Extreme value modeling'],
                'parameters': ['jump_intensity', 'jump_size_mean', 'jump_size_std'],
                'advantages': ['Captures sudden jumps', 'Good for rare events'],
                'limitations': ['Complex parameter estimation', 'May overfit to jumps']
            },
            {
                'type': 'transfer_function',
                'name': 'Transfer Function Model',
                'description': 'Transfer function model for input-output relationships and causal modeling',
                'suitable_for': ['Policy impact analysis', 'Causal modeling', 'Input-output relationships'],
                'parameters': ['input_lags', 'output_lags', 'transfer_function_order'],
                'advantages': ['Captures causal relationships', 'Good for policy analysis'],
                'limitations': ['Requires exogenous input', 'May be sensitive to input quality']
            }
        ]
    }


@router.get("/health")
async def models_health_check():
    """
    Models service health check.
    """
    return {
        'status': 'healthy',
        'service': 'models',
        'available_models': 4,
        'timestamp': datetime.utcnow().isoformat()
    }
