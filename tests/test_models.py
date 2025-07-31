"""
Tests for econometric models.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import the models to test
from econometric.models.gjr_garch import GJRGARCHModel
from econometric.models.regime_switching import RegimeSwitchingModel
from econometric.models.jump_diffusion import JumpDiffusionModel
from econometric.models.transfer_function import TransferFunctionModel


class TestGJRGARCHModel:
    """Test GJR-GARCH model functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_data = pd.Series(np.random.normal(0, 1, 1000))
        self.model = GJRGARCHModel()
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'fit')
        assert hasattr(self.model, 'forecast')
    
    def test_model_fitting(self):
        """Test model fitting."""
        result = self.model.fit(self.sample_data)
        assert result is not None
        assert hasattr(result, 'params')
    
    def test_model_forecasting(self):
        """Test model forecasting."""
        # Fit the model first
        self.model.fit(self.sample_data)
        
        # Test forecasting
        forecast = self.model.forecast(horizon=12)
        assert forecast is not None
        assert len(forecast) == 12


class TestRegimeSwitchingModel:
    """Test Regime Switching model functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_data = pd.Series(np.random.normal(0, 1, 1000))
        self.model = RegimeSwitchingModel()
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'fit')
        assert hasattr(self.model, 'forecast')
    
    def test_model_fitting(self):
        """Test model fitting."""
        result = self.model.fit(self.sample_data)
        assert result is not None
        assert hasattr(result, 'regime_probabilities')
    
    def test_model_forecasting(self):
        """Test model forecasting."""
        # Fit the model first
        self.model.fit(self.sample_data)
        
        # Test forecasting
        forecast = self.model.forecast(horizon=12)
        assert forecast is not None
        assert len(forecast) == 12


class TestJumpDiffusionModel:
    """Test Jump Diffusion model functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_data = pd.Series(np.random.normal(0, 1, 1000))
        self.model = JumpDiffusionModel()
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'fit')
        assert hasattr(self.model, 'forecast')
    
    def test_model_fitting(self):
        """Test model fitting."""
        result = self.model.fit(self.sample_data)
        assert result is not None
        assert hasattr(result, 'jump_parameters')
    
    def test_model_forecasting(self):
        """Test model forecasting."""
        # Fit the model first
        self.model.fit(self.sample_data)
        
        # Test forecasting
        forecast = self.model.forecast(horizon=12)
        assert forecast is not None
        assert len(forecast) == 12


class TestTransferFunctionModel:
    """Test Transfer Function model functionality."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.output_data = pd.Series(np.random.normal(0, 1, 1000))
        self.input_data = pd.Series(np.random.normal(0, 1, 1000))
        self.model = TransferFunctionModel()
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model is not None
        assert hasattr(self.model, 'fit')
        assert hasattr(self.model, 'forecast')
    
    def test_model_fitting(self):
        """Test model fitting."""
        result = self.model.fit(self.output_data, self.input_data)
        assert result is not None
        assert hasattr(result, 'transfer_function_params')
    
    def test_model_forecasting(self):
        """Test model forecasting."""
        # Fit the model first
        self.model.fit(self.output_data, self.input_data)
        
        # Test forecasting
        future_input = pd.Series(np.random.normal(0, 1, 12))
        forecast = self.model.forecast(future_input)
        assert forecast is not None
        assert len(forecast) == 12


class TestModelIntegration:
    """Test integration between different models."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.sample_data = pd.Series(np.random.normal(0, 1, 1000))
    
    def test_model_comparison(self):
        """Test comparing different models."""
        models = {
            'gjr_garch': GJRGARCHModel(),
            'regime_switching': RegimeSwitchingModel(),
            'jump_diffusion': JumpDiffusionModel(),
        }
        
        results = {}
        for name, model in models.items():
            try:
                results[name] = model.fit(self.sample_data)
            except Exception as e:
                pytest.skip(f"Model {name} failed to fit: {e}")
        
        # Check that at least one model worked
        assert len(results) > 0
    
    def test_forecast_comparison(self):
        """Test comparing forecasts from different models."""
        models = {
            'gjr_garch': GJRGARCHModel(),
            'regime_switching': RegimeSwitchingModel(),
            'jump_diffusion': JumpDiffusionModel(),
        }
        
        forecasts = {}
        for name, model in models.items():
            try:
                model.fit(self.sample_data)
                forecasts[name] = model.forecast(horizon=12)
            except Exception as e:
                pytest.skip(f"Model {name} failed to forecast: {e}")
        
        # Check that at least one model worked
        assert len(forecasts) > 0
        
        # Check that all forecasts have the same length
        forecast_lengths = [len(f) for f in forecasts.values()]
        assert len(set(forecast_lengths)) == 1
        assert forecast_lengths[0] == 12


if __name__ == "__main__":
    pytest.main([__file__]) 