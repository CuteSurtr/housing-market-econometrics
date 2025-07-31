"""
Transfer Function Model for Housing Market Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class TransferFunctionModel:
    """Transfer Function model for housing market analysis."""
    
    def __init__(self, input_lags: int = 3, output_lags: int = 3):
        """
        Initialize the transfer function model.
        
        Args:
            input_lags: Number of input lags
            output_lags: Number of output lags
        """
        self.input_lags = input_lags
        self.output_lags = output_lags
        self.fitted = False
        self.output_data = None
        self.input_data = None
        self.params = {}
        self.transfer_function_params = {}
        self.residuals = None
        
    def fit(self, output_data: pd.Series, input_data: pd.Series):
        """
        Fit the transfer function model to the data.
        
        Args:
            output_data: Output time series (e.g., housing returns)
            input_data: Input time series (e.g., Fed rate changes)
        """
        try:
            # Clean data
            self.output_data = output_data.dropna()
            self.input_data = input_data.dropna()
            
            # Align data
            common_index = self.output_data.index.intersection(self.input_data.index)
            self.output_data = self.output_data.loc[common_index]
            self.input_data = input_data.loc[common_index]
            
            if len(self.output_data) < 50:
                raise ValueError("Insufficient data for transfer function estimation")
            
            # Prewhiten input series
            self._prewhiten_input()
            
            # Estimate transfer function parameters
            self._estimate_transfer_function()
            
            # Estimate noise model
            self._estimate_noise_model()
            
            self.fitted = True
            
            # Calculate model statistics
            self._calculate_model_statistics()
            
            return self
            
        except Exception as e:
            raise Exception(f"Error fitting transfer function model: {e}")
    
    def _prewhiten_input(self):
        """Prewhiten the input series using ARIMA model."""
        try:
            # Fit ARIMA model to input series
            input_arima = ARIMA(self.input_data, order=(1, 0, 1))
            input_fit = input_arima.fit()
            
            # Get residuals (prewhitened input)
            self.prewhitened_input = input_fit.resid
            
            # Store ARIMA parameters
            self.input_arima_params = {
                'ar_params': input_fit.arparams().tolist() if hasattr(input_fit, 'arparams') else [],
                'ma_params': input_fit.maparams().tolist() if hasattr(input_fit, 'maparams') else [],
                'sigma2': input_fit.sigma2
            }
            
        except Exception as e:
            logger.warning(f"Error in input prewhitening: {e}")
            # Use original input if prewhitening fails
            self.prewhitened_input = self.input_data
            self.input_arima_params = {}
    
    def _estimate_transfer_function(self):
        """Estimate transfer function parameters."""
        try:
            # Calculate cross-correlation function
            ccf = self._calculate_ccf(self.output_data, self.prewhitened_input, max_lag=20)
            
            # Identify transfer function order based on CCF
            self._identify_transfer_function_order(ccf)
            
            # Estimate transfer function parameters using least squares
            self._estimate_parameters()
            
        except Exception as e:
            logger.error(f"Error in transfer function estimation: {e}")
            # Use simple linear model as fallback
            self._simple_linear_model()
    
    def _calculate_ccf(self, y: pd.Series, x: pd.Series, max_lag: int = 20) -> np.ndarray:
        """Calculate cross-correlation function."""
        try:
            ccf = np.zeros(2 * max_lag + 1)
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # Negative lag: x leads y
                    y_lagged = y.iloc[-lag:]
                    x_shifted = x.iloc[:len(y_lagged)]
                else:
                    # Positive lag: y leads x
                    x_lagged = x.iloc[lag:]
                    y_shifted = y.iloc[:len(x_lagged)]
                
                if lag < 0:
                    correlation = np.corrcoef(y_lagged, x_shifted)[0, 1]
                else:
                    correlation = np.corrcoef(y_shifted, x_lagged)[0, 1]
                
                ccf[lag + max_lag] = correlation if not np.isnan(correlation) else 0
            
            return ccf
            
        except Exception as e:
            logger.error(f"Error calculating CCF: {e}")
            return np.zeros(2 * max_lag + 1)
    
    def _identify_transfer_function_order(self, ccf: np.ndarray):
        """Identify transfer function order from CCF."""
        try:
            # Find significant lags
            threshold = 2 / np.sqrt(len(self.output_data))  # Approximate significance threshold
            
            significant_lags = np.where(np.abs(ccf) > threshold)[0] - len(ccf) // 2
            
            # Determine input and output lags
            if len(significant_lags) > 0:
                # Use the range of significant lags
                min_lag = max(0, -min(significant_lags))
                max_lag = max(0, max(significant_lags))
                
                self.input_lags = min(3, max_lag)
                self.output_lags = min(3, min_lag)
            else:
                # Default values
                self.input_lags = 3
                self.output_lags = 3
                
        except Exception as e:
            logger.error(f"Error identifying transfer function order: {e}")
            self.input_lags = 3
            self.output_lags = 3
    
    def _estimate_parameters(self):
        """Estimate transfer function parameters using least squares."""
        try:
            # Create lagged variables
            X = self._create_lagged_matrix()
            y = self.output_data.iloc[max(self.input_lags, self.output_lags):]
            
            # Ensure X and y have same length
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y.iloc[:min_length]
            
            # Least squares estimation
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Store parameters
            self.transfer_function_params = {
                'input_coeffs': params[:self.input_lags].tolist(),
                'output_coeffs': params[self.input_lags:self.input_lags + self.output_lags].tolist(),
                'intercept': params[-1] if len(params) > self.input_lags + self.output_lags else 0
            }
            
        except Exception as e:
            logger.error(f"Error in parameter estimation: {e}")
            self._simple_linear_model()
    
    def _create_lagged_matrix(self) -> np.ndarray:
        """Create matrix of lagged variables."""
        try:
            n_obs = len(self.output_data)
            max_lag = max(self.input_lags, self.output_lags)
            
            # Initialize matrix
            n_cols = self.input_lags + self.output_lags + 1  # +1 for intercept
            X = np.zeros((n_obs - max_lag, n_cols))
            
            # Input lags
            for i in range(self.input_lags):
                X[:, i] = self.prewhitened_input.iloc[max_lag - i - 1:n_obs - i - 1]
            
            # Output lags
            for i in range(self.output_lags):
                X[:, self.input_lags + i] = self.output_data.iloc[max_lag - i - 1:n_obs - i - 1]
            
            # Intercept
            X[:, -1] = 1
            
            return X
            
        except Exception as e:
            logger.error(f"Error creating lagged matrix: {e}")
            return np.zeros((len(self.output_data), 1))
    
    def _simple_linear_model(self):
        """Simple linear model as fallback."""
        try:
            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                self.prewhitened_input, self.output_data
            )
            
            self.transfer_function_params = {
                'input_coeffs': [slope],
                'output_coeffs': [],
                'intercept': intercept
            }
            
        except Exception as e:
            logger.error(f"Error in simple linear model: {e}")
            self.transfer_function_params = {
                'input_coeffs': [0.0],
                'output_coeffs': [],
                'intercept': self.output_data.mean()
            }
    
    def _estimate_noise_model(self):
        """Estimate noise model for residuals."""
        try:
            # Calculate fitted values
            fitted_values = self._calculate_fitted_values()
            
            # Calculate residuals
            self.residuals = self.output_data.iloc[max(self.input_lags, self.output_lags):] - fitted_values
            
            # Fit ARIMA model to residuals
            if len(self.residuals) > 10:
                try:
                    noise_arima = ARIMA(self.residuals, order=(1, 0, 1))
                    noise_fit = noise_arima.fit()
                    
                    self.noise_model_params = {
                        'ar_params': noise_fit.arparams().tolist() if hasattr(noise_fit, 'arparams') else [],
                        'ma_params': noise_fit.maparams().tolist() if hasattr(noise_fit, 'maparams') else [],
                        'sigma2': noise_fit.sigma2
                    }
                except Exception as e:
                    logger.warning(f"Error fitting noise model: {e}")
                    self.noise_model_params = {}
            else:
                self.noise_model_params = {}
                
        except Exception as e:
            logger.error(f"Error in noise model estimation: {e}")
            self.residuals = None
            self.noise_model_params = {}
    
    def _calculate_fitted_values(self) -> np.ndarray:
        """Calculate fitted values."""
        try:
            X = self._create_lagged_matrix()
            params = (self.transfer_function_params['input_coeffs'] + 
                     self.transfer_function_params['output_coeffs'] + 
                     [self.transfer_function_params['intercept']])
            
            return X @ np.array(params)
            
        except Exception as e:
            logger.error(f"Error calculating fitted values: {e}")
            return np.zeros(len(self.output_data))
    
    def _calculate_model_statistics(self):
        """Calculate model statistics."""
        try:
            if self.residuals is not None:
                # Calculate R-squared
                ss_res = np.sum(self.residuals ** 2)
                ss_tot = np.sum((self.output_data - self.output_data.mean()) ** 2)
                self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate AIC and BIC
                n_params = len(self.transfer_function_params['input_coeffs']) + \
                          len(self.transfer_function_params['output_coeffs']) + 1
                n_obs = len(self.residuals)
                
                self.log_likelihood = -0.5 * n_obs * np.log(2 * np.pi) - \
                                    0.5 * n_obs * np.log(self.residuals.var()) - \
                                    0.5 * ss_res / self.residuals.var()
                
                self.aic = 2 * n_params - 2 * self.log_likelihood
                self.bic = np.log(n_obs) * n_params - 2 * self.log_likelihood
            else:
                self.r_squared = 0
                self.log_likelihood = 0
                self.aic = 0
                self.bic = 0
                
        except Exception as e:
            logger.error(f"Error calculating model statistics: {e}")
            self.r_squared = 0
            self.log_likelihood = 0
            self.aic = 0
            self.bic = 0
    
    def forecast(self, future_input: pd.Series) -> np.ndarray:
        """
        Generate forecasts using future input values.
        
        Args:
            future_input: Future input values
            
        Returns:
            Array of forecasts
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            forecasts = []
            current_output = self.output_data.iloc[-self.output_lags:].values
            current_input = self.prewhitened_input.iloc[-self.input_lags:].values
            
            for i, input_val in enumerate(future_input):
                # Create feature vector
                features = []
                
                # Input lags
                for j in range(self.input_lags):
                    if j < len(current_input):
                        features.append(current_input[-(j+1)])
                    else:
                        features.append(0)
                
                # Output lags
                for j in range(self.output_lags):
                    if j < len(current_output):
                        features.append(current_output[-(j+1)])
                    else:
                        features.append(0)
                
                # Intercept
                features.append(1)
                
                # Calculate forecast
                params = (self.transfer_function_params['input_coeffs'] + 
                         self.transfer_function_params['output_coeffs'] + 
                         [self.transfer_function_params['intercept']])
                
                forecast = np.dot(features, params)
                forecasts.append(forecast)
                
                # Update current values
                current_input = np.append(current_input[1:], input_val)
                current_output = np.append(current_output[1:], forecast)
            
            return np.array(forecasts)
            
        except Exception as e:
            raise Exception(f"Error generating forecasts: {e}")
    
    def get_confidence_intervals(self, future_input: pd.Series, confidence_level: float = 0.95) -> List[Dict[str, float]]:
        """
        Get confidence intervals for forecasts.
        
        Args:
            future_input: Future input values
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            List of confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        try:
            forecasts = self.forecast(future_input)
            confidence_intervals = []
            
            # Calculate forecast error variance
            if self.residuals is not None:
                forecast_error_var = self.residuals.var()
            else:
                forecast_error_var = self.output_data.var()
            
            for i, forecast in enumerate(forecasts):
                # Simple confidence interval
                margin = stats.norm.ppf((1 + confidence_level) / 2) * np.sqrt(forecast_error_var * (i + 1))
                
                confidence_intervals.append({
                    'lower': forecast - margin,
                    'upper': forecast + margin
                })
            
            return confidence_intervals
            
        except Exception as e:
            raise Exception(f"Error calculating confidence intervals: {e}")
    
    def get_transfer_function_params(self) -> Dict[str, Any]:
        """Get transfer function parameters."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting parameters")
        return self.transfer_function_params
    
    def get_residuals(self) -> pd.Series:
        """Get model residuals."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting residuals")
        return self.residuals
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        if not self.fitted:
            return {"error": "Model not fitted"}
        
        return {
            "n_observations": len(self.output_data),
            "input_lags": self.input_lags,
            "output_lags": self.output_lags,
            "r_squared": self.r_squared,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "transfer_function_params": self.transfer_function_params,
            "input_arima_params": self.input_arima_params,
            "noise_model_params": self.noise_model_params
        }
