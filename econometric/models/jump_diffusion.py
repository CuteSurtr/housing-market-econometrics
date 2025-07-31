"""
Jump Diffusion Model for Housing Market Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
from scipy.optimize import minimize
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class JumpDiffusionModel:
    """Merton Jump Diffusion model for housing market analysis."""
    
    def __init__(self):
        """Initialize the jump diffusion model."""
        self.fitted = False
        self.data = None
        self.params = {}
        self.jump_parameters = {}
        self.jump_times = None
        self.jump_sizes = None
        
    def fit(self, data: pd.Series):
        """
        Fit the jump diffusion model to the data.
        
        Args:
            data: Time series data (prices or indices)
        """
        try:
            # Clean data
            self.data = data.dropna()
            
            if len(self.data) < 50:
                raise ValueError("Insufficient data for jump diffusion estimation")
            
            # Calculate returns if data is in levels
            if self.data.min() > 0:  # Likely price data
                self.returns = self.data.pct_change().dropna()
            else:  # Already returns
                self.returns = self.data
            
            # Estimate model parameters
            self._estimate_parameters()
            
            # Detect jumps
            self._detect_jumps()
            
            self.fitted = True
            
            # Calculate model statistics
            self._calculate_model_statistics()
            
            return self
            
        except Exception as e:
            raise Exception(f"Error fitting jump diffusion model: {e}")
    
    def _estimate_parameters(self):
        """Estimate model parameters using maximum likelihood."""
        try:
            # Initial parameter estimates
            mu_0 = self.returns.mean()
            sigma_0 = self.returns.std()
            lambda_0 = 0.1  # Initial jump intensity
            mu_j_0 = 0.0    # Initial jump mean
            sigma_j_0 = sigma_0  # Initial jump volatility
            
            # Parameter bounds
            bounds = [
                (-0.1, 0.1),      # mu
                (0.001, 0.1),     # sigma
                (0.001, 1.0),     # lambda
                (-0.1, 0.1),      # mu_j
                (0.001, 0.1)      # sigma_j
            ]
            
            # Initial parameter vector
            params_0 = [mu_0, sigma_0, lambda_0, mu_j_0, sigma_j_0]
            
            # Optimize using maximum likelihood
            result = minimize(
                self._negative_log_likelihood,
                params_0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.params = {
                    'mu': result.x[0],
                    'sigma': result.x[1],
                    'lambda': result.x[2],
                    'mu_j': result.x[3],
                    'sigma_j': result.x[4]
                }
                self.jump_parameters = self.params.copy()
            else:
                raise ValueError("Parameter estimation failed")
                
        except Exception as e:
            logger.error(f"Error in parameter estimation: {e}")
            # Use simple estimates as fallback
            self.params = {
                'mu': self.returns.mean(),
                'sigma': self.returns.std(),
                'lambda': 0.05,
                'mu_j': 0.0,
                'sigma_j': self.returns.std()
            }
            self.jump_parameters = self.params.copy()
    
    def _negative_log_likelihood(self, params):
        """Calculate negative log-likelihood for parameter estimation."""
        try:
            mu, sigma, lambda_j, mu_j, sigma_j = params
            
            # Ensure positive parameters
            if sigma <= 0 or lambda_j <= 0 or sigma_j <= 0:
                return np.inf
            
            log_likelihood = 0
            
            for return_val in self.returns:
                # Calculate likelihood for each observation
                # This is a simplified version - in practice you'd use more sophisticated methods
                
                # Normal component
                normal_likelihood = stats.norm.pdf(return_val, mu, sigma)
                
                # Jump component (simplified)
                jump_likelihood = lambda_j * stats.norm.pdf(return_val, mu + mu_j, np.sqrt(sigma**2 + sigma_j**2))
                
                # Total likelihood
                total_likelihood = (1 - lambda_j) * normal_likelihood + jump_likelihood
                
                if total_likelihood > 0:
                    log_likelihood += np.log(total_likelihood)
                else:
                    return np.inf
            
            return -log_likelihood
            
        except Exception as e:
            return np.inf
    
    def _detect_jumps(self):
        """Detect jumps in the time series."""
        try:
            # Simple jump detection using threshold method
            threshold = 3 * self.params['sigma']  # 3-sigma threshold
            
            # Find potential jumps
            jump_mask = np.abs(self.returns - self.params['mu']) > threshold
            
            self.jump_times = np.where(jump_mask)[0]
            self.jump_sizes = self.returns[jump_mask].values
            
            # Refine jump detection using likelihood ratio test
            self._refine_jump_detection()
            
        except Exception as e:
            logger.error(f"Error in jump detection: {e}")
            self.jump_times = np.array([])
            self.jump_sizes = np.array([])
    
    def _refine_jump_detection(self):
        """Refine jump detection using likelihood ratio test."""
        try:
            refined_jumps = []
            refined_sizes = []
            
            for i, (time_idx, jump_size) in enumerate(zip(self.jump_times, self.jump_sizes)):
                # Calculate likelihood ratio
                lr = self._calculate_likelihood_ratio(time_idx, jump_size)
                
                # If likelihood ratio is significant, keep the jump
                if lr > 2.0:  # Threshold for significance
                    refined_jumps.append(time_idx)
                    refined_sizes.append(jump_size)
            
            self.jump_times = np.array(refined_jumps)
            self.jump_sizes = np.array(refined_sizes)
            
        except Exception as e:
            logger.error(f"Error in jump refinement: {e}")
    
    def _calculate_likelihood_ratio(self, time_idx, jump_size):
        """Calculate likelihood ratio for jump detection."""
        try:
            # Likelihood under null hypothesis (no jump)
            l0 = stats.norm.pdf(jump_size, self.params['mu'], self.params['sigma'])
            
            # Likelihood under alternative hypothesis (jump)
            l1 = stats.norm.pdf(jump_size, self.params['mu'] + self.params['mu_j'], 
                               np.sqrt(self.params['sigma']**2 + self.params['sigma_j']**2))
            
            if l0 > 0 and l1 > 0:
                return np.log(l1 / l0)
            else:
                return 0
                
        except Exception as e:
            return 0
    
    def _calculate_model_statistics(self):
        """Calculate model statistics."""
        try:
            # Calculate log-likelihood
            self.log_likelihood = -self._negative_log_likelihood([
                self.params['mu'], self.params['sigma'], self.params['lambda'],
                self.params['mu_j'], self.params['sigma_j']
            ])
            
            # Calculate AIC and BIC
            n_params = 5  # mu, sigma, lambda, mu_j, sigma_j
            n_obs = len(self.returns)
            
            self.aic = 2 * n_params - 2 * self.log_likelihood
            self.bic = np.log(n_obs) * n_params - 2 * self.log_likelihood
            
            # Jump statistics
            self.n_jumps = len(self.jump_times)
            self.jump_frequency = self.n_jumps / len(self.returns)
            
            if self.n_jumps > 0:
                self.mean_jump_size = np.mean(self.jump_sizes)
                self.jump_volatility = np.std(self.jump_sizes)
            else:
                self.mean_jump_size = 0
                self.jump_volatility = 0
                
        except Exception as e:
            logger.error(f"Error calculating model statistics: {e}")
    
    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Generate forecasts for the specified horizon.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Array of forecasts
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            forecasts = []
            current_value = self.data.iloc[-1] if self.data.iloc[-1] > 0 else 100  # Use last value or default
            
            for h in range(horizon):
                # Simulate one step ahead
                # Drift component
                drift = self.params['mu']
                
                # Diffusion component
                diffusion = self.params['sigma'] * np.random.normal(0, 1)
                
                # Jump component
                jump_prob = np.random.random()
                if jump_prob < self.params['lambda']:
                    jump = np.random.normal(self.params['mu_j'], self.params['sigma_j'])
                else:
                    jump = 0
                
                # Total return
                total_return = drift + diffusion + jump
                
                # Update value
                current_value = current_value * (1 + total_return)
                forecasts.append(current_value)
            
            return np.array(forecasts)
            
        except Exception as e:
            raise Exception(f"Error generating forecasts: {e}")
    
    def get_confidence_intervals(self, horizon: int, confidence_level: float = 0.95) -> List[Dict[str, float]]:
        """
        Get confidence intervals for forecasts.
        
        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            List of confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        try:
            forecasts = self.forecast(horizon)
            confidence_intervals = []
            
            # Calculate total volatility including jumps
            total_volatility = np.sqrt(self.params['sigma']**2 + self.params['lambda'] * 
                                     (self.params['mu_j']**2 + self.params['sigma_j']**2))
            
            for i, forecast in enumerate(forecasts):
                # Confidence interval based on total volatility
                margin = stats.norm.ppf((1 + confidence_level) / 2) * total_volatility * np.sqrt(i + 1)
                
                confidence_intervals.append({
                    'lower': forecast - margin,
                    'upper': forecast + margin
                })
            
            return confidence_intervals
            
        except Exception as e:
            raise Exception(f"Error calculating confidence intervals: {e}")
    
    def get_jump_times(self) -> np.ndarray:
        """Get detected jump times."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting jump times")
        return self.jump_times
    
    def get_jump_sizes(self) -> np.ndarray:
        """Get detected jump sizes."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting jump sizes")
        return self.jump_sizes
    
    def get_jump_parameters(self) -> Dict[str, float]:
        """Get jump model parameters."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting jump parameters")
        return self.jump_parameters
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        if not self.fitted:
            return {"error": "Model not fitted"}
        
        return {
            "n_observations": len(self.returns),
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "parameters": self.params,
            "jump_statistics": {
                "n_jumps": self.n_jumps,
                "jump_frequency": self.jump_frequency,
                "mean_jump_size": self.mean_jump_size,
                "jump_volatility": self.jump_volatility
            },
            "jump_times": self.jump_times.tolist(),
            "jump_sizes": self.jump_sizes.tolist()
        }
    
    def simulate_path(self, n_steps: int = 252, initial_value: float = 100) -> np.ndarray:
        """
        Simulate a path using the fitted model.
        
        Args:
            n_steps: Number of simulation steps
            initial_value: Initial value for simulation
            
        Returns:
            Simulated path
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before simulation")
        
        try:
            path = [initial_value]
            current_value = initial_value
            
            for step in range(n_steps):
                # Drift component
                drift = self.params['mu']
                
                # Diffusion component
                diffusion = self.params['sigma'] * np.random.normal(0, 1)
                
                # Jump component
                jump_prob = np.random.random()
                if jump_prob < self.params['lambda']:
                    jump = np.random.normal(self.params['mu_j'], self.params['sigma_j'])
                else:
                    jump = 0
                
                # Total return
                total_return = drift + diffusion + jump
                
                # Update value
                current_value = current_value * (1 + total_return)
                path.append(current_value)
            
            return np.array(path)
            
        except Exception as e:
            raise Exception(f"Error in path simulation: {e}")
