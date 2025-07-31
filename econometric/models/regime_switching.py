"""
Regime Switching Model for Housing Market Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats
from scipy.optimize import minimize
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class RegimeSwitchingModel:
    """Markov Regime Switching model for housing market analysis."""
    
    def __init__(self, n_regimes: int = 2):
        """
        Initialize the regime switching model.
        
        Args:
            n_regimes: Number of regimes (default: 2)
        """
        self.n_regimes = n_regimes
        self.fitted = False
        self.data = None
        self.params = {}
        self.regime_probabilities = None
        self.transition_probabilities = None
        self.regime_means = None
        self.regime_volatilities = None
        
    def fit(self, data: pd.Series):
        """
        Fit the regime switching model to the data.
        
        Args:
            data: Time series data
        """
        try:
            # Clean data
            self.data = data.dropna()
            
            if len(self.data) < 50:
                raise ValueError("Insufficient data for regime switching estimation")
            
            # Initialize parameters
            self._initialize_parameters()
            
            # Fit model using EM algorithm
            self._fit_em_algorithm()
            
            self.fitted = True
            
            # Calculate model statistics
            self._calculate_model_statistics()
            
            return self
            
        except Exception as e:
            raise Exception(f"Error fitting regime switching model: {e}")
    
    def _initialize_parameters(self):
        """Initialize model parameters."""
        # Initialize regime means using quantiles
        quantiles = np.linspace(0.1, 0.9, self.n_regimes)
        self.regime_means = np.percentile(self.data, quantiles * 100)
        
        # Initialize regime volatilities
        self.regime_volatilities = np.array([self.data.std()] * self.n_regimes)
        
        # Initialize transition probabilities (diagonal dominant)
        self.transition_probabilities = np.eye(self.n_regimes) * 0.8 + np.random.uniform(0, 0.1, (self.n_regimes, self.n_regimes))
        self.transition_probabilities = self.transition_probabilities / self.transition_probabilities.sum(axis=1, keepdims=True)
        
        # Initialize regime probabilities
        self.regime_probabilities = np.ones(self.n_regimes) / self.n_regimes
    
    def _fit_em_algorithm(self, max_iterations: int = 100, tolerance: float = 1e-6):
        """Fit model using Expectation-Maximization algorithm."""
        log_likelihood_old = -np.inf
        
        for iteration in range(max_iterations):
            # E-step: Calculate regime probabilities
            self._expectation_step()
            
            # M-step: Update parameters
            self._maximization_step()
            
            # Calculate log-likelihood
            log_likelihood_new = self._calculate_log_likelihood()
            
            # Check convergence
            if abs(log_likelihood_new - log_likelihood_old) < tolerance:
                break
                
            log_likelihood_old = log_likelihood_new
        
        self.log_likelihood = log_likelihood_new
    
    def _expectation_step(self):
        """E-step: Calculate regime probabilities."""
        n_obs = len(self.data)
        
        # Forward pass
        alpha = np.zeros((n_obs, self.n_regimes))
        alpha[0] = self.regime_probabilities * self._emission_probabilities(0)
        alpha[0] = alpha[0] / alpha[0].sum()
        
        for t in range(1, n_obs):
            for j in range(self.n_regimes):
                alpha[t, j] = self._emission_probabilities(t)[j] * np.sum(alpha[t-1] * self.transition_probabilities[:, j])
            alpha[t] = alpha[t] / alpha[t].sum()
        
        # Backward pass
        beta = np.zeros((n_obs, self.n_regimes))
        beta[-1] = np.ones(self.n_regimes)
        
        for t in range(n_obs-2, -1, -1):
            for i in range(self.n_regimes):
                beta[t, i] = np.sum(self.transition_probabilities[i, :] * self._emission_probabilities(t+1) * beta[t+1])
            beta[t] = beta[t] / beta[t].sum()
        
        # Calculate regime probabilities
        self.regime_probabilities = alpha * beta
        self.regime_probabilities = self.regime_probabilities / self.regime_probabilities.sum(axis=1, keepdims=True)
    
    def _maximization_step(self):
        """M-step: Update model parameters."""
        n_obs = len(self.data)
        
        # Update regime means
        for k in range(self.n_regimes):
            self.regime_means[k] = np.sum(self.regime_probabilities[:, k] * self.data) / np.sum(self.regime_probabilities[:, k])
        
        # Update regime volatilities
        for k in range(self.n_regimes):
            mean_squared_diff = np.sum(self.regime_probabilities[:, k] * (self.data - self.regime_means[k]) ** 2)
            self.regime_volatilities[k] = np.sqrt(mean_squared_diff / np.sum(self.regime_probabilities[:, k]))
        
        # Update transition probabilities
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = 0
                denominator = 0
                
                for t in range(n_obs - 1):
                    if t < n_obs - 1:
                        numerator += self.regime_probabilities[t, i] * self.transition_probabilities[i, j] * self._emission_probabilities(t+1)[j]
                    denominator += self.regime_probabilities[t, i]
                
                if denominator > 0:
                    self.transition_probabilities[i, j] = numerator / denominator
        
        # Normalize transition probabilities
        self.transition_probabilities = self.transition_probabilities / self.transition_probabilities.sum(axis=1, keepdims=True)
    
    def _emission_probabilities(self, t: int) -> np.ndarray:
        """Calculate emission probabilities for observation at time t."""
        probabilities = np.zeros(self.n_regimes)
        
        for k in range(self.n_regimes):
            # Normal distribution probability
            z_score = (self.data.iloc[t] - self.regime_means[k]) / self.regime_volatilities[k]
            probabilities[k] = stats.norm.pdf(z_score) / self.regime_volatilities[k]
        
        return probabilities
    
    def _calculate_log_likelihood(self) -> float:
        """Calculate log-likelihood of the model."""
        n_obs = len(self.data)
        log_likelihood = 0
        
        for t in range(n_obs):
            emission_probs = self._emission_probabilities(t)
            log_likelihood += np.log(np.sum(self.regime_probabilities[t] * emission_probs))
        
        return log_likelihood
    
    def _calculate_model_statistics(self):
        """Calculate model statistics."""
        # AIC and BIC
        n_params = self.n_regimes * 2 + self.n_regimes * (self.n_regimes - 1)  # means, volatilities, transition probs
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = np.log(len(self.data)) * n_params - 2 * self.log_likelihood
        
        # Regime classification
        self.regime_classification = np.argmax(self.regime_probabilities, axis=1)
        
        # Regime durations
        self.regime_durations = self._calculate_regime_durations()
    
    def _calculate_regime_durations(self) -> Dict[int, float]:
        """Calculate average duration for each regime."""
        durations = {}
        
        for regime in range(self.n_regimes):
            regime_mask = self.regime_classification == regime
            if regime_mask.sum() > 0:
                # Calculate average duration
                duration_sequences = []
                current_duration = 0
                
                for is_regime in regime_mask:
                    if is_regime:
                        current_duration += 1
                    else:
                        if current_duration > 0:
                            duration_sequences.append(current_duration)
                            current_duration = 0
                
                if current_duration > 0:
                    duration_sequences.append(current_duration)
                
                durations[regime] = np.mean(duration_sequences) if duration_sequences else 0
            else:
                durations[regime] = 0
        
        return durations
    
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
            current_regime_probs = self.regime_probabilities[-1]
            
            for h in range(horizon):
                # Predict regime probabilities
                regime_probs = current_regime_probs @ self.transition_probabilities
                
                # Calculate forecast as weighted average of regime means
                forecast = np.sum(regime_probs * self.regime_means)
                forecasts.append(forecast)
                
                # Update regime probabilities for next step
                current_regime_probs = regime_probs
            
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
            
            # Calculate overall volatility
            overall_volatility = np.sqrt(np.sum(self.regime_probabilities[-1] * self.regime_volatilities ** 2))
            
            for i, forecast in enumerate(forecasts):
                # Simple confidence interval based on regime-weighted volatility
                margin = stats.norm.ppf((1 + confidence_level) / 2) * overall_volatility * np.sqrt(i + 1)
                
                confidence_intervals.append({
                    'lower': forecast - margin,
                    'upper': forecast + margin
                })
            
            return confidence_intervals
            
        except Exception as e:
            raise Exception(f"Error calculating confidence intervals: {e}")
    
    def get_regime_probabilities(self) -> np.ndarray:
        """Get regime probabilities for each observation."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting regime probabilities")
        return self.regime_probabilities
    
    def get_transition_probabilities(self) -> np.ndarray:
        """Get transition probability matrix."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting transition probabilities")
        return self.transition_probabilities
    
    def get_regime_means(self) -> np.ndarray:
        """Get regime means."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting regime means")
        return self.regime_means
    
    def get_regime_volatilities(self) -> np.ndarray:
        """Get regime volatilities."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting regime volatilities")
        return self.regime_volatilities
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        if not self.fitted:
            return {"error": "Model not fitted"}
        
        return {
            "n_regimes": self.n_regimes,
            "n_observations": len(self.data),
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "regime_means": self.regime_means.tolist(),
            "regime_volatilities": self.regime_volatilities.tolist(),
            "transition_probabilities": self.transition_probabilities.tolist(),
            "regime_durations": self.regime_durations,
            "regime_classification": self.regime_classification.tolist()
        }
