"""
Model diagnostics for econometric models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ModelDiagnostics:
    """Model diagnostics for evaluating econometric model performance."""
    
    def __init__(self):
        """Initialize diagnostics."""
        pass
    
    def run_diagnostics(self, model: Any, data: pd.Series) -> Dict[str, Any]:
        """
        Run comprehensive diagnostics on a fitted model.
        
        Args:
            model: Fitted econometric model
            data: Original data series
            
        Returns:
            Dictionary containing diagnostic results
        """
        try:
            diagnostics = {}
            
            # Get residuals if available
            residuals = self._get_residuals(model, data)
            if residuals is not None:
                diagnostics['residual_analysis'] = self._analyze_residuals(residuals)
            
            # Model fit statistics
            diagnostics['fit_statistics'] = self._calculate_fit_statistics(model, data)
            
            # Normality tests
            if residuals is not None:
                diagnostics['normality_tests'] = self._test_normality(residuals)
            
            # Autocorrelation tests
            if residuals is not None:
                diagnostics['autocorrelation_tests'] = self._test_autocorrelation(residuals)
            
            # Heteroscedasticity tests
            if residuals is not None:
                diagnostics['heteroscedasticity_tests'] = self._test_heteroscedasticity(residuals)
            
            # Model-specific diagnostics
            diagnostics['model_specific'] = self._model_specific_diagnostics(model, data)
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            return {'error': str(e)}
    
    def _get_residuals(self, model: Any, data: pd.Series) -> Optional[pd.Series]:
        """Extract residuals from model."""
        try:
            if hasattr(model, 'resid'):
                return pd.Series(model.resid)
            elif hasattr(model, 'residuals'):
                return pd.Series(model.residuals)
            elif hasattr(model, 'get_residuals'):
                return pd.Series(model.get_residuals())
            else:
                # Try to calculate residuals manually
                if hasattr(model, 'predict'):
                    predictions = model.predict(data)
                    residuals = data - predictions
                    return pd.Series(residuals)
                else:
                    return None
        except Exception as e:
            logger.warning(f"Could not extract residuals: {e}")
            return None
    
    def _analyze_residuals(self, residuals: pd.Series) -> Dict[str, Any]:
        """Analyze residual properties."""
        try:
            # Remove NaN values
            residuals_clean = residuals.dropna()
            
            if len(residuals_clean) == 0:
                return {'error': 'No valid residuals available'}
            
            analysis = {
                'mean': float(residuals_clean.mean()),
                'std': float(residuals_clean.std()),
                'skewness': float(stats.skew(residuals_clean)),
                'kurtosis': float(stats.kurtosis(residuals_clean)),
                'min': float(residuals_clean.min()),
                'max': float(residuals_clean.max()),
                'count': len(residuals_clean)
            }
            
            # Check for zero mean (approximately)
            analysis['zero_mean_test'] = {
                'p_value': float(stats.ttest_1samp(residuals_clean, 0)[1]),
                'significant': analysis['mean'] != 0
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing residuals: {e}")
            return {'error': str(e)}
    
    def _calculate_fit_statistics(self, model: Any, data: pd.Series) -> Dict[str, Any]:
        """Calculate model fit statistics."""
        try:
            stats_dict = {}
            
            # AIC and BIC
            if hasattr(model, 'aic'):
                stats_dict['aic'] = float(model.aic)
            if hasattr(model, 'bic'):
                stats_dict['bic'] = float(model.bic)
            
            # Log likelihood
            if hasattr(model, 'log_likelihood'):
                stats_dict['log_likelihood'] = float(model.log_likelihood)
            elif hasattr(model, 'llf'):
                stats_dict['log_likelihood'] = float(model.llf)
            
            # R-squared (if applicable)
            if hasattr(model, 'rsquared'):
                stats_dict['r_squared'] = float(model.rsquared)
            elif hasattr(model, 'r2'):
                stats_dict['r_squared'] = float(model.r2)
            
            # Adjusted R-squared
            if hasattr(model, 'rsquared_adj'):
                stats_dict['r_squared_adj'] = float(model.rsquared_adj)
            
            # Number of observations
            stats_dict['n_observations'] = len(data)
            
            # Number of parameters
            if hasattr(model, 'params'):
                stats_dict['n_parameters'] = len(model.params)
            elif hasattr(model, 'n_params'):
                stats_dict['n_parameters'] = model.n_params
            
            return stats_dict
            
        except Exception as e:
            logger.error(f"Error calculating fit statistics: {e}")
            return {'error': str(e)}
    
    def _test_normality(self, residuals: pd.Series) -> Dict[str, Any]:
        """Test for normality of residuals."""
        try:
            residuals_clean = residuals.dropna()
            
            if len(residuals_clean) < 3:
                return {'error': 'Insufficient data for normality tests'}
            
            tests = {}
            
            # Shapiro-Wilk test
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals_clean)
                tests['shapiro_wilk'] = {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                }
            except Exception as e:
                tests['shapiro_wilk'] = {'error': str(e)}
            
            # Anderson-Darling test
            try:
                anderson_result = stats.anderson(residuals_clean)
                tests['anderson_darling'] = {
                    'statistic': float(anderson_result.statistic),
                    'critical_values': anderson_result.critical_values.tolist(),
                    'significance_levels': anderson_result.significance_level.tolist()
                }
            except Exception as e:
                tests['anderson_darling'] = {'error': str(e)}
            
            # Jarque-Bera test
            try:
                jb_stat, jb_p = stats.jarque_bera(residuals_clean)
                tests['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_p),
                    'is_normal': jb_p > 0.05
                }
            except Exception as e:
                tests['jarque_bera'] = {'error': str(e)}
            
            return tests
            
        except Exception as e:
            logger.error(f"Error testing normality: {e}")
            return {'error': str(e)}
    
    def _test_autocorrelation(self, residuals: pd.Series) -> Dict[str, Any]:
        """Test for autocorrelation in residuals."""
        try:
            residuals_clean = residuals.dropna()
            
            if len(residuals_clean) < 4:
                return {'error': 'Insufficient data for autocorrelation tests'}
            
            tests = {}
            
            # Durbin-Watson test
            try:
                dw_stat = self._durbin_watson(residuals_clean)
                tests['durbin_watson'] = {
                    'statistic': float(dw_stat),
                    'interpretation': self._interpret_dw(dw_stat)
                }
            except Exception as e:
                tests['durbin_watson'] = {'error': str(e)}
            
            # Ljung-Box test
            try:
                lb_stat, lb_p = stats.acf(residuals_clean, nlags=min(10, len(residuals_clean)//4), fft=False)
                # Calculate Ljung-Box manually
                n = len(residuals_clean)
                k = min(10, n//4)
                lb_stat = n * (n + 2) * sum([(lb_stat[i]**2) / (n - i - 1) for i in range(1, k+1)])
                lb_p = 1 - stats.chi2.cdf(lb_stat, k)
                
                tests['ljung_box'] = {
                    'statistic': float(lb_stat),
                    'p_value': float(lb_p),
                    'no_autocorrelation': lb_p > 0.05
                }
            except Exception as e:
                tests['ljung_box'] = {'error': str(e)}
            
            return tests
            
        except Exception as e:
            logger.error(f"Error testing autocorrelation: {e}")
            return {'error': str(e)}
    
    def _test_heteroscedasticity(self, residuals: pd.Series) -> Dict[str, Any]:
        """Test for heteroscedasticity in residuals."""
        try:
            residuals_clean = residuals.dropna()
            
            if len(residuals_clean) < 10:
                return {'error': 'Insufficient data for heteroscedasticity tests'}
            
            tests = {}
            
            # Breusch-Pagan test (simplified)
            try:
                # Use squared residuals vs fitted values
                bp_stat = self._breusch_pagan_test(residuals_clean)
                tests['breusch_pagan'] = {
                    'statistic': float(bp_stat),
                    'interpretation': 'Heteroscedasticity present' if bp_stat > 3.84 else 'No heteroscedasticity'
                }
            except Exception as e:
                tests['breusch_pagan'] = {'error': str(e)}
            
            # White test (simplified)
            try:
                white_stat = self._white_test(residuals_clean)
                tests['white'] = {
                    'statistic': float(white_stat),
                    'interpretation': 'Heteroscedasticity present' if white_stat > 5.99 else 'No heteroscedasticity'
                }
            except Exception as e:
                tests['white'] = {'error': str(e)}
            
            return tests
            
        except Exception as e:
            logger.error(f"Error testing heteroscedasticity: {e}")
            return {'error': str(e)}
    
    def _model_specific_diagnostics(self, model: Any, data: pd.Series) -> Dict[str, Any]:
        """Model-specific diagnostics."""
        try:
            diagnostics = {}
            
            # GARCH-specific diagnostics
            if hasattr(model, 'conditional_volatility'):
                diagnostics['garch'] = {
                    'volatility_persistence': self._check_volatility_persistence(model),
                    'leverage_effects': self._check_leverage_effects(model)
                }
            
            # Regime switching diagnostics
            if hasattr(model, 'regime_probabilities'):
                diagnostics['regime_switching'] = {
                    'regime_stability': self._check_regime_stability(model),
                    'transition_probabilities': self._get_transition_probabilities(model)
                }
            
            # Jump diffusion diagnostics
            if hasattr(model, 'jump_parameters'):
                diagnostics['jump_diffusion'] = {
                    'jump_intensity': self._get_jump_intensity(model),
                    'jump_size_distribution': self._analyze_jump_sizes(model)
                }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error in model-specific diagnostics: {e}")
            return {'error': str(e)}
    
    def _durbin_watson(self, residuals: pd.Series) -> float:
        """Calculate Durbin-Watson statistic."""
        diff = residuals.diff().dropna()
        return (diff ** 2).sum() / (residuals ** 2).sum()
    
    def _interpret_dw(self, dw_stat: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No autocorrelation"
    
    def _breusch_pagan_test(self, residuals: pd.Series) -> float:
        """Simplified Breusch-Pagan test."""
        # Use squared residuals as dependent variable
        squared_resid = residuals ** 2
        # Simple regression of squared residuals on index
        x = np.arange(len(squared_resid))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, squared_resid)
        return r_value ** 2 * len(residuals)
    
    def _white_test(self, residuals: pd.Series) -> float:
        """Simplified White test."""
        # Use squared residuals as dependent variable
        squared_resid = residuals ** 2
        # Simple regression of squared residuals on index and squared index
        x = np.arange(len(squared_resid))
        x_squared = x ** 2
        # Multiple regression
        X = np.column_stack([x, x_squared])
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, squared_resid)
        return r_value ** 2 * len(residuals)
    
    def _check_volatility_persistence(self, model: Any) -> Dict[str, Any]:
        """Check volatility persistence in GARCH models."""
        try:
            if hasattr(model, 'params'):
                # Sum of GARCH parameters
                garch_params = [p for p in model.params if 'alpha' in str(p) or 'beta' in str(p)]
                persistence = sum(garch_params)
                return {
                    'persistence': float(persistence),
                    'stationary': persistence < 1.0
                }
            return {'error': 'No GARCH parameters found'}
        except Exception as e:
            return {'error': str(e)}
    
    def _check_leverage_effects(self, model: Any) -> Dict[str, Any]:
        """Check for leverage effects in GARCH models."""
        try:
            if hasattr(model, 'params'):
                # Look for gamma parameter (leverage effect)
                leverage_params = [p for p in model.params if 'gamma' in str(p)]
                if leverage_params:
                    return {
                        'leverage_effect': float(leverage_params[0]),
                        'significant': abs(leverage_params[0]) > 0.1
                    }
            return {'leverage_effect': 'Not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _check_regime_stability(self, model: Any) -> Dict[str, Any]:
        """Check regime stability in regime switching models."""
        try:
            if hasattr(model, 'regime_probabilities'):
                probs = model.regime_probabilities
                # Calculate regime stability
                stability = np.mean(np.max(probs, axis=1))
                return {
                    'stability': float(stability),
                    'interpretation': 'Stable' if stability > 0.8 else 'Unstable'
                }
            return {'error': 'No regime probabilities found'}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_transition_probabilities(self, model: Any) -> Dict[str, Any]:
        """Get transition probabilities for regime switching models."""
        try:
            if hasattr(model, 'transition_probabilities'):
                return {
                    'probabilities': model.transition_probabilities.tolist()
                }
            return {'error': 'No transition probabilities found'}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_jump_intensity(self, model: Any) -> Dict[str, Any]:
        """Get jump intensity for jump diffusion models."""
        try:
            if hasattr(model, 'jump_parameters'):
                params = model.jump_parameters
                if 'lambda' in params:
                    return {
                        'intensity': float(params['lambda']),
                        'jumps_per_year': float(params['lambda'] * 252)  # Assuming daily data
                    }
            return {'error': 'No jump parameters found'}
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_jump_sizes(self, model: Any) -> Dict[str, Any]:
        """Analyze jump size distribution."""
        try:
            if hasattr(model, 'jump_parameters'):
                params = model.jump_parameters
                if 'mu_j' in params and 'sigma_j' in params:
                    return {
                        'mean_jump_size': float(params['mu_j']),
                        'jump_volatility': float(params['sigma_j']),
                        'expected_jump_size': float(params['mu_j'] + 0.5 * params['sigma_j']**2)
                    }
            return {'error': 'No jump size parameters found'}
        except Exception as e:
            return {'error': str(e)}
