"""
Risk metrics calculator for financial time series analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """Calculator for various risk metrics used in financial analysis."""
    
    def __init__(self):
        """Initialize risk metrics calculator."""
        pass
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for a return series.
        
        Args:
            returns: Time series of returns
            
        Returns:
            Dictionary containing various risk metrics
        """
        try:
            # Clean data
            returns_clean = returns.dropna()
            
            if len(returns_clean) < 10:
                return {'error': 'Insufficient data for risk calculations'}
            
            metrics = {}
            
            # Basic statistics
            metrics['basic_statistics'] = self._calculate_basic_statistics(returns_clean)
            
            # Volatility metrics
            metrics['volatility_metrics'] = self._calculate_volatility_metrics(returns_clean)
            
            # Risk measures
            metrics['risk_measures'] = self._calculate_risk_measures(returns_clean)
            
            # Distribution metrics
            metrics['distribution_metrics'] = self._calculate_distribution_metrics(returns_clean)
            
            # Tail risk metrics
            metrics['tail_risk_metrics'] = self._calculate_tail_risk_metrics(returns_clean)
            
            # Drawdown metrics
            metrics['drawdown_metrics'] = self._calculate_drawdown_metrics(returns_clean)
            
            # Correlation metrics
            metrics['correlation_metrics'] = self._calculate_correlation_metrics(returns_clean)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_basic_statistics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic statistical measures."""
        try:
            return {
                'mean': float(returns.mean()),
                'median': float(returns.median()),
                'std': float(returns.std()),
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'min': float(returns.min()),
                'max': float(returns.max()),
                'range': float(returns.max() - returns.min()),
                'count': len(returns)
            }
        except Exception as e:
            logger.error(f"Error calculating basic statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate volatility-related metrics."""
        try:
            # Annualized volatility (assuming daily data)
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # Rolling volatility
            rolling_vol_30 = returns.rolling(window=30).std() * np.sqrt(252)
            rolling_vol_60 = returns.rolling(window=60).std() * np.sqrt(252)
            
            # Volatility of volatility
            vol_of_vol = rolling_vol_30.std()
            
            # Parkinson volatility (high-low estimator)
            parkinson_vol = self._parkinson_volatility(returns)
            
            return {
                'daily_volatility': float(daily_vol),
                'annualized_volatility': float(annual_vol),
                'rolling_volatility_30d': {
                    'mean': float(rolling_vol_30.mean()),
                    'std': float(rolling_vol_30.std()),
                    'min': float(rolling_vol_30.min()),
                    'max': float(rolling_vol_30.max())
                },
                'rolling_volatility_60d': {
                    'mean': float(rolling_vol_60.mean()),
                    'std': float(rolling_vol_60.std()),
                    'min': float(rolling_vol_60.min()),
                    'max': float(rolling_vol_60.max())
                },
                'volatility_of_volatility': float(vol_of_vol),
                'parkinson_volatility': float(parkinson_vol) if parkinson_vol else None
            }
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_risk_measures(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate various risk measures."""
        try:
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = returns[returns <= var_95].mean()
            es_99 = returns[returns <= var_99].mean()
            
            # Semi-deviation
            negative_returns = returns[returns < 0]
            semi_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
            
            # Downside deviation
            downside_deviation = np.sqrt(np.mean(np.minimum(returns - returns.mean(), 0) ** 2))
            
            # Sortino ratio
            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
            sortino_ratio = (returns.mean() - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Sharpe ratio
            sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std() if returns.std() > 0 else 0
            
            # Calmar ratio (annualized return / max drawdown)
            annual_return = returns.mean() * 252
            max_drawdown = self._calculate_max_drawdown(returns)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            return {
                'value_at_risk': {
                    'var_95': float(var_95),
                    'var_99': float(var_99)
                },
                'expected_shortfall': {
                    'es_95': float(es_95),
                    'es_99': float(es_99)
                },
                'semi_deviation': float(semi_deviation),
                'downside_deviation': float(downside_deviation),
                'sortino_ratio': float(sortino_ratio),
                'sharpe_ratio': float(sharpe_ratio),
                'calmar_ratio': float(calmar_ratio)
            }
        except Exception as e:
            logger.error(f"Error calculating risk measures: {e}")
            return {'error': str(e)}
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate distribution-related metrics."""
        try:
            # Jarque-Bera test for normality
            jb_stat, jb_p = stats.jarque_bera(returns)
            
            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(returns)
            
            # Kolmogorov-Smirnov test against normal distribution
            normal_sample = np.random.normal(returns.mean(), returns.std(), len(returns))
            ks_stat, ks_p = stats.ks_2samp(returns, normal_sample)
            
            # Quantile-based measures
            q_01 = np.percentile(returns, 1)
            q_05 = np.percentile(returns, 5)
            q_10 = np.percentile(returns, 10)
            q_25 = np.percentile(returns, 25)
            q_50 = np.percentile(returns, 50)
            q_75 = np.percentile(returns, 75)
            q_90 = np.percentile(returns, 90)
            q_95 = np.percentile(returns, 95)
            q_99 = np.percentile(returns, 99)
            
            return {
                'normality_tests': {
                    'jarque_bera': {
                        'statistic': float(jb_stat),
                        'p_value': float(jb_p),
                        'is_normal': jb_p > 0.05
                    },
                    'anderson_darling': {
                        'statistic': float(ad_stat),
                        'critical_values': ad_critical.tolist(),
                        'significance_levels': ad_significance.tolist()
                    },
                    'kolmogorov_smirnov': {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p)
                    }
                },
                'quantiles': {
                    'q_01': float(q_01),
                    'q_05': float(q_05),
                    'q_10': float(q_10),
                    'q_25': float(q_25),
                    'q_50': float(q_50),
                    'q_75': float(q_75),
                    'q_90': float(q_90),
                    'q_95': float(q_95),
                    'q_99': float(q_99)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating distribution metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_tail_risk_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate tail risk metrics."""
        try:
            # Tail dependence
            left_tail = returns[returns <= np.percentile(returns, 5)]
            right_tail = returns[returns >= np.percentile(returns, 95)]
            
            # Tail ratios
            left_tail_ratio = len(left_tail) / len(returns)
            right_tail_ratio = len(right_tail) / len(returns)
            
            # Tail means
            left_tail_mean = left_tail.mean() if len(left_tail) > 0 else 0
            right_tail_mean = right_tail.mean() if len(right_tail) > 0 else 0
            
            # Tail standard deviations
            left_tail_std = left_tail.std() if len(left_tail) > 1 else 0
            right_tail_std = right_tail.std() if len(right_tail) > 1 else 0
            
            # Extreme value theory (simplified)
            evt_metrics = self._extreme_value_theory(returns)
            
            return {
                'tail_ratios': {
                    'left_tail_ratio': float(left_tail_ratio),
                    'right_tail_ratio': float(right_tail_ratio)
                },
                'tail_means': {
                    'left_tail_mean': float(left_tail_mean),
                    'right_tail_mean': float(right_tail_mean)
                },
                'tail_standard_deviations': {
                    'left_tail_std': float(left_tail_std),
                    'right_tail_std': float(right_tail_std)
                },
                'extreme_value_theory': evt_metrics
            }
        except Exception as e:
            logger.error(f"Error calculating tail risk metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate drawdown-related metrics."""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative_returns.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            
            # Drawdown duration
            max_dd_idx = drawdown.idxmin()
            peak_idx = running_max.loc[:max_dd_idx].idxmax()
            drawdown_duration = (max_dd_idx - peak_idx).days if hasattr(max_dd_idx, 'days') else len(returns.loc[peak_idx:max_dd_idx])
            
            # Average drawdown
            avg_drawdown = drawdown[drawdown < 0].mean()
            
            # Drawdown frequency
            drawdown_frequency = len(drawdown[drawdown < 0]) / len(drawdown)
            
            # Recovery time (simplified)
            recovery_time = self._calculate_recovery_time(cumulative_returns, max_dd_idx)
            
            return {
                'maximum_drawdown': float(max_drawdown),
                'drawdown_duration': int(drawdown_duration),
                'average_drawdown': float(avg_drawdown),
                'drawdown_frequency': float(drawdown_frequency),
                'recovery_time': int(recovery_time) if recovery_time else None
            }
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_correlation_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate correlation-related metrics."""
        try:
            # Autocorrelation
            autocorr_1 = returns.autocorr(lag=1)
            autocorr_5 = returns.autocorr(lag=5)
            autocorr_10 = returns.autocorr(lag=10)
            
            # Rolling correlation
            rolling_corr = returns.rolling(window=30).corr(returns.shift(1))
            
            # Correlation with lagged values
            lag_correlations = {}
            for lag in [1, 2, 3, 5, 10]:
                lag_correlations[f'lag_{lag}'] = float(returns.corr(returns.shift(lag)))
            
            # Volatility clustering (GARCH-like)
            volatility_clustering = self._calculate_volatility_clustering(returns)
            
            return {
                'autocorrelation': {
                    'lag_1': float(autocorr_1) if not pd.isna(autocorr_1) else None,
                    'lag_5': float(autocorr_5) if not pd.isna(autocorr_5) else None,
                    'lag_10': float(autocorr_10) if not pd.isna(autocorr_10) else None
                },
                'lag_correlations': lag_correlations,
                'rolling_correlation': {
                    'mean': float(rolling_corr.mean()),
                    'std': float(rolling_corr.std())
                },
                'volatility_clustering': volatility_clustering
            }
        except Exception as e:
            logger.error(f"Error calculating correlation metrics: {e}")
            return {'error': str(e)}
    
    def _parkinson_volatility(self, returns: pd.Series) -> Optional[float]:
        """Calculate Parkinson volatility estimator."""
        try:
            # This is a simplified version - in practice you'd need high-low data
            # For now, we'll use a proxy based on squared returns
            squared_returns = returns ** 2
            parkinson_vol = np.sqrt(np.mean(squared_returns) / (2 * np.log(2)))
            return float(parkinson_vol)
        except Exception as e:
            logger.warning(f"Could not calculate Parkinson volatility: {e}")
            return None
    
    def _extreme_value_theory(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate extreme value theory metrics."""
        try:
            # Fit generalized Pareto distribution to tail
            threshold = np.percentile(returns, 95)
            tail_data = returns[returns > threshold] - threshold
            
            if len(tail_data) < 10:
                return {'error': 'Insufficient tail data for EVT'}
            
            # Fit GPD parameters
            shape, loc, scale = stats.genpareto.fit(tail_data)
            
            # Calculate return levels
            return_level_10 = threshold + scale * ((0.1 ** (-shape) - 1) / shape) if shape != 0 else threshold + scale * np.log(10)
            return_level_100 = threshold + scale * ((0.01 ** (-shape) - 1) / shape) if shape != 0 else threshold + scale * np.log(100)
            
            return {
                'threshold': float(threshold),
                'shape_parameter': float(shape),
                'scale_parameter': float(scale),
                'return_levels': {
                    '10_year': float(return_level_10),
                    '100_year': float(return_level_100)
                }
            }
        except Exception as e:
            logger.error(f"Error in extreme value theory: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return float(drawdown.min())
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_recovery_time(self, cumulative_returns: pd.Series, max_dd_idx) -> Optional[int]:
        """Calculate recovery time from maximum drawdown."""
        try:
            max_dd_value = cumulative_returns.loc[max_dd_idx]
            peak_value = cumulative_returns.loc[:max_dd_idx].max()
            
            # Find when we recover to peak value
            recovery_mask = cumulative_returns.loc[max_dd_idx:] >= peak_value
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                recovery_time = (recovery_idx - max_dd_idx).days if hasattr(recovery_idx, 'days') else len(cumulative_returns.loc[max_dd_idx:recovery_idx])
                return recovery_time
            return None
        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return None
    
    def _calculate_volatility_clustering(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility clustering metrics."""
        try:
            # Squared returns autocorrelation
            squared_returns = returns ** 2
            vol_clustering_1 = squared_returns.autocorr(lag=1)
            vol_clustering_5 = squared_returns.autocorr(lag=5)
            vol_clustering_10 = squared_returns.autocorr(lag=10)
            
            return {
                'lag_1': float(vol_clustering_1) if not pd.isna(vol_clustering_1) else None,
                'lag_5': float(vol_clustering_5) if not pd.isna(vol_clustering_5) else None,
                'lag_10': float(vol_clustering_10) if not pd.isna(vol_clustering_10) else None
            }
        except Exception as e:
            logger.error(f"Error calculating volatility clustering: {e}")
            return {'error': str(e)}
    
    def calculate_portfolio_risk(self, returns_matrix: pd.DataFrame, weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics.
        
        Args:
            returns_matrix: DataFrame with returns for multiple assets
            weights: Portfolio weights (equal if not provided)
            
        Returns:
            Dictionary containing portfolio risk metrics
        """
        try:
            if weights is None:
                weights = [1.0 / len(returns_matrix.columns)] * len(returns_matrix.columns)
            
            weights = np.array(weights)
            
            # Portfolio returns
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            
            # Portfolio risk metrics
            portfolio_metrics = self.calculate_risk_metrics(portfolio_returns)
            
            # Correlation matrix
            correlation_matrix = returns_matrix.corr()
            
            # Portfolio variance
            covariance_matrix = returns_matrix.cov()
            portfolio_variance = weights.T @ covariance_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Component VaR (simplified)
            component_var = self._calculate_component_var(returns_matrix, weights)
            
            return {
                'portfolio_metrics': portfolio_metrics,
                'portfolio_volatility': float(portfolio_volatility),
                'correlation_matrix': correlation_matrix.to_dict(),
                'component_var': component_var
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {'error': str(e)}
    
    def _calculate_component_var(self, returns_matrix: pd.DataFrame, weights: np.ndarray) -> Dict[str, float]:
        """Calculate component VaR for portfolio."""
        try:
            # Simplified component VaR calculation
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            portfolio_var_95 = np.percentile(portfolio_returns, 5)
            
            component_var = {}
            for i, asset in enumerate(returns_matrix.columns):
                # Marginal contribution to VaR
                marginal_var = returns_matrix[asset].corr(portfolio_returns) * returns_matrix[asset].std() / portfolio_returns.std()
                component_var[asset] = float(weights[i] * marginal_var * abs(portfolio_var_95))
            
            return component_var
        except Exception as e:
            logger.error(f"Error calculating component VaR: {e}")
            return {'error': str(e)}
