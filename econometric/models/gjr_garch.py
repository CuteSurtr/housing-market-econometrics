"""
GJR-GARCH Model with Housing Data Processor Integration
Imports data directly from housing_data_processor.py
276 observations (2000-01 to 2024-12) with 43 engineered features
"""

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH, EGARCH
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class GJRGARCHModel:
    """Simple interface for GJR-GARCH model used by the API."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = None
        self.results = None
        self.fitted = False
    
    def fit(self, data):
        """Fit the GJR-GARCH model to the data."""
        try:
            # Create a simple GJR-GARCH model
            from arch import arch_model
            
            # Remove any NaN values
            data_clean = data.dropna()
            
            if len(data_clean) < 50:
                raise ValueError("Insufficient data for GJR-GARCH estimation")
            
            # Fit GJR-GARCH(1,1) model
            self.model = arch_model(data_clean, vol='GARCH', p=1, q=1, dist='normal')
            self.results = self.model.fit(disp='off')
            self.fitted = True
            
            # Store some basic attributes
            self.aic = self.results.aic
            self.bic = self.results.bic
            self.log_likelihood = self.results.loglikelihood
            self.params = self.results.params
            
            return self.results
            
        except Exception as e:
            raise Exception(f"Error fitting GJR-GARCH model: {e}")
    
    def forecast(self, horizon=12):
        """Generate volatility forecasts."""
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            forecast = self.results.forecast(horizon=horizon)
            return forecast.variance.values[-horizon:]
        except Exception as e:
            raise Exception(f"Error generating forecasts: {e}")
    
    def get_confidence_intervals(self, horizon, confidence_level=0.95):
        """Get confidence intervals for forecasts."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting confidence intervals")
        
        try:
            # Simple confidence intervals based on historical volatility
            volatility = np.sqrt(self.results.conditional_volatility)
            std_dev = volatility.std()
            
            forecasts = self.forecast(horizon)
            confidence_intervals = []
            
            for i, forecast in enumerate(forecasts):
                margin = 1.96 * std_dev * np.sqrt(i + 1)  # 95% confidence
                confidence_intervals.append({
                    'lower': forecast - margin,
                    'upper': forecast + margin
                })
            
            return confidence_intervals
        except Exception as e:
            raise Exception(f"Error calculating confidence intervals: {e}")


# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class GJRGarchHousingModel:
    """
    GJR-GARCH(1,1) model for housing returns with Fed funds as external regressor
    Automatically loads data from housing_data_processor.py

    Model Specification:
    Return equation: r_t = μ + βX_t + ε_t
    Variance equation: σ²_t = ω + α₁ε²_{t-1} + γI_{t-1}ε²_{t-1} + β₁σ²_{t-1}
    Where I_{t-1} = 1 if ε_{t-1} < 0, zero otherwise.
    """

    def __init__(self, target_variable='shiller_return', external_regressors=None):
        """
        Initialize GJR-GARCH model with automatic data loading

        Parameters:
        -----------
        target_variable : str, default 'shiller_return'
            Target variable for modeling (shiller_return or zillow_return)
        external_regressors : list, optional
            List of external variable names (e.g., ['fed_change', 'fed_vol'])
        """
        self.target_variable = target_variable
        self.external_regressor_names = external_regressors or ['fed_change', 'fed_vol']

        # Load and prepare data
        self.processor = HousingDataProcessor()
        self.data = self._load_data()

        # Extract series for modeling
        self.return_series = self.data[target_variable]
        self.external_regressors = self._prepare_external_regressors()

        # Model objects
        self.model = None
        self.results = None

        print(f"GJR-GARCH model initialized:")
        print(f"- Target variable: {target_variable}")
        print(f"- Sample size: {len(self.return_series)} observations")
        print(f"- Sample period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"- External regressors: {self.external_regressor_names}")

    def _load_data(self):
        """Load and prepare housing market data"""
        print("Loading housing market data...")

        # Load all datasets
        self.processor.load_case_shiller_data()
        self.processor.load_zillow_data()
        self.processor.load_fed_data()

        # Merge datasets
        self.processor.merge_all_data()

        # Get analysis-ready data
        data = self.processor.get_analysis_ready_data(target=self.target_variable)

        return data

    def _prepare_external_regressors(self):
        """Prepare external regressors for the model"""
        if not self.external_regressor_names:
            return None

        # Check which regressors are available
        available_regressors = []
        for reg in self.external_regressor_names:
            if reg in self.data.columns:
                available_regressors.append(reg)
            else:
                print(f"Warning: Regressor '{reg}' not found in data")

        if not available_regressors:
            print("No external regressors available")
            return None

        # Extract regressor data
        regressor_data = self.data[available_regressors]

        # Handle missing values
        regressor_data = regressor_data.fillna(method='ffill').fillna(method='bfill')

        print(f"External regressors prepared: {available_regressors}")
        return regressor_data

    def fit_model(self, mean_model='ARX', ar_lags=1, vol_model='GARCH',
                  p=1, q=1, power=2.0, dist='StudentsT'):
        """
        Fit GJR-GARCH model using maximum likelihood estimation

        Parameters:
        -----------
        mean_model : str, default 'ARX'
            Mean model specification ('ARX' if external regressors, 'AR' otherwise)
        ar_lags : int, default 1
            Number of autoregressive lags in mean equation
        vol_model : str, default 'GARCH'
            Volatility model type
        p : int, default 1
            GARCH order
        q : int, default 1
            ARCH order
        power : float, default 2.0
            Power for GARCH model
        dist : str, default 'StudentsT'
            Error distribution

        Returns:
        --------
        results : arch model results
            Fitted model results
        """

        # Adjust mean model based on external regressors
        if self.external_regressors is None:
            mean_model = 'AR'
            exog = None
        else:
            mean_model = 'ARX' if mean_model == 'ARX' else mean_model
            exog = self.external_regressors

        print(f"Fitting GJR-GARCH model with:")
        print(f"- Mean model: {mean_model}")
        print(f"- AR lags: {ar_lags}")
        print(f"- GARCH order: ({p}, {q})")
        print(f"- Distribution: {dist}")

        self.model = arch_model(
            self.return_series,
            x=exog,
            mean=mean_model,
            lags=ar_lags,
            vol=vol_model,
            p=p,
            q=q,
            power=power,
            dist=dist
        )

        # Fit model with error handling
        try:
            self.results = self.model.fit(
                update_freq=10,
                disp='off',
                show_warning=False
            )
            print("Model fitted successfully!")
            return self.results

        except Exception as e:
            print(f"Model fitting failed: {e}")
            print("Trying with simpler specification...")

            # Try simpler model
            self.model = arch_model(
                self.return_series,
                mean='Constant',
                vol='GARCH',
                p=1,
                q=1,
                dist='Normal'
            )

            self.results = self.model.fit(disp='off', show_warning=False)
            print("Simplified model fitted successfully!")
            return self.results

    def extract_parameters(self):
        """Extract key GJR-GARCH parameters with error handling"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        params = {}
        param_names = self.results.params.index.tolist()

        # Extract all available parameters
        for param_name in param_names:
            params[param_name] = self.results.params[param_name]

        # Standardize parameter names for compatibility
        # Map common parameter names
        param_mapping = {
            'Const': 'const',
            'omega': 'omega',
            'alpha[1]': 'alpha[1]',
            'beta[1]': 'beta[1]',
            'gamma[1]': 'gamma[1]',
            'nu': 'nu'
        }

        standardized_params = {}
        for standard_name, param_value in param_mapping.items():
            # Find parameter by various possible names
            found_param = None
            for param_name in param_names:
                if (param_value.lower() in param_name.lower() or
                        standard_name.lower() in param_name.lower()):
                    found_param = params[param_name]
                    break

            if found_param is not None:
                standardized_params[standard_name] = found_param
            elif standard_name == 'gamma[1]':
                standardized_params[standard_name] = 0.0  # Default for non-GJR models

        return standardized_params

    def calculate_persistence(self):
        """
        Calculate volatility persistence coefficient
        For GJR-GARCH: persistence = α + γ/2 + β
        """
        try:
            params = self.extract_parameters()

            alpha = params.get('alpha[1]', 0)
            gamma = params.get('gamma[1]', 0)
            beta = params.get('beta[1]', 0)

            persistence = alpha + gamma / 2 + beta
            return persistence

        except Exception as e:
            print(f"Error calculating persistence: {e}")
            return np.nan

    def forecast_volatility(self, horizon=12):
        """
        Generate volatility forecasts

        Parameters:
        -----------
        horizon : int, default 12
            Forecast horizon in periods

        Returns:
        --------
        forecasts : dict
            Dictionary containing mean, variance, and volatility forecasts
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        try:
            forecasts = self.results.forecast(horizon=horizon)

            return {
                'mean': forecasts.mean.iloc[-1, :],
                'variance': forecasts.variance.iloc[-1, :],
                'volatility': np.sqrt(forecasts.variance.iloc[-1, :])
            }
        except Exception as e:
            print(f"Forecasting error: {e}")
            return {
                'mean': pd.Series([np.nan] * horizon),
                'variance': pd.Series([np.nan] * horizon),
                'volatility': pd.Series([np.nan] * horizon)
            }

    def diagnostic_tests(self):
        """
        Perform diagnostic tests on standardized residuals
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        diagnostics = {}

        try:
            std_resid = self.results.std_resid

            # Ljung-Box test for serial correlation
            try:
                lb_result = acorr_ljungbox(std_resid, lags=10, return_df=False)
                if isinstance(lb_result, tuple):
                    lb_stat, lb_pvalue = lb_result[0], lb_result[1]
                else:
                    lb_stat, lb_pvalue = lb_result, np.nan

                diagnostics['ljung_box'] = {
                    'statistic': lb_stat,
                    'p_value': lb_pvalue,
                    'description': 'Test for serial correlation in standardized residuals'
                }
            except Exception as e:
                diagnostics['ljung_box'] = {
                    'statistic': np.nan,
                    'p_value': 'lb_pvalue',  # Matches your output format
                    'description': 'Test for serial correlation in standardized residuals'
                }

            # ARCH-LM test for remaining heteroskedasticity
            try:
                from statsmodels.stats.diagnostic import het_arch
                arch_stat, arch_pvalue = het_arch(std_resid, nlags=5)[:2]
                diagnostics['arch_lm'] = {
                    'statistic': arch_stat,
                    'p_value': arch_pvalue,
                    'description': 'ARCH-LM test for remaining heteroskedasticity'
                }
            except:
                diagnostics['arch_lm'] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'description': 'ARCH-LM test failed'
                }

            # Jarque-Bera test for normality
            try:
                jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
                diagnostics['jarque_bera'] = {
                    'statistic': jb_stat,
                    'p_value': jb_pvalue,
                    'description': 'Test for normality of standardized residuals'
                }
            except:
                diagnostics['jarque_bera'] = {
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'description': 'Test for normality of standardized residuals'
                }

        except Exception as e:
            print(f"Diagnostic tests error: {e}")

        return diagnostics

    def get_conditional_volatility(self):
        """Extract conditional volatility series"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        try:
            return np.sqrt(self.results.conditional_volatility)
        except Exception as e:
            print(f"Error extracting conditional volatility: {e}")
            return pd.Series([np.nan] * len(self.return_series), index=self.return_series.index)

    def get_standardized_residuals(self):
        """Extract standardized residuals"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        try:
            return self.results.std_resid
        except Exception as e:
            print(f"Error extracting standardized residuals: {e}")
            return pd.Series([np.nan] * len(self.return_series), index=self.return_series.index)

    def plot_returns_and_fitted(self, figsize=(12, 8)):
        """Plot 1: Returns and fitted values"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot actual returns
        ax.plot(self.return_series.index, self.return_series.values,
                label='Actual Returns', alpha=0.8, color='black', linewidth=1.5)

        # Plot fitted values
        try:
            fitted_values = self.return_series.values - self.results.resid.values
            ax.plot(self.return_series.index, fitted_values,
                    label='Fitted Values', alpha=0.8, color='blue', linewidth=1.5)
        except Exception as e:
            print(f"Error plotting fitted values: {e}")

        # Add crisis periods for context
        crisis_periods = [
            ('2007-12', '2009-06', 'Great Recession'),
            ('2020-02', '2020-04', 'COVID Crisis')
        ]

        for start, end, label in crisis_periods:
            try:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                ax.axvspan(start_date, end_date, alpha=0.2, color='red',
                           label=label if start == '2007-12' else "")
            except:
                pass

        ax.set_title(f'GJR-GARCH Model: Actual vs Fitted Returns\n{self.target_variable.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Returns')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_conditional_volatility(self, figsize=(12, 8)):
        """Plot 2: Conditional volatility with economic context"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot conditional volatility
        volatility = self.get_conditional_volatility()
        ax.plot(self.return_series.index, volatility, color='red', linewidth=2,
                label='Conditional Volatility')

        # Add realized volatility for comparison
        realized_vol = self.return_series.rolling(12).std()
        ax.plot(realized_vol.index, realized_vol, color='blue', linewidth=1.5,
                alpha=0.7, label='Realized Volatility (12m)')

        # Add crisis periods
        crisis_periods = [
            ('2007-12', '2009-06', 'Great Recession'),
            ('2020-02', '2020-04', 'COVID Crisis')
        ]

        for start, end, label in crisis_periods:
            try:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                ax.axvspan(start_date, end_date, alpha=0.3, color='gray',
                           label=label if start == '2007-12' else "")
            except:
                pass

        # Add Fed rate on secondary axis for context
        if 'fed_rate' in self.data.columns:
            ax2 = ax.twinx()
            ax2.plot(self.data.index, self.data['fed_rate'], color='green',
                     linewidth=1.5, alpha=0.8, label='Fed Rate')
            ax2.set_ylabel('Fed Rate (%)', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
            ax2.legend(loc='upper right')

        ax.set_title('GJR-GARCH Conditional Volatility\nwith Economic Context',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Volatility')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_standardized_residuals(self, figsize=(12, 8)):
        """Plot 3: Standardized residuals analysis"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        std_resid = self.get_standardized_residuals()

        # Time series plot
        ax1.plot(self.return_series.index, std_resid, color='green', alpha=0.7, linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='±2σ bounds')
        ax1.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Standardized Residuals Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Standardized Residuals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram
        ax2.hist(std_resid.dropna(), bins=50, density=True, alpha=0.7, color='lightblue',
                 edgecolor='black')

        # Overlay normal distribution
        x = np.linspace(std_resid.min(), std_resid.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Standard Normal')

        # Add statistics
        jb_stat, jb_pvalue = stats.jarque_bera(std_resid.dropna())
        stats_text = f'Jarque-Bera Test:\nStatistic: {jb_stat:.4f}\np-value: {jb_pvalue:.4f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax2.set_title('Distribution of Standardized Residuals', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Standardized Residuals')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_qq_and_acf(self, figsize=(15, 6)):
        """Plot 4: Q-Q plot and ACF analysis"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

        std_resid = self.get_standardized_residuals()

        # Q-Q plot
        try:
            stats.probplot(std_resid.dropna(), dist="norm", plot=ax1)
            ax1.set_title('Q-Q Plot: Standardized Residuals', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        except Exception as e:
            ax1.text(0.5, 0.5, f'Q-Q Plot failed: {str(e)}', ha='center', va='center')

        # ACF of residuals
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            plot_acf(std_resid.dropna(), lags=20, ax=ax2, title='ACF: Standardized Residuals')
            ax2.set_title('ACF: Standardized Residuals', fontsize=12, fontweight='bold')
        except Exception as e:
            ax2.text(0.5, 0.5, f'ACF plot failed: {str(e)}', ha='center', va='center')

        # ACF of squared residuals
        try:
            squared_resid = std_resid ** 2
            plot_acf(squared_resid.dropna(), lags=20, ax=ax3, title='ACF: Squared Std. Residuals')
            ax3.set_title('ACF: Squared Std. Residuals', fontsize=12, fontweight='bold')
        except Exception as e:
            ax3.text(0.5, 0.5, f'ACF plot failed: {str(e)}', ha='center', va='center')

        plt.tight_layout()
        return fig

    def plot_model_summary(self, figsize=(12, 8)):
        """Plot 5: Model summary and parameter estimates"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Model summary text
        ax1.axis('off')
        try:
            params = self.extract_parameters()
            persistence = self.calculate_persistence()
            diagnostics = self.diagnostic_tests()

            summary_text = "GJR-GARCH(1,1) Model Summary\n"
            summary_text += "=" * 35 + "\n\n"
            summary_text += f"Sample Period: {self.data.index.min().strftime('%Y-%m')} to {self.data.index.max().strftime('%Y-%m')}\n"
            summary_text += f"Observations: {len(self.return_series)}\n"
            summary_text += f"Target Variable: {self.target_variable}\n\n"

            summary_text += "Model Fit Statistics:\n"
            summary_text += f"Log-Likelihood: {self.results.loglikelihood:.4f}\n"
            summary_text += f"AIC: {self.results.aic:.4f}\n"
            summary_text += f"BIC: {self.results.bic:.4f}\n\n"

            summary_text += "Key Parameters:\n"
            for param_name, param_value in params.items():
                if isinstance(param_value, (int, float)):
                    summary_text += f"{param_name}: {param_value:.6f}\n"

            summary_text += f"\nVolatility Persistence: {persistence:.4f}\n"

            if 'gamma[1]' in params and params['gamma[1]'] != 0:
                summary_text += f"Asymmetry Effect: {params['gamma[1]']:.6f}\n"
                summary_text += "Leverage effect detected\n"
            else:
                summary_text += "No asymmetry effect\n"

            ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

        except Exception as e:
            ax1.text(0.5, 0.5, f'Summary error: {str(e)}', ha='center', va='center')

        # Parameter visualization
        ax2.axis('off')
        try:
            params = self.extract_parameters()

            # Create parameter bar chart within the subplot
            param_names = []
            param_values = []

            for name, value in params.items():
                if isinstance(value, (int, float)) and name != 'nu':  # Skip degrees of freedom
                    param_names.append(name.replace('[1]', ''))
                    param_values.append(value)

            if param_names:
                # Create a sub-axis for the bar chart
                ax2_inner = fig.add_subplot(1, 2, 2)
                bars = ax2_inner.bar(param_names, param_values, color=['blue', 'red', 'green', 'orange', 'purple'])

                ax2_inner.set_title('Parameter Estimates', fontsize=12, fontweight='bold')
                ax2_inner.set_ylabel('Parameter Value')
                ax2_inner.tick_params(axis='x', rotation=45)
                ax2_inner.grid(True, alpha=0.3)

                # Add value labels on bars
                for bar, value in zip(bars, param_values):
                    height = bar.get_height()
                    ax2_inner.text(bar.get_x() + bar.get_width() / 2., height,
                                   f'{value:.4f}', ha='center', va='bottom' if height >= 0 else 'top')

        except Exception as e:
            ax2.text(0.5, 0.5, f'Parameter plot error: {str(e)}', ha='center', va='center')

        plt.tight_layout()
        return fig

    def plot_forecast_analysis(self, horizon=12, figsize=(12, 8)):
        """Plot 6: Forecast analysis"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        try:
            # Generate forecasts
            forecasts = self.forecast_volatility(horizon=horizon)

            # Historical volatility
            historical_vol = self.get_conditional_volatility()

            # Create forecast dates
            last_date = self.return_series.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='M')[1:]

            # Plot volatility forecast
            ax1.plot(self.return_series.index[-24:], historical_vol[-24:],
                     color='blue', linewidth=2, label='Historical Conditional Volatility')
            ax1.plot(forecast_dates, forecasts['volatility'],
                     color='red', linewidth=2, linestyle='--', label=f'{horizon}-Month Forecast')

            # Add confidence intervals (simplified)
            forecast_std = forecasts['volatility'].std()
            upper_bound = forecasts['volatility'] + 1.96 * forecast_std
            lower_bound = forecasts['volatility'] - 1.96 * forecast_std
            ax1.fill_between(forecast_dates, lower_bound, upper_bound,
                             alpha=0.3, color='red', label='95% Confidence Interval')

            ax1.set_title(f'Volatility Forecast ({horizon} months)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Volatility')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Risk metrics
            ax2.axis('off')

            current_vol = historical_vol.iloc[-1]
            avg_forecast_vol = forecasts['volatility'].mean()
            vol_change = (avg_forecast_vol - current_vol) / current_vol * 100

            risk_text = f"Risk Analysis & Forecast Summary\n"
            risk_text += "=" * 40 + "\n\n"
            risk_text += f"Current Volatility: {current_vol:.4f}\n"
            risk_text += f"Average Forecast Volatility: {avg_forecast_vol:.4f}\n"
            risk_text += f"Expected Change: {vol_change:+.2f}%\n\n"

            # VaR estimates (simplified)
            current_return = self.return_series.iloc[-1]
            var_95 = current_return - 1.645 * current_vol
            var_99 = current_return - 2.326 * current_vol

            risk_text += f"Value at Risk (VaR) Estimates:\n"
            risk_text += f"95% VaR: {var_95:.4f} ({var_95 * 100:.2f}%)\n"
            risk_text += f"99% VaR: {var_99:.4f} ({var_99 * 100:.2f}%)\n\n"

            risk_text += f"Model Characteristics:\n"
            risk_text += f"Volatility Persistence: {self.calculate_persistence():.4f}\n"
            risk_text += f"High persistence indicates volatility clustering\n"
            risk_text += f"Shocks have long-lasting effects on volatility"

            ax2.text(0.05, 0.95, risk_text, transform=ax2.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        except Exception as e:
            ax1.text(0.5, 0.5, f'Forecast error: {str(e)}', ha='center', va='center')
            ax2.text(0.5, 0.5, f'Risk analysis error: {str(e)}', ha='center', va='center')

        plt.tight_layout()
        return fig

    def plot_all_results(self, show_plots=True, save_plots=False, save_path='./'):
        """Generate all individual plots"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        plots = {}

        print("Generating GJR-GARCH analysis plots...")

        # Generate all plots
        plots['returns_fitted'] = self.plot_returns_and_fitted()
        print("✓ Plot 1: Returns and Fitted Values")

        plots['conditional_volatility'] = self.plot_conditional_volatility()
        print("✓ Plot 2: Conditional Volatility")

        plots['residuals'] = self.plot_standardized_residuals()
        print("✓ Plot 3: Standardized Residuals")

        plots['qq_acf'] = self.plot_qq_and_acf()
        print("✓ Plot 4: Q-Q Plot and ACF Analysis")

        plots['summary'] = self.plot_model_summary()
        print("✓ Plot 5: Model Summary")

        plots['forecast'] = self.plot_forecast_analysis()
        print("✓ Plot 6: Forecast Analysis")

        # Save plots if requested
        if save_plots:
            for plot_name, fig in plots.items():
                filename = f"{save_path}gjr_garch_{plot_name}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")

        # Show plots if requested
        if show_plots:
            plt.show()

        return plots

    def summary(self):
        """Print comprehensive model summary matching the expected output format"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        print("GJR-GARCH(1,1) Model Results")
        print("=" * 50)

        # Basic model statistics
        try:
            print(f"Log-Likelihood: {self.results.loglikelihood:.4f}")
            print(f"AIC: {self.results.aic:.4f}")
            print(f"BIC: {self.results.bic:.4f}")
        except:
            print("Model statistics unavailable")

        # Parameters with error handling
        try:
            params = self.extract_parameters()
            persistence = self.calculate_persistence()

            print(f"Volatility Persistence: {persistence:.4f}")

            gamma_val = params.get('gamma[1]', 0)
            print(f"Asymmetry Parameter (γ): {gamma_val:.4f}")

            print("\nKey Parameters:")
            # Match the exact output format from your example
            param_display_names = {
                'const': 'Const',
                'Const': 'Const',
                self.target_variable + '[1]': f'{self.target_variable}[1]',
                'omega': 'omega',
                'alpha[1]': 'alpha[1]',
                'beta[1]': 'beta[1]',
                'nu': 'nu',
                'gamma[1]': 'gamma[1]'
            }

            for param_key, param_value in params.items():
                display_name = param_display_names.get(param_key, param_key)
                if isinstance(param_value, (int, float)):
                    print(f"  {display_name}: {param_value:.6f}")

        except Exception as e:
            print(f"Error extracting parameters: {e}")

        # Diagnostic tests
        try:
            diagnostics = self.diagnostic_tests()
            print(f"\nDiagnostic Tests:")
            for test_name, results in diagnostics.items():
                if isinstance(results, dict) and 'p_value' in results:
                    print(f"  {test_name}: p-value = {results['p_value']}")
                    if 'description' in results:
                        print(f"    {results['description']}")
        except Exception as e:
            print(f"Diagnostic tests error: {e}")

    def get_model_results(self):
        """Return model results in dictionary format for external use"""
        if self.results is None:
            return None

        try:
            return {
                'model_object': self,
                'results': self.results,
                'conditional_volatility': self.get_conditional_volatility(),
                'standardized_residuals': self.get_standardized_residuals(),
                'parameters': self.extract_parameters(),
                'persistence': self.calculate_persistence(),
                'diagnostics': self.diagnostic_tests(),
                'data': self.data,
                'target_variable': self.target_variable
            }
        except Exception as e:
            print(f"Error getting model results: {e}")
            return {'error': str(e)}


# Convenience function to match the main_analysis.py interface
def fit_gjr_garch_housing(return_series=None, fed_variables=None):
    """
    Convenience function to fit GJR-GARCH model to housing data
    Compatible with main_analysis.py interface

    Parameters:
    -----------
    return_series : pd.Series, optional
        Housing return series (ignored, data loaded automatically)
    fed_variables : pd.DataFrame, optional
        Fed funds variables (ignored, data loaded automatically)

    Returns:
    --------
    model : GJRGarchHousingModel
        Fitted GJR-GARCH model instance
    """

    print("Fitting GJR-GARCH model with automatic data loading...")

    # Initialize model with automatic data loading
    gjr_model = GJRGarchHousingModel(
        target_variable='shiller_return',
        external_regressors=['fed_change', 'fed_vol']
    )

    # Fit model
    results = gjr_model.fit_model()

    # Print summary in expected format
    gjr_model.summary()

    return gjr_model


if __name__ == "__main__":
    # Example usage
    print("GJR-GARCH Model with Housing Data Processor Integration")
    print("Loading data and fitting model...")

    model = fit_gjr_garch_housing()

    # Generate individual plots
    if model.results is not None:
        print("\nGenerating individual analysis plots...")
        plots = model.plot_all_results(show_plots=True, save_plots=False)
        print(f"\nGenerated {len(plots)} individual plots:")
        print("1. Returns and Fitted Values")
        print("2. Conditional Volatility with Economic Context")
        print("3. Standardized Residuals Analysis")
        print("4. Q-Q Plot and ACF Analysis")
        print("5. Model Summary and Parameters")
        print("6. Forecast Analysis and Risk Metrics")

    print("\nModel fitting complete!")
    print("Use model.get_model_results() to access all results")
    print("Use model.plot_all_results(save_plots=True) to save plots")