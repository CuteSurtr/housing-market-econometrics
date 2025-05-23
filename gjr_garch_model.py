import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH, EGARCH
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings

warnings.filterwarnings('ignore')


class GJRGarchModel:
    """
    GJR-GARCH(1,1) model for housing returns with Fed funds as external regressor

    Model Specification:
    Return equation: r_t = μ + ε_t
    Variance equation: σ²_t = ω + α₁ε²_{t-1} + γI_{t-1}ε²_{t-1} + β₁σ²_{t-1}
    Where I_{t-1} = 1 if ε_{t-1} < 0, zero otherwise.
    """

    def __init__(self, return_series, external_regressors=None):
        """
        Initialize GJR-GARCH model

        Parameters:
        -----------
        return_series : pd.Series
            Time series of returns (e.g., housing returns)
        external_regressors : pd.DataFrame, optional
            External variables for mean equation (e.g., Fed funds rate changes)
        """
        self.return_series = return_series
        self.external_regressors = external_regressors
        self.model = None
        self.results = None

    def fit_model(self, mean_model='AR', ar_lags=1, vol_model='GARCH',
                  p=1, q=1, power=2.0, dist='StudentsT'):
        """
        Fit GJR-GARCH model using maximum likelihood estimation

        Parameters:
        -----------
        mean_model : str, default 'AR'
            Mean model specification
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
        if self.external_regressors is not None:
            exog = self.external_regressors
        else:
            exog = None

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

        self.results = self.model.fit(
            update_freq=10,
            disp='off',
            show_warning=False
        )

        return self.results

    def extract_parameters(self):
        """Extract key GJR-GARCH parameters with error handling"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        params = {}
        param_names = self.results.params.index.tolist()

        # Just extract what's available
        for param_name in param_names:
            params[param_name] = self.results.params[param_name]

        # Standardize names for compatibility
        if 'omega' not in params and 'const' in params:
            params['omega'] = params['const']

        # Find alpha parameter (ARCH effect)
        alpha_params = [p for p in param_names if 'alpha' in p.lower()]
        if alpha_params:
            params['alpha[1]'] = params[alpha_params[0]]

        # Find gamma parameter (asymmetry)
        gamma_params = [p for p in param_names if 'gamma' in p.lower()]
        if gamma_params:
            params['gamma[1]'] = params[gamma_params[0]]
        else:
            params['gamma[1]'] = 0.0  # Default if not found

        # Find beta parameter (GARCH effect)
        beta_params = [p for p in param_names if 'beta' in p.lower() and 'x' not in p.lower()]
        if beta_params:
            params['beta[1]'] = params[beta_params[0]]

        return params

    def calculate_persistence(self):
        """
        Calculate volatility persistence coefficient

        For GJR-GARCH: persistence = α + γ/2 + β

        Returns:
        --------
        persistence : float
            Volatility persistence coefficient
        """
        params = self.extract_parameters()

        persistence = (params['alpha[1]'] +
                       params['gamma[1]'] / 2 +
                       params['beta[1]'])

        return persistence

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

        forecasts = self.results.forecast(horizon=horizon)

        return {
            'mean': forecasts.mean.iloc[-1, :],
            'variance': forecasts.variance.iloc[-1, :],
            'volatility': np.sqrt(forecasts.variance.iloc[-1, :])
        }

    def diagnostic_tests(self):
        """
        Perform diagnostic tests on standardized residuals

        Returns:
        --------
        diagnostics : dict
            Dictionary containing test statistics and p-values
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        diagnostics = {}
        std_resid = self.results.std_resid

        # Ljung-Box test for serial correlation in standardized residuals
        lb_stat, lb_pvalue = acorr_ljungbox(std_resid, lags=10, return_df=False)
        diagnostics['ljung_box'] = {
            'statistic': lb_stat,
            'p_value': lb_pvalue,
            'description': 'Test for serial correlation in standardized residuals'
        }

        # ARCH-LM test for remaining heteroskedasticity
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_stat, arch_pvalue = het_arch(std_resid, nlags=5)[:2]
            diagnostics['arch_lm'] = {
                'statistic': arch_stat,
                'p_value': arch_pvalue,
                'description': 'Test for remaining ARCH effects'
            }
        except:
            diagnostics['arch_lm'] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'description': 'ARCH-LM test failed'
            }

        # Jarque-Bera test for normality of standardized residuals
        jb_stat, jb_pvalue = stats.jarque_bera(std_resid)
        diagnostics['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'description': 'Test for normality of standardized residuals'
        }

        return diagnostics

    def get_conditional_volatility(self):
        """
        Extract conditional volatility series

        Returns:
        --------
        volatility : pd.Series
            Conditional volatility time series
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        return np.sqrt(self.results.conditional_volatility)

    def get_standardized_residuals(self):
        """
        Extract standardized residuals

        Returns:
        --------
        std_resid : pd.Series
            Standardized residuals
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        return self.results.std_resid

    def plot_results(self, figsize=(12, 10)):
        """
        Plot model results and diagnostics

        Parameters:
        -----------
        figsize : tuple, default (12, 10)
            Figure size for plots

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing plots
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Returns and fitted values
        axes[0, 0].plot(self.return_series.index, self.return_series.values,
                        label='Actual Returns', alpha=0.7, color='black')
        axes[0, 0].plot(self.return_series.index, self.results.resid,
                        label='Residuals', alpha=0.7, color='red')
        axes[0, 0].set_title('Returns and Residuals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Conditional volatility
        volatility = self.get_conditional_volatility()
        axes[0, 1].plot(self.return_series.index, volatility, color='blue', linewidth=1.5)
        axes[0, 1].set_title('Conditional Volatility')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Standardized residuals
        std_resid = self.get_standardized_residuals()
        axes[1, 0].plot(self.return_series.index, std_resid, color='green', alpha=0.7)
        axes[1, 0].set_title('Standardized Residuals')
        axes[1, 0].set_ylabel('Std. Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. QQ plot
        stats.probplot(std_resid, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Standardized Residuals')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """
        Print comprehensive model summary with error handling
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        print("GJR-GARCH(1,1) Model Results")
        print("=" * 50)
        print(f"Log-Likelihood: {self.results.loglikelihood:.4f}")
        print(f"AIC: {self.results.aic:.4f}")
        print(f"BIC: {self.results.bic:.4f}")

        # Parameters
        try:
            params = self.extract_parameters()
            persistence = self.calculate_persistence()

            print(f"\nVolatility Persistence: {persistence:.4f}")
            gamma_val = params.get('gamma[1]', 0)
            print(f"Asymmetry Parameter (γ): {gamma_val:.4f}")

            if gamma_val > 0:
                print("Interpretation: Negative shocks increase volatility more than positive shocks")

            print("\nKey Parameters:")
            for param, value in params.items():
                if isinstance(value, (int, float)):
                    print(f"  {param}: {value:.6f}")
                else:
                    print(f"  {param}: {value}")
        except Exception as e:
            print(f"Error extracting parameters: {e}")

        # Diagnostics with error handling
        try:
            diagnostics = self.diagnostic_tests()
            print(f"\nDiagnostic Tests:")
            for test, results in diagnostics.items():
                try:
                    p_val = float(results['p_value'])
                    print(f"  {test}: p-value = {p_val:.4f}")
                except (ValueError, TypeError):
                    print(f"  {test}: p-value = {results['p_value']}")

                if 'description' in results:
                    print(f"    {results['description']}")
        except Exception as e:
            print(f"Error in diagnostic tests: {e}")
# Usage example function
def fit_gjr_garch_housing(return_series, fed_variables=None):
    """
    Convenience function to fit GJR-GARCH model to housing data

    Parameters:
    -----------
    return_series : pd.Series
        Housing return series
    fed_variables : pd.DataFrame, optional
        Fed funds variables (changes, volatility, etc.)

    Returns:
    --------
    model : GJRGarchModel
        Fitted GJR-GARCH model instance
    """
    # Initialize model
    gjr_model = GJRGarchModel(return_series, fed_variables)

    # Fit model
    results = gjr_model.fit_model()

    # Print summary
    gjr_model.summary()

    return gjr_model


if __name__ == "__main__":
    # Example usage
    print("GJR-GARCH Model Implementation")
    print("This module provides GJR-GARCH modeling capabilities for financial time series")
    print("Import this module and use GJRGarchModel class or fit_gjr_garch_housing function")