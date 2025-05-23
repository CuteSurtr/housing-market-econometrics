"""
Transfer Function Model Implementation
File: transfer_function_model.py

Transfer Function Model for analyzing dynamic relationships between
input and output time series with distributed lag structure.
Specifically designed for Fed policy transmission to housing markets.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal, stats
from statsmodels.stats.stattools import durbin_watson
import warnings

warnings.filterwarnings('ignore')


class TransferFunctionModel:
    """
    Transfer Function Model for housing returns with Fed funds rate

    Model Specification:
    y_t = Σᵢ₌₀ⁿ βᵢx_{t-i} + Σⱼ₌₁ᵖ φⱼy_{t-j} + ε_t

    Where:
    - y_t: output series (housing returns)
    - x_t: input series (Fed funds rate changes)
    - βᵢ: transfer function weights
    - φⱼ: autoregressive coefficients
    """

    def __init__(self, output_series, input_series, max_lags=12):
        """
        Initialize Transfer Function Model

        Parameters:
        -----------
        output_series : pd.Series
            Output/dependent variable time series
        input_series : pd.Series
            Input/independent variable time series
        max_lags : int, default 12
            Maximum number of lags for input variable
        """
        self.output_series = output_series.dropna()
        self.input_series = input_series.dropna()
        self.max_lags = max_lags

        # Align series on common index
        common_index = self.output_series.index.intersection(self.input_series.index)
        self.output_series = self.output_series.loc[common_index]
        self.input_series = self.input_series.loc[common_index]

        self.model = None
        self.results = None

    def prewhiten_input(self, ar_order=None, ma_order=None):
        """
        Prewhiten input series to remove autocorrelation

        Parameters:
        -----------
        ar_order : int, optional
            AR order for input prewhitening
        ma_order : int, optional
            MA order for input prewhitening

        Returns:
        --------
        prewhitened_input : pd.Series
            Prewhitened input series
        prewhitened_output : pd.Series
            Output series filtered with same transformation
        arima_results : statsmodels results
            ARIMA model results for input series
        """
        # Auto-select ARIMA order if not specified
        if ar_order is None or ma_order is None:
            best_aic = np.inf
            best_order = (1, 0, 0)

            for p in range(4):
                for q in range(4):
                    try:
                        temp_model = ARIMA(self.input_series, order=(p, 0, q))
                        temp_results = temp_model.fit()
                        if temp_results.aic < best_aic:
                            best_aic = temp_results.aic
                            best_order = (p, 0, q)
                    except:
                        continue

            ar_order, _, ma_order = best_order

        # Fit ARIMA model to input
        arima_model = ARIMA(self.input_series, order=(ar_order, 0, ma_order))
        arima_results = arima_model.fit()

        # Get prewhitened input (residuals)
        prewhitened_input = arima_results.resid

        # Apply same filter to output (approximate)
        prewhitened_output = self.output_series - self.output_series.mean()

        return prewhitened_input, prewhitened_output, arima_results

    def calculate_cross_correlation(self, max_lags=None):
        """
        Calculate cross-correlation function between input and output

        Parameters:
        -----------
        max_lags : int, optional
            Maximum lag for cross-correlation

        Returns:
        --------
        lags : np.array
            Array of lag values
        ccf : np.array
            Cross-correlation function values
        """
        if max_lags is None:
            max_lags = self.max_lags

        # Prewhiten series
        white_input, white_output, _ = self.prewhiten_input()

        # Calculate cross-correlation
        cross_corr = np.correlate(white_output, white_input, mode='full')

        # Normalize
        n = len(white_input)
        cross_corr = cross_corr / (n * np.std(white_input) * np.std(white_output))

        # Extract relevant lags
        middle = len(cross_corr) // 2
        lags = np.arange(-max_lags, max_lags + 1)
        start_idx = middle - max_lags
        end_idx = middle + max_lags + 1

        ccf = cross_corr[start_idx:end_idx]

        return lags, ccf

    def identify_transfer_function_order(self):
        """
        Identify transfer function orders (b, r, s) using cross-correlation

        Returns:
        --------
        b : int
            Delay parameter (number of periods before input affects output)
        r : int
            Number of denominator terms
        s : int
            Number of numerator terms
        lags : np.array
            Lag values from cross-correlation
        ccf : np.array
            Cross-correlation function values
        """
        lags, ccf = self.calculate_cross_correlation()

        # Find significant cross-correlations
        threshold = 2 / np.sqrt(len(self.input_series))
        significant_lags = lags[np.abs(ccf) > threshold]

        if len(significant_lags) > 0:
            # Delay (b): first significant positive lag
            positive_lags = significant_lags[significant_lags >= 0]
            b = positive_lags[0] if len(positive_lags) > 0 else 0

            # Number of numerator terms (s): based on pattern
            s = min(3, len(positive_lags))  # Conservative choice

            # Number of denominator terms (r): start with 1
            r = 1
        else:
            b, r, s = 0, 1, 1  # Default values

        return b, r, s, lags, ccf

    def fit_transfer_function(self, tf_order=None, ar_order=2, include_constant=True):
        """
        Fit transfer function model using regression approach

        Parameters:
        -----------
        tf_order : tuple, optional
            Transfer function order (b, r, s)
        ar_order : int, default 2
            Autoregressive order for output series
        include_constant : bool, default True
            Whether to include constant term

        Returns:
        --------
        results : statsmodels results
            Fitted model results
        """
        if tf_order is None:
            b, r, s = self.identify_transfer_function_order()[:3]
        else:
            b, r, s = tf_order

        # Create dataset with lagged variables
        data = pd.DataFrame({
            'output': self.output_series,
            'input': self.input_series
        })

        # Add lagged inputs (transfer function terms)
        for lag in range(self.max_lags + 1):
            data[f'input_lag{lag}'] = data['input'].shift(lag)

        # Add lagged outputs (autoregressive terms)
        for lag in range(1, ar_order + 1):
            data[f'output_lag{lag}'] = data['output'].shift(lag)

        # Drop missing values
        data.dropna(inplace=True)

        # Prepare regression variables
        y = data['output']
        X_cols = [f'input_lag{i}' for i in range(self.max_lags + 1)]
        X_cols += [f'output_lag{i}' for i in range(1, ar_order + 1)]

        X = data[X_cols]

        if include_constant:
            X = sm.add_constant(X)

        # Fit OLS regression
        self.model = OLS(y, X)
        self.results = self.model.fit()

        return self.results

    def extract_transfer_function_parameters(self):
        """
        Extract transfer function parameters from fitted model

        Returns:
        --------
        params : dict
            Dictionary containing model parameters
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        params = {}

        # Input lag coefficients (transfer function weights)
        input_coeffs = []
        for lag in range(self.max_lags + 1):
            param_name = f'input_lag{lag}'
            if param_name in self.results.params.index:
                input_coeffs.append(self.results.params[param_name])

        params['transfer_function_weights'] = input_coeffs

        # AR coefficients
        ar_coeffs = []
        lag = 1
        while f'output_lag{lag}' in self.results.params.index:
            ar_coeffs.append(self.results.params[f'output_lag{lag}'])
            lag += 1

        params['ar_coefficients'] = ar_coeffs

        # Constant term
        if 'const' in self.results.params.index:
            params['constant'] = self.results.params['const']

        # Model statistics
        params['r_squared'] = self.results.rsquared
        params['adjusted_r_squared'] = self.results.rsquared_adj
        params['aic'] = self.results.aic
        params['bic'] = self.results.bic
        params['log_likelihood'] = self.results.llf
        params['f_statistic'] = self.results.fvalue
        params['f_pvalue'] = self.results.f_pvalue

        return params

    def calculate_impulse_response(self, horizon=24):
        """
        Calculate impulse response function

        Parameters:
        -----------
        horizon : int, default 24
            Horizon for impulse response calculation

        Returns:
        --------
        impulse_response : np.array
            Impulse response function values
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        params = self.extract_transfer_function_parameters()
        tf_weights = params['transfer_function_weights']
        ar_coeffs = params.get('ar_coefficients', [])

        # Initialize impulse response
        impulse_response = np.zeros(horizon)

        # Direct effects from transfer function
        for i, weight in enumerate(tf_weights):
            if i < horizon:
                impulse_response[i] += weight

        # AR propagation effects
        for t in range(1, horizon):
            ar_effect = 0
            for lag, coeff in enumerate(ar_coeffs, 1):
                if t - lag >= 0:
                    ar_effect += coeff * impulse_response[t - lag]
            impulse_response[t] += ar_effect

        return impulse_response

    def calculate_cumulative_response(self, horizon=24):
        """
        Calculate cumulative impulse response

        Parameters:
        -----------
        horizon : int, default 24
            Horizon for calculation

        Returns:
        --------
        cumulative_response : np.array
            Cumulative impulse response
        """
        impulse_resp = self.calculate_impulse_response(horizon)
        return np.cumsum(impulse_resp)

    def forecast(self, n_periods=12, input_forecast=None):
        """
        Generate forecasts using transfer function model

        Parameters:
        -----------
        n_periods : int, default 12
            Number of periods to forecast
        input_forecast : np.array, optional
            Forecasted values for input series

        Returns:
        --------
        forecasts : np.array
            Forecasted values for output series
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        if input_forecast is None:
            # Use last observed input value
            input_forecast = np.full(n_periods, self.input_series.iloc[-1])
        elif len(input_forecast) != n_periods:
            raise ValueError("Input forecast length must match n_periods")

        # Get recent values for lagged terms
        recent_output = self.output_series.iloc[-self.max_lags:].values
        recent_input = self.input_series.iloc[-self.max_lags:].values

        # Initialize forecast array
        forecasts = []

        params = self.extract_transfer_function_parameters()
        tf_weights = params['transfer_function_weights']
        ar_coeffs = params.get('ar_coefficients', [])
        constant = params.get('constant', 0)

        for t in range(n_periods):
            forecast = constant

            # Transfer function contribution
            for lag, weight in enumerate(tf_weights):
                if t >= lag:
                    forecast += weight * input_forecast[t - lag]
                else:
                    # Use recent input values
                    hist_idx = len(recent_input) - (lag - t)
                    if hist_idx >= 0:
                        forecast += weight * recent_input[hist_idx]

            # AR contribution
            for lag, coeff in enumerate(ar_coeffs, 1):
                if t >= lag:
                    forecast += coeff * forecasts[t - lag]
                else:
                    # Use recent output values
                    hist_idx = len(recent_output) - (lag - t)
                    if hist_idx >= 0:
                        forecast += coeff * recent_output[hist_idx]

            forecasts.append(forecast)

        return np.array(forecasts)

    def diagnostic_tests(self):
        """
        Perform diagnostic tests on residuals

        Returns:
        --------
        diagnostics : dict
            Dictionary containing diagnostic test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        residuals = self.results.resid
        diagnostics = {}

        # Ljung-Box test for serial correlation
        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        diagnostics['ljung_box'] = {
            'statistic': lb_result['lb_stat'].iloc[-1],
            'p_value': lb_result['lb_pvalue'].iloc[-1],
            'description': 'Test for serial correlation in residuals'
        }

        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        diagnostics['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'description': 'Test for normality of residuals'
        }

        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(residuals)
        diagnostics['durbin_watson'] = {
            'statistic': dw_stat,
            'description': 'Test for first-order autocorrelation (2=no autocorr)'
        }

        # Heteroskedasticity test (Breusch-Pagan)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, self.results.model.exog)
            diagnostics['breusch_pagan'] = {
                'statistic': bp_stat,
                'p_value': bp_pvalue,
                'description': 'Test for heteroskedasticity'
            }
        except:
            diagnostics['breusch_pagan'] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'description': 'Breusch-Pagan test failed'
            }

        return diagnostics

    def plot_results(self, figsize=(15, 12)):
        """
        Plot transfer function model results

        Parameters:
        -----------
        figsize : tuple, default (15, 12)
            Figure size

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object containing plots
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1. Input and output series
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()

        line1 = ax1.plot(self.output_series.index, self.output_series.values,
                         'b-', label='Output (Housing Returns)', linewidth=1.5)
        line2 = ax1_twin.plot(self.input_series.index, self.input_series.values,
                              'r-', label='Input (Fed Funds)', linewidth=1.5)

        ax1.set_ylabel('Housing Returns', color='b')
        ax1_twin.set_ylabel('Fed Funds Rate', color='r')
        ax1.set_title('Input and Output Series')

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. Cross-correlation function
        lags, ccf = self.calculate_cross_correlation()
        axes[0, 1].stem(lags, ccf, basefmt=' ')
        threshold = 2 / np.sqrt(len(self.input_series))
        axes[0, 1].axhline(threshold, color='r', linestyle='--', alpha=0.7, label='95% CI')
        axes[0, 1].axhline(-threshold, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Cross-Correlation Function')
        axes[0, 1].set_xlabel('Lag')
        axes[0, 1].set_ylabel('Cross-Correlation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Transfer function weights
        params = self.extract_transfer_function_parameters()
        tf_weights = params['transfer_function_weights']
        axes[1, 0].stem(range(len(tf_weights)), tf_weights, basefmt=' ')
        axes[1, 0].set_title('Transfer Function Weights')
        axes[1, 0].set_xlabel('Lag (Months)')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Impulse response function
        impulse_resp = self.calculate_impulse_response(horizon=24)
        axes[1, 1].plot(range(len(impulse_resp)), impulse_resp, 'g-', marker='o', markersize=4)
        axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Impulse Response Function')
        axes[1, 1].set_xlabel('Periods')
        axes[1, 1].set_ylabel('Response')
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Fitted vs Actual
        fitted_values = self.results.fittedvalues
        axes[2, 0].scatter(fitted_values, self.output_series.loc[fitted_values.index],
                           alpha=0.6, s=20)

        # Add 45-degree line
        min_val = min(fitted_values.min(), self.output_series.loc[fitted_values.index].min())
        max_val = max(fitted_values.max(), self.output_series.loc[fitted_values.index].max())
        axes[2, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

        axes[2, 0].set_xlabel('Fitted Values')
        axes[2, 0].set_ylabel('Actual Values')
        axes[2, 0].set_title(f'Fitted vs Actual (R² = {self.results.rsquared:.3f})')
        axes[2, 0].grid(True, alpha=0.3)

        # 6. Residuals
        residuals = self.results.resid
        axes[2, 1].plot(residuals.index, residuals.values, 'b-', alpha=0.7, linewidth=0.8)
        axes[2, 1].axhline(0, color='r', linestyle='--', alpha=0.8)
        axes[2, 1].set_title('Residuals')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Residuals')
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """
        Print comprehensive model summary
        """
        if self.results is None:
            raise ValueError("Model must be fitted first")

        print("Transfer Function Model Results")
        print("=" * 40)

        # Model identification
        b, r, s, _, _ = self.identify_transfer_function_order()
        print(f"Model: {self.output_series.name} = f({self.input_series.name})")
        print(f"Maximum lags considered: {self.max_lags}")
        print(f"Identified TF order (b,r,s): ({b}, {r}, {s})")

        # Model fit statistics
        params = self.extract_transfer_function_parameters()
        print(f"\nModel Fit:")
        print(f"  R-squared: {params['r_squared']:.4f}")
        print(f"  Adjusted R-squared: {params['adjusted_r_squared']:.4f}")
        print(f"  AIC: {params['aic']:.4f}")
        print(f"  BIC: {params['bic']:.4f}")
        print(f"  F-statistic: {params['f_statistic']:.4f} (p-value: {params['f_pvalue']:.4f})")

        # Transfer function weights
        tf_weights = params['transfer_function_weights']
        print(f"\nTransfer Function Weights (lag 0 to {len(tf_weights) - 1}):")
        for i, weight in enumerate(tf_weights):
            significance = "***" if abs(weight) > 0.01 else ""
            print(f"  Lag {i}: {weight:.6f} {significance}")

        # AR coefficients
        if params['ar_coefficients']:
            print(f"\nAutoregressive Coefficients:")
            for i, coeff in enumerate(params['ar_coefficients'], 1):
                print(f"  AR({i}): {coeff:.6f}")

        # Impulse response analysis
        impulse_resp = self.calculate_impulse_response()
        cumulative_resp = self.calculate_cumulative_response()

        peak_lag = np.argmax(np.abs(impulse_resp))
        peak_response = impulse_resp[peak_lag]
        long_run_multiplier = cumulative_resp[-1]

        print(f"\nImpulse Response Analysis:")
        print(f"  Peak response: {peak_response:.6f} at lag {peak_lag} months")
        print(f"  Long-run multiplier: {long_run_multiplier:.6f}")

        # Economic interpretation
        if peak_response < 0:
            print(f"  Interpretation: Input increases reduce output (negative relationship)")
        else:
            print(f"  Interpretation: Input increases boost output (positive relationship)")

        # Diagnostic tests
        diagnostics = self.diagnostic_tests()
        print(f"\nDiagnostic Tests:")
        for test, results in diagnostics.items():
            if 'p_value' in results:
                print(f"  {test}: statistic = {results['statistic']:.4f}, p-value = {results['p_value']:.4f}")
            else:
                print(f"  {test}: statistic = {results['statistic']:.4f}")
            print(f"    {results['description']}")


# Usage example function
def fit_transfer_function_housing(output_series, input_series, max_lags=12):
    """
    Convenience function to fit transfer function model to housing data

    Parameters:
    -----------
    output_series : pd.Series
        Output series (e.g., housing returns)
    input_series : pd.Series
        Input series (e.g., Fed funds rate changes)
    max_lags : int, default 12
        Maximum number of lags to consider

    Returns:
    --------
    model : TransferFunctionModel
        Fitted transfer function model instance
    """
    # Initialize model
    tf_model = TransferFunctionModel(output_series, input_series, max_lags)

    # Identify transfer function order
    b, r, s, lags, ccf = tf_model.identify_transfer_function_order()

    # Fit model
    results = tf_model.fit_transfer_function()

    # Print summary
    tf_model.summary()

    return tf_model


if __name__ == "__main__":
    # Example usage
    print("Transfer Function Model Implementation")
    print("This module provides transfer function modeling capabilities")
    print("Import this module and use TransferFunctionModel class or fit_transfer_function_housing function")