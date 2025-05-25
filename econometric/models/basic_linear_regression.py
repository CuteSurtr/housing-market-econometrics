"""
Basic Linear Regression Model with Housing Data Processor Integration
Simple OLS baseline model for comparison with sophisticated econometric models
276 observations (2000-01 to 2024-12)
Creates individual plots using basic visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
import warnings

# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class BasicLinearRegressionHousingModel:
    """
    Basic Linear Regression Model for housing returns
    Simple OLS baseline: housing_returns = α + β₁*fed_change + β₂*fed_level + ε

    This serves as a baseline comparison to sophisticated models like:
    - Transfer Function (R² = 0.8980)
    - GJR-GARCH with time-varying volatility
    - Regime Switching with multiple states
    """

    def __init__(self, target_variable='shiller_return', regressors=None):
        """
        Initialize Basic Linear Regression model

        Parameters:
        -----------
        target_variable : str, default 'shiller_return'
            Target variable (shiller_return or zillow_return)
        regressors : list, optional
            List of regressor variables
        """
        self.target_variable = target_variable
        self.regressors = regressors or ['fed_change', 'fed_rate', 'fed_vol']

        # Load and prepare data
        self.processor = HousingDataProcessor()
        self.data = self._load_data()

        # Extract variables
        self.y = self.data[target_variable].dropna()
        self.X = self._prepare_regressors()

        # Model results
        self.model = None
        self.results = None
        self.predictions = None
        self.residuals = None

        print(f"Basic Linear Regression model initialized:")
        print(f"- Target variable: {target_variable}")
        print(f"- Regressors: {self.regressors}")
        print(f"- Sample size: {len(self.y)} observations")
        print(f"- Sample period: {self.data.index.min()} to {self.data.index.max()}")

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

    def _prepare_regressors(self):
        """Prepare regressor variables"""
        available_regressors = []
        for reg in self.regressors:
            if reg in self.data.columns:
                available_regressors.append(reg)
            else:
                print(f"Warning: Regressor '{reg}' not found in data")

        if not available_regressors:
            raise ValueError("No valid regressors found")

        # Extract regressor data aligned with target
        X_data = self.data[available_regressors].loc[self.y.index]

        # Handle missing values
        X_data = X_data.fillna(method='ffill').fillna(method='bfill')

        print(f"Regressors prepared: {available_regressors}")
        return X_data

    def fit_model(self, include_constant=True):
        """
        Fit basic linear regression model using OLS

        Parameters:
        -----------
        include_constant : bool, default True
            Include intercept term
        """
        print("Fitting basic linear regression model...")

        # Align data
        common_index = self.y.index.intersection(self.X.index)
        y_aligned = self.y.loc[common_index]
        X_aligned = self.X.loc[common_index]

        # Add constant if requested
        if include_constant:
            X_aligned = sm.add_constant(X_aligned)

        # Fit OLS model using statsmodels for detailed statistics
        self.model = sm.OLS(y_aligned, X_aligned)
        self.results = self.model.fit()

        # Store predictions and residuals
        self.predictions = self.results.fittedvalues
        self.residuals = self.results.resid

        print("Basic linear regression fitted successfully!")
        print(f"R-squared: {self.results.rsquared:.4f}")
        print(f"Adjusted R-squared: {self.results.rsquared_adj:.4f}")

        return self.results

    def get_model_statistics(self):
        """Extract key model statistics"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        stats_dict = {
            'r_squared': self.results.rsquared,
            'adj_r_squared': self.results.rsquared_adj,
            'f_statistic': self.results.fvalue,
            'f_pvalue': self.results.f_pvalue,
            'aic': self.results.aic,
            'bic': self.results.bic,
            'log_likelihood': self.results.llf,
            'n_observations': self.results.nobs,
            'mse': self.results.mse_resid,
            'rmse': np.sqrt(self.results.mse_resid)
        }

        return stats_dict

    def diagnostic_tests(self):
        """Perform diagnostic tests"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        diagnostics = {}

        # Ljung-Box test for serial correlation
        try:
            lb_result = acorr_ljungbox(self.residuals, lags=10, return_df=True)
            diagnostics['ljung_box'] = {
                'statistic': lb_result['lb_stat'].iloc[-1],
                'p_value': lb_result['lb_pvalue'].iloc[-1],
                'description': 'Test for serial correlation'
            }
        except:
            diagnostics['ljung_box'] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'description': 'Ljung-Box test failed'
            }

        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(self.residuals)
            diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'description': 'Test for normality'
            }
        except:
            diagnostics['jarque_bera'] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'description': 'Jarque-Bera test failed'
            }

        # Durbin-Watson test
        try:
            dw_stat = durbin_watson(self.residuals)
            diagnostics['durbin_watson'] = {
                'statistic': dw_stat,
                'description': 'Test for autocorrelation'
            }
        except:
            diagnostics['durbin_watson'] = {
                'statistic': np.nan,
                'description': 'Durbin-Watson test failed'
            }

        # Breusch-Pagan test for heteroskedasticity
        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(self.residuals, self.results.model.exog)
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

    def plot_regression_analysis(self, figsize=(14, 10)):
        """Plot 1: Basic regression analysis"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Actual vs Fitted values
        ax1.scatter(self.predictions, self.y.loc[self.predictions.index],
                    alpha=0.6, s=30, color='blue', edgecolors='darkblue')

        # 45-degree line
        min_val = min(self.predictions.min(), self.y.loc[self.predictions.index].min())
        max_val = max(self.predictions.max(), self.y.loc[self.predictions.index].max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

        r2 = self.results.rsquared
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, fontsize=12,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Actual Values')
        ax1.set_title('Actual vs Fitted Values\nBasic Linear Regression',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. Residuals over time
        ax2.plot(self.residuals.index, self.residuals.values,
                 color='purple', alpha=0.7, linewidth=1)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Add 2-sigma bands
        resid_std = self.residuals.std()
        ax2.axhline(2 * resid_std, color='red', linestyle=':', alpha=0.7, label='±2σ')
        ax2.axhline(-2 * resid_std, color='red', linestyle=':', alpha=0.7)

        ax2.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q plot for normality
        stats.probplot(self.residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot: Residual Normality', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Residual histogram
        ax4.hist(self.residuals, bins=30, alpha=0.7, color='lightblue',
                 edgecolor='black', density=True)

        # Overlay normal distribution
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, self.residuals.mean(), self.residuals.std()),
                 'r-', linewidth=2, label='Normal Distribution')

        ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_coefficient_analysis(self, figsize=(14, 8)):
        """Plot 2: Coefficient analysis"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Coefficient values with confidence intervals
        params = self.results.params
        conf_int = self.results.conf_int()

        # Exclude constant for cleaner visualization
        if 'const' in params.index:
            params_plot = params.drop('const')
            conf_int_plot = conf_int.drop('const')
        else:
            params_plot = params
            conf_int_plot = conf_int

        y_pos = np.arange(len(params_plot))

        ax1.barh(y_pos, params_plot.values, alpha=0.7, color='steelblue')
        ax1.errorbar(params_plot.values, y_pos,
                     xerr=[params_plot.values - conf_int_plot.iloc[:, 0],
                           conf_int_plot.iloc[:, 1] - params_plot.values],
                     fmt='none', color='red', capsize=5)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(params_plot.index)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Regression Coefficients\nwith 95% Confidence Intervals',
                      fontsize=14, fontweight='bold')
        ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Add significance stars
        pvalues = self.results.pvalues
        for i, (param, pval) in enumerate(pvalues.items()):
            if param != 'const':
                idx = list(params_plot.index).index(param)
                if pval < 0.001:
                    significance = '***'
                elif pval < 0.01:
                    significance = '**'
                elif pval < 0.05:
                    significance = '*'
                else:
                    significance = ''

                ax1.text(params_plot.iloc[idx] + 0.001, idx, significance,
                         va='center', fontsize=12, fontweight='bold')

        # 2. Model summary statistics
        ax2.axis('off')

        stats_dict = self.get_model_statistics()
        diagnostics = self.diagnostic_tests()

        summary_text = "Model Summary Statistics\n"
        summary_text += "=" * 25 + "\n\n"
        summary_text += f"R-squared: {stats_dict['r_squared']:.4f}\n"
        summary_text += f"Adjusted R²: {stats_dict['adj_r_squared']:.4f}\n"
        summary_text += f"F-statistic: {stats_dict['f_statistic']:.2f}\n"
        summary_text += f"F p-value: {stats_dict['f_pvalue']:.6f}\n"
        summary_text += f"AIC: {stats_dict['aic']:.2f}\n"
        summary_text += f"BIC: {stats_dict['bic']:.2f}\n"
        summary_text += f"RMSE: {stats_dict['rmse']:.6f}\n\n"

        summary_text += "Diagnostic Tests:\n"
        summary_text += f"Durbin-Watson: {diagnostics['durbin_watson']['statistic']:.4f}\n"
        summary_text += f"Ljung-Box p-val: {diagnostics['ljung_box']['p_value']:.4f}\n"
        summary_text += f"Jarque-Bera p-val: {diagnostics['jarque_bera']['p_value']:.4f}\n"
        summary_text += f"Breusch-Pagan p-val: {diagnostics['breusch_pagan']['p_value']:.4f}\n\n"

        summary_text += "Significance: *** p<0.001\n"
        summary_text += "             ** p<0.01\n"
        summary_text += "             * p<0.05"

        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_model_comparison(self, figsize=(14, 8)):
        """Plot 3: Comparison with sophisticated models"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Time series: Actual vs Fitted
        actual_values = self.y.loc[self.predictions.index]

        ax1.plot(actual_values.index, actual_values.values,
                 'b-', label='Actual Returns', linewidth=2, alpha=0.8)
        ax1.plot(self.predictions.index, self.predictions.values,
                 'r--', label='Linear Regression Fit', linewidth=2, alpha=0.8)

        # Add economic crisis periods for context
        crisis_periods = [
            ('2007-12', '2009-06', 'Great Recession'),
            ('2020-02', '2020-04', 'COVID Crisis')
        ]

        for start, end, label in crisis_periods:
            try:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)
                ax1.axvspan(start_date, end_date, alpha=0.3, color='gray',
                            label=label if start == '2007-12' else "")
            except:
                pass

        ax1.set_title(f'Basic Linear Regression Fit\nR² = {self.results.rsquared:.4f}',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Housing Returns')
        ax1.set_xlabel('Date')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Model comparison table
        ax2.axis('off')

        comparison_text = "Model Performance Comparison\n"
        comparison_text += "=" * 30 + "\n\n"
        comparison_text += "Basic Linear Regression:\n"
        comparison_text += f"• R² = {self.results.rsquared:.4f}\n"
        comparison_text += f"• RMSE = {np.sqrt(self.results.mse_resid):.6f}\n"
        comparison_text += f"• AIC = {self.results.aic:.1f}\n\n"

        comparison_text += "vs. Sophisticated Models:\n\n"
        comparison_text += "Transfer Function Model:\n"
        comparison_text += "• R² = 0.8980 (123% better)\n"
        comparison_text += "• Complex lag structure\n"
        comparison_text += "• 10-month policy delay\n\n"

        comparison_text += "GJR-GARCH Model:\n"
        comparison_text += "• Time-varying volatility\n"
        comparison_text += "• 98.29% persistence\n"
        comparison_text += "• Asymmetric effects\n\n"

        comparison_text += "Jump-Diffusion Model:\n"
        comparison_text += "• 61 jumps identified\n"
        comparison_text += "• Extreme event modeling\n"
        comparison_text += "• Risk metrics (VaR)\n\n"

        comparison_text += "Linear regression assumes:\n"
        comparison_text += "• Constant parameters\n"
        comparison_text += "• Homoskedastic errors\n"
        comparison_text += "• No regime changes\n"
        comparison_text += "• Immediate policy effects"

        ax2.text(0.05, 0.95, comparison_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_all_results(self, show_plots=True, save_plots=False, save_path='./'):
        """Generate all plots for basic linear regression analysis"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        plots = {}

        print("Generating Basic Linear Regression analysis plots...")

        plots['regression_analysis'] = self.plot_regression_analysis()
        print("✓ Plot 1: Regression Analysis (Fit, Residuals, Diagnostics)")

        plots['coefficient_analysis'] = self.plot_coefficient_analysis()
        print("✓ Plot 2: Coefficient Analysis and Model Statistics")

        plots['model_comparison'] = self.plot_model_comparison()
        print("✓ Plot 3: Model Comparison with Sophisticated Methods")

        # Save plots if requested
        if save_plots:
            for plot_name, fig in plots.items():
                filename = f"{save_path}basic_regression_{plot_name}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")

        # Show plots if requested
        if show_plots:
            plt.show()

        return plots

    def summary(self):
        """Print model summary"""
        if self.results is None:
            raise ValueError("Model must be fitted first")

        print("Basic Linear Regression Model Results")
        print("=" * 45)

        # Model statistics
        stats_dict = self.get_model_statistics()
        print(f"R-squared: {stats_dict['r_squared']:.4f}")
        print(f"Adjusted R-squared: {stats_dict['adj_r_squared']:.4f}")
        print(f"F-statistic: {stats_dict['f_statistic']:.2f} (p-value: {stats_dict['f_pvalue']:.6f})")
        print(f"AIC: {stats_dict['aic']:.2f}")
        print(f"RMSE: {stats_dict['rmse']:.6f}")

        # Coefficients
        print(f"\nRegression Coefficients:")
        for param, coef in self.results.params.items():
            pval = self.results.pvalues[param]
            significance = ""
            if pval < 0.001:
                significance = "***"
            elif pval < 0.01:
                significance = "**"
            elif pval < 0.05:
                significance = "*"

            print(f"  {param}: {coef:.6f} {significance}")

        # Diagnostic tests
        print(f"\nDiagnostic Tests:")
        diagnostics = self.diagnostic_tests()
        for test_name, results in diagnostics.items():
            print(f"  {test_name}: {results['description']}")
            if 'p_value' in results:
                print(f"    p-value: {results['p_value']:.4f}")

    def get_model_results(self):
        """Return model results for external use"""
        return {
            'model_object': self,
            'results': self.results,
            'predictions': self.predictions,
            'residuals': self.residuals,
            'statistics': self.get_model_statistics() if self.results else None,
            'diagnostics': self.diagnostic_tests() if self.results else None,
            'data': self.data,
            'target_variable': self.target_variable
        }


# Convenience function to match interface
def fit_basic_regression_housing(target_variable='shiller_return', regressors=None):
    """
    Convenience function to fit basic linear regression model

    Parameters:
    -----------
    target_variable : str, default 'shiller_return'
        Target variable
    regressors : list, optional
        List of regressor variables

    Returns:
    --------
    model : BasicLinearRegressionHousingModel
        Fitted basic regression model
    """
    print("Fitting Basic Linear Regression model...")

    # Initialize and fit model
    model = BasicLinearRegressionHousingModel(
        target_variable=target_variable,
        regressors=regressors
    )

    # Fit model
    results = model.fit_model()

    # Print summary
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage
    print("Basic Linear Regression Model - Housing Market Analysis")
    print("Simple baseline model for comparison with sophisticated econometrics")

    model = fit_basic_regression_housing()

    # Generate plots
    if model.results is not None:
        print("\nGenerating analysis plots...")
        plots = model.plot_all_results(show_plots=True, save_plots=False)
        print(f"\nGenerated {len(plots)} plots:")
        print("1. Regression Analysis (Actual vs Fitted, Residuals, Q-Q Plot)")
        print("2. Coefficient Analysis and Model Statistics")
        print("3. Model Comparison with Sophisticated Methods")

    print("\nBasic Linear Regression analysis complete!")