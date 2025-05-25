"""
Rolling Correlation Analysis with Housing Data Processor Integration
Time-varying correlation analysis for comparison with sophisticated models
276 observations (2000-01 to 2024-12)
Creates individual plots using basic line graphs and heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import warnings

# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class RollingCorrelationHousingModel:
    """
    Rolling Correlation Analysis for housing returns and economic variables
    Simple time-varying correlation baseline vs sophisticated regime switching

    Shows dynamic relationships without complex econometric modeling:
    - Rolling correlations over time
    - Cross-correlations at different lags
    - Correlation breakdowns by economic periods
    """

    def __init__(self, target_variable='shiller_return', variables=None, window_sizes=None):
        """
        Initialize Rolling Correlation Analysis

        Parameters:
        -----------
        target_variable : str, default 'shiller_return'
            Target variable for correlation analysis
        variables : list, optional
            Variables to analyze correlations with
        window_sizes : list, optional
            Rolling window sizes for correlation calculation
        """
        self.target_variable = target_variable
        self.variables = variables or ['fed_change', 'fed_rate', 'fed_vol', 'zillow_return']
        self.window_sizes = window_sizes or [6, 12, 24]  # 6, 12, 24 month windows

        # Load and prepare data
        self.processor = HousingDataProcessor()
        self.data = self._load_data()

        # Results storage
        self.rolling_correlations = {}
        self.static_correlations = {}
        self.cross_correlations = {}
        self.correlation_breakdowns = {}

        print(f"Rolling Correlation Analysis initialized:")
        print(f"- Target variable: {target_variable}")
        print(f"- Variables for correlation: {self.variables}")
        print(f"- Window sizes: {self.window_sizes} months")
        print(f"- Sample size: {len(self.data)} observations")
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

    def calculate_rolling_correlations(self):
        """Calculate rolling correlations for different window sizes"""
        print("Calculating rolling correlations...")

        target_series = self.data[self.target_variable]

        for window in self.window_sizes:
            self.rolling_correlations[f'{window}m'] = {}

            for var in self.variables:
                if var in self.data.columns:
                    # Calculate rolling correlation
                    rolling_corr = target_series.rolling(window=window).corr(self.data[var])
                    self.rolling_correlations[f'{window}m'][var] = rolling_corr

        print(f"Rolling correlations calculated for {len(self.variables)} variables")
        return self.rolling_correlations

    def calculate_static_correlations(self):
        """Calculate static (full-sample) correlations"""
        print("Calculating static correlations...")

        target_series = self.data[self.target_variable]

        for var in self.variables:
            if var in self.data.columns:
                # Pearson correlation
                corr_coef, p_value = pearsonr(target_series.dropna(),
                                              self.data[var].loc[target_series.dropna().index])

                self.static_correlations[var] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }

        return self.static_correlations

    def calculate_cross_correlations(self, max_lags=12):
        """Calculate cross-correlations at different lags"""
        print(f"Calculating cross-correlations up to {max_lags} lags...")

        target_series = self.data[self.target_variable].dropna()

        for var in self.variables:
            if var in self.data.columns:
                var_series = self.data[var].dropna()

                # Align series
                common_index = target_series.index.intersection(var_series.index)
                target_aligned = target_series.loc[common_index]
                var_aligned = var_series.loc[common_index]

                # Calculate cross-correlations
                lags = range(-max_lags, max_lags + 1)
                cross_corrs = []

                for lag in lags:
                    if lag == 0:
                        corr = target_aligned.corr(var_aligned)
                    elif lag > 0:
                        # Target leads variable
                        corr = target_aligned.iloc[lag:].corr(var_aligned.iloc[:-lag])
                    else:
                        # Variable leads target
                        corr = target_aligned.iloc[:lag].corr(var_aligned.iloc[-lag:])

                    cross_corrs.append(corr)

                self.cross_correlations[var] = {
                    'lags': list(lags),
                    'correlations': cross_corrs
                }

        return self.cross_correlations

    def analyze_correlation_by_periods(self):
        """Analyze correlations during different economic periods"""
        print("Analyzing correlations by economic periods...")

        # Define economic periods
        periods = {
            'Pre-Crisis (2001-2007)': ('2001-01', '2007-11'),
            'Financial Crisis (2007-2009)': ('2007-12', '2009-06'),
            'Recovery (2009-2015)': ('2009-07', '2015-12'),
            'Modern Era (2016-2024)': ('2016-01', '2024-01')
        }

        target_series = self.data[self.target_variable]

        for period_name, (start, end) in periods.items():
            try:
                start_date = pd.to_datetime(start)
                end_date = pd.to_datetime(end)

                # Filter data for period
                mask = (self.data.index >= start_date) & (self.data.index <= end_date)
                period_data = self.data[mask]

                if len(period_data) > 10:  # Ensure sufficient data
                    period_correlations = {}

                    for var in self.variables:
                        if var in period_data.columns:
                            period_target = period_data[self.target_variable].dropna()
                            period_var = period_data[var].dropna()

                            if len(period_target) > 5 and len(period_var) > 5:
                                # Align series
                                common_idx = period_target.index.intersection(period_var.index)
                                if len(common_idx) > 5:
                                    corr_coef, p_value = pearsonr(period_target.loc[common_idx],
                                                                  period_var.loc[common_idx])

                                    period_correlations[var] = {
                                        'correlation': corr_coef,
                                        'p_value': p_value,
                                        'n_obs': len(common_idx)
                                    }

                    self.correlation_breakdowns[period_name] = period_correlations

            except Exception as e:
                print(f"Warning: Could not analyze period {period_name}: {e}")

        return self.correlation_breakdowns

    def plot_rolling_correlations(self, figsize=(15, 10)):
        """Plot 1: Rolling correlations over time"""
        if not self.rolling_correlations:
            self.calculate_rolling_correlations()

        fig, axes = plt.subplots(len(self.window_sizes), 1, figsize=figsize)
        if len(self.window_sizes) == 1:
            axes = [axes]

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        for i, window in enumerate(self.window_sizes):
            ax = axes[i]

            for j, var in enumerate(self.variables):
                if var in self.rolling_correlations[f'{window}m']:
                    rolling_corr = self.rolling_correlations[f'{window}m'][var]

                    ax.plot(rolling_corr.index, rolling_corr.values,
                            label=var.replace('_', ' ').title(),
                            color=colors[j % len(colors)], linewidth=2, alpha=0.8)

            # Add reference lines
            ax.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='±0.5')
            ax.axhline(-0.5, color='gray', linestyle='--', alpha=0.5)

            # Add economic crisis periods
            crisis_periods = [
                ('2007-12', '2009-06', 'Great Recession'),
                ('2020-02', '2020-04', 'COVID Crisis')
            ]

            for start, end, label in crisis_periods:
                try:
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)
                    ax.axvspan(start_date, end_date, alpha=0.2, color='red',
                               label=label if i == 0 and start == '2007-12' else "")
                except:
                    pass

            ax.set_title(f'{window}-Month Rolling Correlations with {self.target_variable.replace("_", " ").title()}',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Correlation Coefficient')
            if i == len(self.window_sizes) - 1:
                ax.set_xlabel('Date')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)

        plt.tight_layout()
        return fig

    def plot_correlation_heatmap(self, figsize=(12, 8)):
        """Plot 2: Correlation heatmap and static analysis"""
        if not self.static_correlations:
            self.calculate_static_correlations()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Full correlation matrix heatmap
        corr_vars = [self.target_variable] + [var for var in self.variables if var in self.data.columns]
        correlation_matrix = self.data[corr_vars].corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                    center=0, square=True, fmt='.3f', cbar_kws={'shrink': 0.8},
                    ax=ax1)
        ax1.set_title('Correlation Matrix\n(Full Sample)', fontsize=14, fontweight='bold')

        # 2. Static correlations with target variable
        target_correlations = {}
        for var in self.variables:
            if var in self.static_correlations:
                target_correlations[var] = self.static_correlations[var]['correlation']

        if target_correlations:
            vars_sorted = sorted(target_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            var_names = [x[0].replace('_', ' ').title() for x in vars_sorted]
            var_corrs = [x[1] for x in vars_sorted]

            colors = ['darkred' if corr < 0 else 'darkblue' for corr in var_corrs]

            bars = ax2.barh(var_names, var_corrs, color=colors, alpha=0.7)

            # Add value labels
            for bar, corr in zip(bars, var_corrs):
                width = bar.get_width()
                ax2.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                         f'{corr:.3f}', ha='left' if width >= 0 else 'right', va='center',
                         fontweight='bold')

            ax2.set_xlabel('Correlation with Housing Returns')
            ax2.set_title(f'Static Correlations\nwith {self.target_variable.replace("_", " ").title()}',
                          fontsize=14, fontweight='bold')
            ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.set_xlim(-1, 1)

        plt.tight_layout()
        return fig

    def plot_cross_correlations(self, figsize=(15, 10)):
        """Plot 3: Cross-correlation analysis"""
        if not self.cross_correlations:
            self.calculate_cross_correlations()

        n_vars = len([var for var in self.variables if var in self.cross_correlations])
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        colors = ['b', 'r', 'g', 'orange']  # Use single char codes where possible

        for i, var in enumerate(self.variables[:4]):  # Show first 4 variables
            if var in self.cross_correlations and i < len(axes):
                ax = axes[i]

                lags = self.cross_correlations[var]['lags']
                correlations = self.cross_correlations[var]['correlations']

                # Stem plot for cross-correlations - use simple approach
                if colors[i] == 'orange':
                    # For orange, use default and set color after
                    markerline, stemlines, baseline = ax.stem(lags, correlations, basefmt=' ')
                    markerline.set_color('orange')
                    stemlines.set_color('orange')  # stemlines is a LineCollection, set color directly
                else:
                    # Use single character codes
                    ax.stem(lags, correlations, linefmt=f'{colors[i]}-',
                            markerfmt=f'{colors[i]}o', basefmt=' ')

                # Add significance threshold
                threshold = 2 / np.sqrt(len(self.data))
                ax.axhline(threshold, color='red', linestyle='--', alpha=0.7,
                           label=f'95% CI (±{threshold:.3f})')
                ax.axhline(-threshold, color='red', linestyle='--', alpha=0.7)
                ax.axhline(0, color='black', linestyle='-', alpha=0.5)

                # Highlight peak correlation
                max_idx = np.argmax(np.abs(correlations))
                peak_lag = lags[max_idx]
                peak_corr = correlations[max_idx]

                ax.scatter(peak_lag, peak_corr, color='red', s=100, zorder=5,
                           label=f'Peak: lag {peak_lag} ({peak_corr:.3f})')

                ax.set_title(f'Cross-Correlation: {var.replace("_", " ").title()}',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('Lag (Months)')
                ax.set_ylabel('Cross-Correlation')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1, 1)

        # Hide unused subplots
        for i in range(len(self.variables), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_period_analysis(self, figsize=(14, 8)):
        """Plot 4: Correlation analysis by economic periods"""
        if not self.correlation_breakdowns:
            self.analyze_correlation_by_periods()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Correlation evolution by periods
        periods = list(self.correlation_breakdowns.keys())
        vars_to_plot = self.variables[:3]  # Plot first 3 variables for clarity

        x = np.arange(len(periods))
        width = 0.25
        colors = ['blue', 'red', 'green']

        for i, var in enumerate(vars_to_plot):
            correlations = []
            for period in periods:
                if var in self.correlation_breakdowns[period]:
                    correlations.append(self.correlation_breakdowns[period][var]['correlation'])
                else:
                    correlations.append(0)

            ax1.bar(x + i * width, correlations, width, label=var.replace('_', ' ').title(),
                    color=colors[i], alpha=0.7)

        ax1.set_xlabel('Economic Period')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.set_title('Correlation Evolution Across Economic Periods',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(periods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)

        # 2. Period comparison summary
        ax2.axis('off')

        summary_text = "Correlation Analysis Summary\n"
        summary_text += "=" * 30 + "\n\n"

        # Overall statistics
        if self.static_correlations:
            summary_text += "Full Sample Correlations:\n"
            for var, stats in list(self.static_correlations.items())[:4]:
                summary_text += f"• {var.replace('_', ' ').title()}: {stats['correlation']:.3f}"
                if stats['significant']:
                    summary_text += " ***\n"
                else:
                    summary_text += "\n"
            summary_text += "\n"

        # Period with highest correlations
        if self.correlation_breakdowns:
            summary_text += "Strongest Correlations by Period:\n"
            for period in periods:
                if period in self.correlation_breakdowns:
                    max_corr = 0
                    max_var = ""
                    for var, stats in self.correlation_breakdowns[period].items():
                        if abs(stats['correlation']) > abs(max_corr):
                            max_corr = stats['correlation']
                            max_var = var

                    if max_var:
                        summary_text += f"• {period}:\n"
                        summary_text += f"  {max_var.replace('_', ' ').title()}: {max_corr:.3f}\n"

        summary_text += f"\nComparison to Sophisticated Models:\n\n"
        summary_text += f"Rolling Correlation Analysis:\n"
        summary_text += f"• Shows time-varying relationships\n"
        summary_text += f"• Simple and interpretable\n"
        summary_text += f"• Assumes linear relationships\n\n"
        summary_text += f"vs. Regime Switching Model:\n"
        summary_text += f"• 2 distinct market regimes\n"
        summary_text += f"• Non-linear state transitions\n"
        summary_text += f"• Regime-specific parameters\n\n"
        summary_text += f"Note: *** indicates p < 0.05"

        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_all_results(self, show_plots=True, save_plots=False, save_path='./'):
        """Generate all plots for rolling correlation analysis"""
        plots = {}

        print("Generating Rolling Correlation analysis plots...")

        # Calculate all correlations first
        self.calculate_rolling_correlations()
        self.calculate_static_correlations()
        self.calculate_cross_correlations()
        self.analyze_correlation_by_periods()

        plots['rolling_correlations'] = self.plot_rolling_correlations()
        print("✓ Plot 1: Rolling Correlations Over Time")

        plots['correlation_heatmap'] = self.plot_correlation_heatmap()
        print("✓ Plot 2: Correlation Heatmap and Static Analysis")

        plots['cross_correlations'] = self.plot_cross_correlations()
        print("✓ Plot 3: Cross-Correlation Analysis")

        plots['period_analysis'] = self.plot_period_analysis()
        print("✓ Plot 4: Correlation Analysis by Economic Periods")

        # Save plots if requested
        if save_plots:
            for plot_name, fig in plots.items():
                filename = f"{save_path}rolling_correlation_{plot_name}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")

        # Show plots if requested
        if show_plots:
            plt.show()

        return plots

    def summary(self):
        """Print correlation analysis summary"""
        print("Rolling Correlation Analysis Results")
        print("=" * 40)

        # Static correlations
        if self.static_correlations:
            print("Full Sample Correlations:")
            for var, stats in self.static_correlations.items():
                significance = "***" if stats['significant'] else ""
                print(f"  {var}: {stats['correlation']:.4f} {significance}")

        # Rolling correlation ranges
        if self.rolling_correlations:
            print(f"\nRolling Correlation Ranges (12-month window):")
            if '12m' in self.rolling_correlations:
                for var, rolling_corr in self.rolling_correlations['12m'].items():
                    min_corr = rolling_corr.min()
                    max_corr = rolling_corr.max()
                    print(f"  {var}: [{min_corr:.4f}, {max_corr:.4f}]")

        # Period analysis
        if self.correlation_breakdowns:
            print(f"\nCorrelation Breakdown by Period:")
            for period, correlations in self.correlation_breakdowns.items():
                print(f"  {period}:")
                for var, stats in correlations.items():
                    print(f"    {var}: {stats['correlation']:.4f}")

    def get_model_results(self):
        """Return analysis results for external use"""
        return {
            'model_object': self,
            'rolling_correlations': self.rolling_correlations,
            'static_correlations': self.static_correlations,
            'cross_correlations': self.cross_correlations,
            'period_correlations': self.correlation_breakdowns,
            'data': self.data,
            'target_variable': self.target_variable
        }


# Convenience function
def analyze_rolling_correlations_housing(target_variable='shiller_return', variables=None):
    """
    Convenience function for rolling correlation analysis

    Parameters:
    -----------
    target_variable : str, default 'shiller_return'
        Target variable for analysis
    variables : list, optional
        Variables to analyze correlations with

    Returns:
    --------
    model : RollingCorrelationHousingModel
        Fitted correlation analysis model
    """
    print("Running Rolling Correlation Analysis...")

    # Initialize and run analysis
    model = RollingCorrelationHousingModel(
        target_variable=target_variable,
        variables=variables
    )

    # Calculate all correlations
    model.calculate_rolling_correlations()
    model.calculate_static_correlations()
    model.calculate_cross_correlations()
    model.analyze_correlation_by_periods()

    # Print summary
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage
    print("Rolling Correlation Analysis - Housing Market")
    print("Time-varying correlation baseline for comparison")

    model = analyze_rolling_correlations_housing()

    # Generate plots
    print("\nGenerating analysis plots...")
    plots = model.plot_all_results(show_plots=True, save_plots=False)
    print(f"\nGenerated {len(plots)} plots:")
    print("1. Rolling Correlations Over Time")
    print("2. Correlation Heatmap and Static Analysis")
    print("3. Cross-Correlation Analysis")
    print("4. Correlation Analysis by Economic Periods")

    print("\nRolling Correlation analysis complete!")