"""
Historical Volatility Model with Housing Data Processor Integration
Simple rolling volatility baseline for comparison with GJR-GARCH model
276 observations (2000-01 to 2024-12)
Creates individual plots using basic line graphs and histograms
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class HistoricalVolatilityHousingModel:
    """
    Historical Volatility Model for housing returns
    Simple rolling standard deviation baseline vs sophisticated GJR-GARCH

    Comparison:
    - Historical Vol: Simple rolling standard deviation
    - GJR-GARCH: Conditional volatility with 98.29% persistence
    - Historical Vol: Assumes constant volatility within windows
    - GJR-GARCH: Time-varying, asymmetric volatility clustering
    """

    def __init__(self, target_variable='shiller_return', window_sizes=None):
        """
        Initialize Historical Volatility Model

        Parameters:
        -----------
        target_variable : str, default 'shiller_return'
            Target variable for volatility analysis
        window_sizes : list, optional
            Rolling window sizes for volatility calculation (in months)
        """
        self.target_variable = target_variable
        self.window_sizes = window_sizes or [3, 6, 12, 24]  # Multiple windows

        # Load and prepare data
        self.processor = HousingDataProcessor()
        self.data = self._load_data()

        # Extract return series
        self.returns = self.data[target_variable].dropna()

        # Results storage
        self.volatility_series = {}
        self.volatility_stats = {}
        self.volatility_regimes = {}

        print(f"Historical Volatility Model initialized:")
        print(f"- Target variable: {target_variable}")
        print(f"- Window sizes: {self.window_sizes} months")
        print(f"- Sample size: {len(self.returns)} observations")
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

    def calculate_rolling_volatility(self):
        """Calculate rolling volatility for different window sizes"""
        print("Calculating rolling volatility...")

        for window in self.window_sizes:
            # Rolling standard deviation
            rolling_vol = self.returns.rolling(window=window, min_periods=max(3, window // 2)).std()

            # Annualized volatility (assuming monthly data)
            annualized_vol = rolling_vol * np.sqrt(12)

            self.volatility_series[f'{window}m'] = {
                'monthly': rolling_vol,
                'annualized': annualized_vol,
                'window_size': window
            }

        print(f"Rolling volatility calculated for {len(self.window_sizes)} window sizes")
        return self.volatility_series

    def calculate_volatility_statistics(self):
        """Calculate comprehensive volatility statistics"""
        print("Calculating volatility statistics...")

        for window_key, vol_data in self.volatility_series.items():
            monthly_vol = vol_data['monthly'].dropna()
            annualized_vol = vol_data['annualized'].dropna()

            self.volatility_stats[window_key] = {
                'mean_monthly': monthly_vol.mean(),
                'std_monthly': monthly_vol.std(),
                'min_monthly': monthly_vol.min(),
                'max_monthly': monthly_vol.max(),
                'mean_annualized': annualized_vol.mean(),
                'std_annualized': annualized_vol.std(),
                'min_annualized': annualized_vol.min(),
                'max_annualized': annualized_vol.max(),
                'volatility_of_volatility': monthly_vol.std(),  # Second moment
                'skewness': monthly_vol.skew(),
                'kurtosis': monthly_vol.kurtosis(),
                'autocorr_lag1': monthly_vol.autocorr(lag=1),
                'percentile_25': monthly_vol.quantile(0.25),
                'percentile_75': monthly_vol.quantile(0.75),
                'n_observations': len(monthly_vol)
            }

        return self.volatility_stats

    def identify_volatility_regimes(self, window='12m'):
        """Identify high/low volatility regimes (simple threshold approach)"""
        print(f"Identifying volatility regimes using {window} window...")

        if window not in self.volatility_series:
            raise ValueError(f"Window {window} not calculated")

        monthly_vol = self.volatility_series[window]['monthly'].dropna()

        # Define regimes based on median/percentiles
        median_vol = monthly_vol.median()
        q25_vol = monthly_vol.quantile(0.25)
        q75_vol = monthly_vol.quantile(0.75)

        # Create regime classification
        regime_series = pd.Series(index=monthly_vol.index, dtype=str)

        regime_series[monthly_vol <= q25_vol] = 'Low Volatility'
        regime_series[(monthly_vol > q25_vol) & (monthly_vol <= q75_vol)] = 'Normal Volatility'
        regime_series[monthly_vol > q75_vol] = 'High Volatility'

        # Calculate regime statistics
        regime_stats = {}
        for regime_name in ['Low Volatility', 'Normal Volatility', 'High Volatility']:
            regime_mask = regime_series == regime_name

            if regime_mask.any():
                # Align indices properly
                common_index = regime_mask.index.intersection(self.returns.index)
                aligned_regime_mask = regime_mask.loc[common_index]
                aligned_returns = self.returns.loc[common_index]
                aligned_vol = monthly_vol.loc[common_index]

                # Filter by regime
                regime_returns = aligned_returns[aligned_regime_mask]
                regime_vol = aligned_vol[aligned_regime_mask]

                if len(regime_returns) > 0:
                    regime_stats[regime_name] = {
                        'n_periods': len(regime_returns),
                        'frequency': len(regime_returns) / len(aligned_returns),
                        'mean_return': regime_returns.mean(),
                        'mean_volatility': regime_vol.mean(),
                        'return_volatility': regime_returns.std(),
                        'min_volatility': regime_vol.min(),
                        'max_volatility': regime_vol.max()
                    }

        self.volatility_regimes = {
            'regime_series': regime_series,
            'regime_stats': regime_stats,
            'thresholds': {
                'q25': q25_vol,
                'median': median_vol,
                'q75': q75_vol
            }
        }

        return self.volatility_regimes

    def compare_with_returns(self):
        """Analyze relationship between volatility and returns"""
        print("Analyzing volatility-return relationship...")

        comparison_stats = {}

        # Use 12-month window for comparison
        if '12m' in self.volatility_series:
            vol_12m = self.volatility_series['12m']['monthly'].dropna()

            # Align volatility and returns
            common_index = vol_12m.index.intersection(self.returns.index)
            aligned_vol = vol_12m.loc[common_index]
            aligned_returns = self.returns.loc[common_index]

            # Calculate correlations
            vol_return_corr = aligned_vol.corr(aligned_returns)
            vol_return_corr_abs = aligned_vol.corr(aligned_returns.abs())

            # Volatility clustering test (simple)
            vol_changes = aligned_vol.diff().dropna()
            vol_clustering = vol_changes.autocorr(lag=1)

            comparison_stats = {
                'volatility_return_correlation': vol_return_corr,
                'volatility_abs_return_correlation': vol_return_corr_abs,
                'volatility_clustering_lag1': vol_clustering,
                'mean_volatility': aligned_vol.mean(),
                'volatility_persistence': aligned_vol.autocorr(lag=1)
            }

        return comparison_stats

    def plot_volatility_evolution(self, figsize=(15, 10)):
        """Plot 1: Volatility evolution over time"""
        if not self.volatility_series:
            self.calculate_rolling_volatility()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        colors = ['blue', 'red', 'green', 'orange']

        for i, (window_key, vol_data) in enumerate(self.volatility_series.items()):
            if i < len(axes):
                ax = axes[i]

                # Plot annualized volatility
                annualized_vol = vol_data['annualized']
                window_size = vol_data['window_size']

                ax.plot(annualized_vol.index, annualized_vol.values,
                        color=colors[i], linewidth=2, alpha=0.8,
                        label=f'{window_size}-Month Rolling Volatility')

                # Add mean line
                mean_vol = annualized_vol.mean()
                ax.axhline(mean_vol, color=colors[i], linestyle='--', alpha=0.7,
                           label=f'Mean: {mean_vol:.2%}')

                # Add economic crisis periods
                crisis_periods = [
                    ('2007-12', '2009-06', 'Great Recession'),
                    ('2020-02', '2020-04', 'COVID Crisis')
                ]

                for start, end, label in crisis_periods:
                    try:
                        start_date = pd.to_datetime(start)
                        end_date = pd.to_datetime(end)
                        ax.axvspan(start_date, end_date, alpha=0.3, color='gray',
                                   label=label if i == 0 and start == '2007-12' else "")
                    except:
                        pass

                ax.set_title(f'{window_size}-Month Rolling Volatility (Annualized)',
                             fontsize=12, fontweight='bold')
                ax.set_ylabel('Volatility')
                if i >= 2:
                    ax.set_xlabel('Date')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Format y-axis as percentage
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        plt.tight_layout()
        return fig

    def plot_volatility_distributions(self, figsize=(14, 10)):
        """Plot 2: Volatility distributions and statistics"""
        if not self.volatility_stats:
            self.calculate_volatility_statistics()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Volatility distributions for different windows
        colors = ['blue', 'red', 'green', 'orange']

        for i, (window_key, vol_data) in enumerate(self.volatility_series.items()):
            monthly_vol = vol_data['monthly'].dropna()
            window_size = vol_data['window_size']

            ax1.hist(monthly_vol, bins=20, alpha=0.6, color=colors[i],
                     label=f'{window_size}M Window', density=True)

        ax1.set_title('Volatility Distributions\n(Monthly, Different Windows)',
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('Monthly Volatility')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Volatility statistics comparison
        window_names = [f"{vol_data['window_size']}M" for vol_data in self.volatility_series.values()]
        mean_vols = [stats['mean_annualized'] for stats in self.volatility_stats.values()]
        std_vols = [stats['std_annualized'] for stats in self.volatility_stats.values()]

        x = np.arange(len(window_names))
        width = 0.35

        bars1 = ax2.bar(x - width / 2, mean_vols, width, label='Mean Volatility',
                        color='steelblue', alpha=0.7)
        bars2 = ax2.bar(x + width / 2, std_vols, width, label='Volatility of Volatility',
                        color='lightcoral', alpha=0.7)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}', ha='center', va='bottom')

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1%}', ha='center', va='bottom')

        ax2.set_title('Volatility Statistics by Window Size', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('Annualized Volatility')
        ax2.set_xticks(x)
        ax2.set_xticklabels(window_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # 3. Volatility clustering analysis
        if '12m' in self.volatility_series:
            vol_12m = self.volatility_series['12m']['monthly'].dropna()

            # Lag plot for volatility clustering
            vol_lagged = vol_12m.shift(1).dropna()
            vol_current = vol_12m.loc[vol_lagged.index]

            ax3.scatter(vol_lagged, vol_current, alpha=0.6, s=20)

            # Add correlation info
            correlation = vol_lagged.corr(vol_current)
            ax3.text(0.05, 0.95, f'Autocorr(1): {correlation:.3f}',
                     transform=ax3.transAxes, fontsize=11,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax3.set_title('Volatility Clustering\n(12M Window)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Volatility (t-1)')
            ax3.set_ylabel('Volatility (t)')
            ax3.grid(True, alpha=0.3)

        # 4. Model comparison summary
        ax4.axis('off')

        comparison_stats = self.compare_with_returns()

        summary_text = "Volatility Model Comparison\n"
        summary_text += "=" * 30 + "\n\n"
        summary_text += "Historical Volatility Model:\n"
        summary_text += f"• Mean Vol (12M): {self.volatility_stats['12m']['mean_annualized']:.2%}\n"
        summary_text += f"• Vol of Vol: {self.volatility_stats['12m']['volatility_of_volatility']:.4f}\n"
        summary_text += f"• Persistence: {self.volatility_stats['12m']['autocorr_lag1']:.3f}\n"
        if comparison_stats:
            summary_text += f"• Clustering: {comparison_stats['volatility_clustering_lag1']:.3f}\n"
        summary_text += "\n"

        summary_text += "vs. GJR-GARCH Model:\n"
        summary_text += "• Conditional volatility\n"
        summary_text += "• Persistence: 0.9829 (98.29%)\n"
        summary_text += "• Asymmetric effects\n"
        summary_text += "• Time-varying parameters\n\n"

        summary_text += "Historical Vol Limitations:\n"
        summary_text += "• Assumes constant vol in window\n"
        summary_text += "• No volatility clustering\n"
        summary_text += "• No asymmetric effects\n"
        summary_text += "• Backward-looking only\n\n"

        summary_text += "GJR-GARCH Advantages:\n"
        summary_text += "• Forward-looking forecasts\n"
        summary_text += "• Captures vol clustering\n"
        summary_text += "• Leverage effects\n"
        summary_text += "• Higher persistence"

        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_volatility_regimes(self, figsize=(14, 8)):
        """Plot 3: Volatility regime analysis"""
        if not self.volatility_regimes:
            self.identify_volatility_regimes()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # 1. Time series with regime highlighting
        vol_12m = self.volatility_series['12m']['annualized']
        regime_series = self.volatility_regimes['regime_series']

        # Plot volatility
        ax1.plot(vol_12m.index, vol_12m.values, color='black', linewidth=1.5, alpha=0.8)

        # Color background by regime
        regime_colors = {
            'Low Volatility': 'lightgreen',
            'Normal Volatility': 'lightblue',
            'High Volatility': 'lightcoral'
        }

        for regime_name, color in regime_colors.items():
            regime_mask = regime_series == regime_name
            if regime_mask.any():
                ax1.fill_between(vol_12m.index, 0, vol_12m.values,
                                 where=regime_mask.reindex(vol_12m.index, fill_value=False),
                                 alpha=0.3, color=color, label=regime_name)

        # Add threshold lines
        thresholds = self.volatility_regimes['thresholds']
        ax1.axhline(thresholds['q25'] * np.sqrt(12), color='green', linestyle='--',
                    alpha=0.7, label='25th Percentile')
        ax1.axhline(thresholds['q75'] * np.sqrt(12), color='red', linestyle='--',
                    alpha=0.7, label='75th Percentile')

        ax1.set_title('Volatility Regimes\n(12-Month Rolling, Annualized)',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Volatility')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

        # 2. Regime statistics
        regime_stats = self.volatility_regimes['regime_stats']

        regimes = list(regime_stats.keys())
        frequencies = [stats['frequency'] * 100 for stats in regime_stats.values()]
        mean_returns = [stats['mean_return'] for stats in regime_stats.values()]
        mean_vols = [stats['mean_volatility'] * np.sqrt(12) for stats in regime_stats.values()]

        x = np.arange(len(regimes))
        width = 0.25

        # Frequency bars
        bars1 = ax2.bar(x - width, frequencies, width, label='Frequency (%)',
                        color='steelblue', alpha=0.7)

        # Mean return bars (scaled for visualization)
        mean_returns_scaled = [ret * 1000 for ret in mean_returns]  # Scale by 1000
        bars2 = ax2.bar(x, mean_returns_scaled, width, label='Mean Return (×1000)',
                        color='green', alpha=0.7)

        # Mean volatility bars (as percentage)
        mean_vols_pct = [vol * 100 for vol in mean_vols]
        bars3 = ax2.bar(x + width, mean_vols_pct, width, label='Mean Vol (%)',
                        color='red', alpha=0.7)

        # Add value labels
        for bars, values in [(bars1, frequencies), (bars2, mean_returns_scaled),
                             (bars3, mean_vols_pct)]:
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{value:.1f}', ha='center', va='bottom', fontsize=9)

        ax2.set_title('Regime Statistics Comparison', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Volatility Regime')
        ax2.set_ylabel('Value')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regimes, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_all_results(self, show_plots=True, save_plots=False, save_path='./'):
        """Generate all plots for historical volatility analysis"""
        plots = {}

        print("Generating Historical Volatility analysis plots...")

        # Calculate all volatility measures
        self.calculate_rolling_volatility()
        self.calculate_volatility_statistics()
        self.identify_volatility_regimes()

        plots['volatility_evolution'] = self.plot_volatility_evolution()
        print("✓ Plot 1: Volatility Evolution Over Time")

        plots['volatility_distributions'] = self.plot_volatility_distributions()
        print("✓ Plot 2: Volatility Distributions and Statistics")

        plots['volatility_regimes'] = self.plot_volatility_regimes()
        print("✓ Plot 3: Volatility Regime Analysis")

        # Save plots if requested
        if save_plots:
            for plot_name, fig in plots.items():
                filename = f"{save_path}historical_volatility_{plot_name}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")

        # Show plots if requested
        if show_plots:
            plt.show()

        return plots

    def summary(self):
        """Print volatility analysis summary"""
        print("Historical Volatility Model Results")
        print("=" * 40)

        # Volatility statistics
        if self.volatility_stats:
            print("Volatility Statistics (12-month window):")
            stats_12m = self.volatility_stats['12m']
            print(f"  Mean Annualized Vol: {stats_12m['mean_annualized']:.2%}")
            print(f"  Volatility of Volatility: {stats_12m['volatility_of_volatility']:.4f}")
            print(f"  Autocorrelation (lag 1): {stats_12m['autocorr_lag1']:.3f}")
            print(f"  Range: [{stats_12m['min_annualized']:.2%}, {stats_12m['max_annualized']:.2%}]")

        # Regime analysis
        if self.volatility_regimes:
            print(f"\nVolatility Regime Analysis:")
            regime_stats = self.volatility_regimes['regime_stats']
            for regime_name, stats in regime_stats.items():
                print(f"  {regime_name}:")
                print(f"    Frequency: {stats['frequency']:.1%}")
                print(f"    Mean Return: {stats['mean_return']:.4f}")
                print(f"    Mean Volatility: {stats['mean_volatility'] * np.sqrt(12):.2%} (ann.)")

        # Comparison
        comparison_stats = self.compare_with_returns()
        if comparison_stats:
            print(f"\nComparison with Returns:")
            print(f"  Vol-Return Correlation: {comparison_stats['volatility_return_correlation']:.3f}")
            print(f"  Volatility Persistence: {comparison_stats['volatility_persistence']:.3f}")

    def get_model_results(self):
        """Return model results for external use"""
        return {
            'model_object': self,
            'volatility_series': self.volatility_series,
            'volatility_stats': self.volatility_stats,
            'volatility_regimes': self.volatility_regimes,
            'comparison_stats': self.compare_with_returns(),
            'data': self.data,
            'target_variable': self.target_variable
        }


# Convenience function
def analyze_historical_volatility_housing(target_variable='shiller_return', window_sizes=None):
    """
    Convenience function for historical volatility analysis

    Parameters:
    -----------
    target_variable : str, default 'shiller_return'
        Target variable for analysis
    window_sizes : list, optional
        Rolling window sizes for volatility calculation

    Returns:
    --------
    model : HistoricalVolatilityHousingModel
        Historical volatility analysis model
    """
    print("Running Historical Volatility Analysis...")

    # Initialize and run analysis
    model = HistoricalVolatilityHousingModel(
        target_variable=target_variable,
        window_sizes=window_sizes
    )

    # Calculate all volatility measures
    model.calculate_rolling_volatility()
    model.calculate_volatility_statistics()
    model.identify_volatility_regimes()

    # Print summary
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage
    print("Historical Volatility Model - Housing Market")
    print("Simple rolling volatility baseline vs GJR-GARCH")

    model = analyze_historical_volatility_housing()

    # Generate plots
    print("\nGenerating analysis plots...")
    plots = model.plot_all_results(show_plots=True, save_plots=False)
    print(f"\nGenerated {len(plots)} plots:")
    print("1. Volatility Evolution Over Time")
    print("2. Volatility Distributions and Statistics")
    print("3. Volatility Regime Analysis")

    print("\nHistorical Volatility analysis complete!")