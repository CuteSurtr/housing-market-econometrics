"""
Simple Moving Average Model with Housing Data Processor Integration
Basic trend analysis baseline for comparison with sophisticated models
276 observations (2000-01 to 2024-12)
Creates individual plots using basic line graphs and trend analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class SimpleMovingAverageHousingModel:
    """
    Simple Moving Average Model for housing trend analysis
    Basic trend identification baseline vs sophisticated econometric models

    Comparison:
    - Moving Averages: Simple trend smoothing
    - Transfer Function: Complex lag relationships (R² = 0.8980)
    - Moving Averages: Linear trend extrapolation
    - Sophisticated Models: Non-linear, time-varying parameters
    """

    def __init__(self, target_variable='shiller_return', ma_windows=None):
        """
        Initialize Simple Moving Average Model

        Parameters:
        -----------
        target_variable : str, default 'shiller_return'
            Target variable for trend analysis
        ma_windows : list, optional
            Moving average window sizes (in months)
        """
        self.target_variable = target_variable
        self.ma_windows = ma_windows or [3, 6, 12, 24]  # Multiple MA windows

        # Load and prepare data
        self.processor = HousingDataProcessor()
        self.data = self._load_data()

        # Extract series for analysis
        self.price_series = self.data['shiller_index'].dropna() if 'shiller_index' in self.data.columns else None
        self.return_series = self.data[target_variable].dropna()

        # Results storage
        self.moving_averages = {}
        self.trend_signals = {}
        self.ma_statistics = {}
        self.crossover_signals = {}

        print(f"Simple Moving Average Model initialized:")
        print(f"- Target variable: {target_variable}")
        print(f"- MA windows: {self.ma_windows} months")
        print(f"- Sample size: {len(self.return_series)} observations")
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

    def calculate_moving_averages(self):
        """Calculate moving averages for different window sizes"""
        print("Calculating moving averages...")

        # For price series (if available)
        if self.price_series is not None:
            for window in self.ma_windows:
                ma_prices = self.price_series.rolling(window=window, min_periods=max(2, window // 2)).mean()
                self.moving_averages[f'price_ma_{window}m'] = {
                    'values': ma_prices,
                    'window': window,
                    'type': 'price'
                }

        # For return series
        for window in self.ma_windows:
            ma_returns = self.return_series.rolling(window=window, min_periods=max(2, window // 2)).mean()
            self.moving_averages[f'return_ma_{window}m'] = {
                'values': ma_returns,
                'window': window,
                'type': 'return'
            }

        print(f"Moving averages calculated for {len(self.ma_windows)} window sizes")
        return self.moving_averages

    def generate_trend_signals(self):
        """Generate simple trend signals based on moving averages"""
        print("Generating trend signals...")

        # Price trend signals (if price data available)
        if self.price_series is not None and f'price_ma_12m' in self.moving_averages:
            ma_12m = self.moving_averages['price_ma_12m']['values']

            # Simple trend: price above/below 12-month MA
            price_trend = pd.Series(index=self.price_series.index, dtype=str)
            price_trend[self.price_series > ma_12m] = 'Uptrend'
            price_trend[self.price_series <= ma_12m] = 'Downtrend'

            self.trend_signals['price_trend'] = price_trend

        # Return trend signals
        if f'return_ma_6m' in self.moving_averages:
            ma_6m_returns = self.moving_averages['return_ma_6m']['values']

            # Return trend: 6-month MA above/below zero
            return_trend = pd.Series(index=self.return_series.index, dtype=str)
            return_trend[ma_6m_returns > 0] = 'Positive Returns'
            return_trend[ma_6m_returns <= 0] = 'Negative Returns'

            self.trend_signals['return_trend'] = return_trend

        return self.trend_signals

    def detect_crossover_signals(self):
        """Detect moving average crossover signals"""
        print("Detecting crossover signals...")

        # Use 6-month and 12-month MAs for crossover signals
        if (f'return_ma_6m' in self.moving_averages and
                f'return_ma_12m' in self.moving_averages):
            ma_6m = self.moving_averages['return_ma_6m']['values']
            ma_12m = self.moving_averages['return_ma_12m']['values']

            # Calculate crossover signals
            ma_diff = ma_6m - ma_12m

            # Detect crossovers
            crossovers = pd.Series(index=ma_diff.index, dtype=str)

            # Golden cross: short MA crosses above long MA
            golden_cross = (ma_diff > 0) & (ma_diff.shift(1) <= 0)
            # Death cross: short MA crosses below long MA
            death_cross = (ma_diff < 0) & (ma_diff.shift(1) >= 0)

            crossovers[golden_cross] = 'Golden Cross'
            crossovers[death_cross] = 'Death Cross'
            crossovers[~(golden_cross | death_cross)] = 'No Signal'

            self.crossover_signals = {
                'signals': crossovers,
                'ma_difference': ma_diff,
                'golden_crosses': crossovers[crossovers == 'Golden Cross'],
                'death_crosses': crossovers[crossovers == 'Death Cross']
            }

        return self.crossover_signals

    def calculate_ma_statistics(self):
        """Calculate moving average performance statistics"""
        print("Calculating MA statistics...")

        for ma_key, ma_data in self.moving_averages.items():
            ma_values = ma_data['values'].dropna()
            window = ma_data['window']
            ma_type = ma_data['type']

            if len(ma_values) > 0:
                # Basic statistics
                stats_dict = {
                    'mean': ma_values.mean(),
                    'std': ma_values.std(),
                    'min': ma_values.min(),
                    'max': ma_values.max(),
                    'autocorr_lag1': ma_values.autocorr(lag=1) if len(ma_values) > 1 else np.nan,
                    'trend_slope': self._calculate_trend_slope(ma_values),
                    'smoothness': self._calculate_smoothness(ma_values),
                    'window_size': window,
                    'type': ma_type
                }

                # Return-specific statistics
                if ma_type == 'return':
                    aligned_returns = self.return_series.loc[ma_values.index]
                    stats_dict.update({
                        'tracking_error': (aligned_returns - ma_values).std(),
                        'correlation_with_returns': ma_values.corr(aligned_returns)
                    })

                self.ma_statistics[ma_key] = stats_dict

        return self.ma_statistics

    def _calculate_trend_slope(self, series):
        """Calculate trend slope using linear regression"""
        if len(series) < 2:
            return np.nan

        x = np.arange(len(series))
        y = series.values

        # Handle NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan

        slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
        return slope

    def _calculate_smoothness(self, series):
        """Calculate smoothness as inverse of second derivative variance"""
        if len(series) < 3:
            return np.nan

        # Second differences as proxy for smoothness
        second_diff = series.diff().diff().dropna()

        if len(second_diff) == 0:
            return np.nan

        # Lower variance in second differences = smoother
        return 1 / (1 + second_diff.var())

    def evaluate_forecasting_performance(self):
        """Simple forecasting evaluation using moving averages"""
        print("Evaluating simple forecasting performance...")

        forecast_results = {}

        # Use 12-month MA for simple trend forecasting
        if f'return_ma_12m' in self.moving_averages:
            ma_12m = self.moving_averages['return_ma_12m']['values']

            # Simple forecast: next period = current MA
            forecasts = ma_12m.shift(1)  # Lag by 1 period for forecasting
            actuals = self.return_series

            # Align forecasts and actuals
            common_index = forecasts.index.intersection(actuals.index)
            forecasts_aligned = forecasts.loc[common_index]
            actuals_aligned = actuals.loc[common_index]

            # Remove NaN values
            mask = ~(forecasts_aligned.isna() | actuals_aligned.isna())
            forecasts_clean = forecasts_aligned[mask]
            actuals_clean = actuals_aligned[mask]

            if len(forecasts_clean) > 0:
                # Calculate forecast errors
                errors = actuals_clean - forecasts_clean

                forecast_results = {
                    'mae': np.abs(errors).mean(),
                    'mse': (errors ** 2).mean(),
                    'rmse': np.sqrt((errors ** 2).mean()),
                    'mape': np.abs(errors / actuals_clean).mean() * 100,
                    'directional_accuracy': ((forecasts_clean > 0) == (actuals_clean > 0)).mean(),
                    'correlation': forecasts_clean.corr(actuals_clean),
                    'n_forecasts': len(forecasts_clean)
                }

        return forecast_results

    def plot_moving_averages(self, figsize=(15, 10)):
        """Plot 1: Moving averages overlay"""
        if not self.moving_averages:
            self.calculate_moving_averages()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        colors = ['blue', 'red', 'green', 'orange']

        # 1. Price series with moving averages (if available)
        if self.price_series is not None:
            ax1.plot(self.price_series.index, self.price_series.values,
                     color='black', linewidth=1, alpha=0.7, label='Actual Prices')

            i = 0
            for ma_key, ma_data in self.moving_averages.items():
                if ma_data['type'] == 'price':
                    ma_values = ma_data['values']
                    window = ma_data['window']

                    ax1.plot(ma_values.index, ma_values.values,
                             color=colors[i % len(colors)], linewidth=2, alpha=0.8,
                             label=f'{window}-Month MA')
                    i += 1

            ax1.set_title('Housing Prices with Moving Averages', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price Index')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add economic crisis periods
            crisis_periods = [
                ('2007-12', '2009-06', 'Great Recession'),
                ('2020-02', '2020-04', 'COVID Crisis')
            ]

            for start, end, label in crisis_periods:
                try:
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)
                    ax1.axvspan(start_date, end_date, alpha=0.2, color='gray',
                                label=label if start == '2007-12' else "")
                except:
                    pass
        else:
            ax1.text(0.5, 0.5, 'Price data not available', ha='center', va='center',
                     transform=ax1.transAxes, fontsize=14)

        # 2. Return series with moving averages
        ax2.plot(self.return_series.index, self.return_series.values,
                 color='black', linewidth=1, alpha=0.5, label='Actual Returns')

        i = 0
        for ma_key, ma_data in self.moving_averages.items():
            if ma_data['type'] == 'return':
                ma_values = ma_data['values']
                window = ma_data['window']

                ax2.plot(ma_values.index, ma_values.values,
                         color=colors[i % len(colors)], linewidth=2, alpha=0.8,
                         label=f'{window}-Month MA')
                i += 1

        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Housing Returns with Moving Averages', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Returns')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trend_signals(self, figsize=(14, 10)):
        """Plot 2: Trend signals and crossovers"""
        if not self.trend_signals:
            self.generate_trend_signals()
        if not self.crossover_signals:
            self.detect_crossover_signals()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Return trend signals
        if 'return_trend' in self.trend_signals:
            return_trend = self.trend_signals['return_trend']
            ma_6m = self.moving_averages['return_ma_6m']['values']

            ax1.plot(self.return_series.index, self.return_series.values,
                     color='gray', alpha=0.5, linewidth=1, label='Returns')
            ax1.plot(ma_6m.index, ma_6m.values,
                     color='blue', linewidth=2, label='6-Month MA')

            # Color background by trend
            positive_mask = return_trend == 'Positive Returns'
            negative_mask = return_trend == 'Negative Returns'

            if positive_mask.any():
                ax1.fill_between(self.return_series.index, -0.1, 0.1,
                                 where=positive_mask.reindex(self.return_series.index, fill_value=False),
                                 alpha=0.2, color='green', label='Positive Trend')
            if negative_mask.any():
                ax1.fill_between(self.return_series.index, -0.1, 0.1,
                                 where=negative_mask.reindex(self.return_series.index, fill_value=False),
                                 alpha=0.2, color='red', label='Negative Trend')

            ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax1.set_title('Return Trend Signals', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Returns')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # 2. Moving average crossovers
        if self.crossover_signals:
            ma_6m = self.moving_averages['return_ma_6m']['values']
            ma_12m = self.moving_averages['return_ma_12m']['values']

            ax2.plot(ma_6m.index, ma_6m.values, color='blue', linewidth=2, label='6-Month MA')
            ax2.plot(ma_12m.index, ma_12m.values, color='red', linewidth=2, label='12-Month MA')

            # Mark crossover points
            golden_crosses = self.crossover_signals['golden_crosses']
            death_crosses = self.crossover_signals['death_crosses']

            if len(golden_crosses) > 0:
                for date in golden_crosses.index:
                    if date in ma_6m.index:
                        ax2.scatter(date, ma_6m.loc[date], color='green', s=100,
                                    marker='^', zorder=5,
                                    label='Golden Cross' if date == golden_crosses.index[0] else "")

            if len(death_crosses) > 0:
                for date in death_crosses.index:
                    if date in ma_6m.index:
                        ax2.scatter(date, ma_6m.loc[date], color='red', s=100,
                                    marker='v', zorder=5, label='Death Cross' if date == death_crosses.index[0] else "")

            ax2.set_title('Moving Average Crossovers', fontsize=12, fontweight='bold')
            ax2.set_ylabel('MA Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. MA difference (momentum indicator)
        if self.crossover_signals:
            ma_diff = self.crossover_signals['ma_difference']

            ax3.plot(ma_diff.index, ma_diff.values, color='purple', linewidth=2)
            ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax3.fill_between(ma_diff.index, 0, ma_diff.values,
                             where=(ma_diff.values > 0), alpha=0.3, color='green', label='Bullish')
            ax3.fill_between(ma_diff.index, 0, ma_diff.values,
                             where=(ma_diff.values <= 0), alpha=0.3, color='red', label='Bearish')

            ax3.set_title('MA Momentum (6M - 12M)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('MA Difference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Signal statistics
        ax4.axis('off')

        signal_text = "Moving Average Signals Summary\n"
        signal_text += "=" * 35 + "\n\n"

        # Crossover statistics
        if self.crossover_signals:
            n_golden = len(self.crossover_signals['golden_crosses'])
            n_death = len(self.crossover_signals['death_crosses'])
            signal_text += f"Crossover Signals:\n"
            signal_text += f"• Golden Crosses: {n_golden}\n"
            signal_text += f"• Death Crosses: {n_death}\n"
            signal_text += f"• Signal Frequency: {(n_golden + n_death) / len(self.return_series) * 100:.1f}%\n\n"

        # Trend statistics
        if 'return_trend' in self.trend_signals:
            trend_counts = self.trend_signals['return_trend'].value_counts()
            for trend, count in trend_counts.items():
                pct = count / len(self.trend_signals['return_trend']) * 100
                signal_text += f"• {trend}: {pct:.1f}%\n"
            signal_text += "\n"

        # Performance comparison
        forecast_results = self.evaluate_forecasting_performance()
        if forecast_results:
            signal_text += f"Simple Forecasting Performance:\n"
            signal_text += f"• RMSE: {forecast_results['rmse']:.6f}\n"
            signal_text += f"• Directional Accuracy: {forecast_results['directional_accuracy']:.1%}\n"
            signal_text += f"• Correlation: {forecast_results['correlation']:.3f}\n\n"

        signal_text += f"vs. Sophisticated Models:\n\n"
        signal_text += f"Transfer Function Model:\n"
        signal_text += f"• R² = 0.8980\n"
        signal_text += f"• 10-month policy delay\n"
        signal_text += f"• Complex lag structure\n\n"
        signal_text += f"Simple MA Limitations:\n"
        signal_text += f"• Lagging indicator\n"
        signal_text += f"• No economic variables\n"
        signal_text += f"• Linear trend assumption\n"
        signal_text += f"• No volatility modeling"

        ax4.text(0.05, 0.95, signal_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_ma_performance(self, figsize=(14, 8)):
        """Plot 3: Moving average performance analysis"""
        if not self.ma_statistics:
            self.calculate_ma_statistics()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Smoothness vs Window Size
        return_mas = {k: v for k, v in self.ma_statistics.items() if v['type'] == 'return'}

        windows = [stats['window_size'] for stats in return_mas.values()]
        smoothness = [stats['smoothness'] for stats in return_mas.values()]
        autocorrs = [stats['autocorr_lag1'] for stats in return_mas.values()]

        ax1.plot(windows, smoothness, 'o-', color='blue', linewidth=2, markersize=8, label='Smoothness')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(windows, autocorrs, 's-', color='red', linewidth=2, markersize=8, label='Autocorrelation')

        ax1.set_xlabel('Window Size (Months)')
        ax1.set_ylabel('Smoothness', color='blue')
        ax1_twin.set_ylabel('Autocorrelation', color='red')
        ax1.set_title('MA Smoothness vs Window Size', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add legends
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')

        # 2. Tracking Error
        tracking_errors = [stats.get('tracking_error', np.nan) for stats in return_mas.values()]

        ax2.bar(range(len(windows)), tracking_errors, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('Tracking Error')
        ax2.set_title('MA Tracking Error', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(windows)))
        ax2.set_xticklabels([f'{w}M' for w in windows])
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for i, te in enumerate(tracking_errors):
            if not np.isnan(te):
                ax2.text(i, te, f'{te:.4f}', ha='center', va='bottom')

        # 3. Trend slopes
        trend_slopes = [stats['trend_slope'] * 1000 for stats in return_mas.values()]  # Scale for visualization

        colors = ['green' if slope > 0 else 'red' for slope in trend_slopes]
        bars = ax3.bar(range(len(windows)), trend_slopes, color=colors, alpha=0.7)

        ax3.set_xlabel('Window Size')
        ax3.set_ylabel('Trend Slope (×1000)')
        ax3.set_title('MA Trend Slopes', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(windows)))
        ax3.set_xticklabels([f'{w}M' for w in windows])
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for i, slope in enumerate(trend_slopes):
            height = bars[i].get_height()
            ax3.text(i, height + (0.1 if height >= 0 else -0.1), f'{slope:.2f}',
                     ha='center', va='bottom' if height >= 0 else 'top')

        # 4. Performance summary table
        ax4.axis('off')

        perf_text = "Moving Average Performance\n"
        perf_text += "=" * 30 + "\n\n"

        for i, (window, stats) in enumerate(zip(windows, return_mas.values())):
            perf_text += f"{window}-Month MA:\n"
            perf_text += f"  Smoothness: {stats['smoothness']:.3f}\n"
            perf_text += f"  Tracking Error: {stats.get('tracking_error', 0):.4f}\n"
            perf_text += f"  Autocorr: {stats['autocorr_lag1']:.3f}\n"
            perf_text += f"  Trend Slope: {stats['trend_slope'] * 1000:.2f}\n\n"

        # Add forecasting performance
        forecast_results = self.evaluate_forecasting_performance()
        if forecast_results:
            perf_text += f"Simple Forecasting:\n"
            perf_text += f"  RMSE: {forecast_results['rmse']:.6f}\n"
            perf_text += f"  MAE: {forecast_results['mae']:.6f}\n"
            perf_text += f"  Directional: {forecast_results['directional_accuracy']:.1%}\n\n"

        perf_text += f"Best Window Selection:\n"
        if len(tracking_errors) > 0:
            best_window_idx = np.nanargmin(tracking_errors)
            perf_text += f"• Lowest tracking error: {windows[best_window_idx]}M\n"
        if len(smoothness) > 0:
            smoothest_idx = np.nanargmax(smoothness)
            perf_text += f"• Smoothest: {windows[smoothest_idx]}M"

        ax4.text(0.05, 0.95, perf_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_all_results(self, show_plots=True, save_plots=False, save_path='./'):
        """Generate all plots for moving average analysis"""
        plots = {}

        print("Generating Simple Moving Average analysis plots...")

        # Calculate all moving averages and signals
        self.calculate_moving_averages()
        self.generate_trend_signals()
        self.detect_crossover_signals()
        self.calculate_ma_statistics()

        plots['moving_averages'] = self.plot_moving_averages()
        print("✓ Plot 1: Moving Averages Overlay")

        plots['trend_signals'] = self.plot_trend_signals()
        print("✓ Plot 2: Trend Signals and Crossovers")

        plots['ma_performance'] = self.plot_ma_performance()
        print("✓ Plot 3: Moving Average Performance Analysis")

        # Save plots if requested
        if save_plots:
            for plot_name, fig in plots.items():
                filename = f"{save_path}moving_average_{plot_name}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")

        # Show plots if requested
        if show_plots:
            plt.show()

        return plots

    def summary(self):
        """Print moving average analysis summary"""
        print("Simple Moving Average Model Results")
        print("=" * 40)

        # Moving average statistics
        if self.ma_statistics:
            print("Moving Average Statistics:")
            return_mas = {k: v for k, v in self.ma_statistics.items() if v['type'] == 'return'}
            for ma_key, stats in return_mas.items():
                window = stats['window_size']
                print(f"  {window}-Month MA:")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Tracking Error: {stats.get('tracking_error', 0):.6f}")
                print(f"    Smoothness: {stats['smoothness']:.3f}")

        # Signal statistics
        if self.crossover_signals:
            n_golden = len(self.crossover_signals['golden_crosses'])
            n_death = len(self.crossover_signals['death_crosses'])
            print(f"\nCrossover Signals:")
            print(f"  Golden Crosses: {n_golden}")
            print(f"  Death Crosses: {n_death}")

        # Forecasting performance
        forecast_results = self.evaluate_forecasting_performance()
        if forecast_results:
            print(f"\nSimple Forecasting Performance:")
            print(f"  RMSE: {forecast_results['rmse']:.6f}")
            print(f"  Directional Accuracy: {forecast_results['directional_accuracy']:.1%}")
            print(f"  Correlation: {forecast_results['correlation']:.3f}")

    def get_model_results(self):
        """Return model results for external use"""
        return {
            'model_object': self,
            'moving_averages': self.moving_averages,
            'trend_signals': self.trend_signals,
            'crossover_signals': self.crossover_signals,
            'ma_statistics': self.ma_statistics,
            'forecast_performance': self.evaluate_forecasting_performance(),
            'data': self.data,
            'target_variable': self.target_variable
        }


# Convenience function
def analyze_moving_averages_housing(target_variable='shiller_return', ma_windows=None):
    """
    Convenience function for moving average analysis

    Parameters:
    -----------
    target_variable : str, default 'shiller_return'
        Target variable for analysis
    ma_windows : list, optional
        Moving average window sizes

    Returns:
    --------
    model : SimpleMovingAverageHousingModel
        Moving average analysis model
    """
    print("Running Simple Moving Average Analysis...")

    # Initialize and run analysis
    model = SimpleMovingAverageHousingModel(
        target_variable=target_variable,
        ma_windows=ma_windows
    )

    # Calculate all moving averages and signals
    model.calculate_moving_averages()
    model.generate_trend_signals()
    model.detect_crossover_signals()
    model.calculate_ma_statistics()

    # Print summary
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage
    print("Simple Moving Average Model - Housing Market")
    print("Basic trend analysis baseline for sophisticated models")

    model = analyze_moving_averages_housing()

    # Generate plots
    print("\nGenerating analysis plots...")
    plots = model.plot_all_results(show_plots=True, save_plots=False)
    print(f"\nGenerated {len(plots)} plots:")
    print("1. Moving Averages Overlay")
    print("2. Trend Signals and Crossovers")
    print("3. Moving Average Performance Analysis")

    print("\nMoving Average analysis complete!")