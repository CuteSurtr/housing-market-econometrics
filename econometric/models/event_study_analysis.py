"""
Event Study Analysis with Housing Data Processor Integration
Simple before/after event analysis baseline for comparison with sophisticated models
276 observations (2000-01 to 2024-12)
Creates individual plots using basic box plots, bar charts, and line graphs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Import housing data processor
from housing_data_processor import HousingDataProcessor

warnings.filterwarnings('ignore')


class EventStudyHousingModel:
    """
    Event Study Analysis for housing market around economic events
    Simple before/after comparison baseline vs sophisticated econometric models

    Comparison:
    - Event Study: Simple before/after event analysis
    - Transfer Function: 10-month policy transmission delay (R² = 0.8980)
    - Event Study: Immediate impact assumption
    - Sophisticated Models: Complex lag structures and time-varying effects
    """

    def __init__(self, target_variable='shiller_return', event_windows=None):
        """
        Initialize Event Study Analysis

        Parameters:
        -----------
        target_variable : str, default 'shiller_return'
            Target variable for event analysis
        event_windows : dict, optional
            Event window specifications (pre, post periods)
        """
        self.target_variable = target_variable
        self.event_windows = event_windows or {
            'pre_event': 6,  # 6 months before
            'post_event': 12,  # 12 months after
            'short_term': 3,  # 3 months for short-term analysis
            'long_term': 24  # 24 months for long-term analysis
        }

        # Load and prepare data
        self.processor = HousingDataProcessor()
        self.data = self._load_data()

        # Extract series for analysis
        self.return_series = self.data[target_variable].dropna()
        self.fed_series = self.data['fed_rate'].dropna() if 'fed_rate' in self.data.columns else None

        # Event definitions
        self.events = self._define_events()

        # Results storage
        self.event_results = {}
        self.abnormal_returns = {}
        self.event_statistics = {}

        print(f"Event Study Analysis initialized:")
        print(f"- Target variable: {target_variable}")
        print(f"- Event windows: {self.event_windows}")
        print(f"- Sample size: {len(self.return_series)} observations")
        print(f"- Sample period: {self.data.index.min()} to {self.data.index.max()}")
        print(f"- Events defined: {len(self.events)} event categories")

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

    def _define_events(self):
        """Define economic events for study"""
        print("Defining economic events...")

        events = {
            'fed_rate_increases': {
                'description': 'Federal funds rate increases (>0.25%)',
                'dates': [],
                'type': 'monetary_policy'
            },
            'fed_rate_decreases': {
                'description': 'Federal funds rate decreases (>0.25%)',
                'dates': [],
                'type': 'monetary_policy'
            },
            'major_crises': {
                'description': 'Major economic crises',
                'dates': [
                    '2001-03-01',  # Dot-com crash
                    '2007-12-01',  # Great Recession start
                    '2020-03-01',  # COVID crisis
                ],
                'type': 'crisis'
            },
            'policy_regime_changes': {
                'description': 'Major policy regime changes',
                'dates': [
                    '2008-12-01',  # Zero Interest Rate Policy
                    '2015-12-01',  # First rate hike post-crisis
                    '2020-03-01',  # Emergency rate cuts
                ],
                'type': 'policy_regime'
            },
            'housing_policy_events': {
                'description': 'Housing-specific policy events',
                'dates': [
                    '2008-09-01',  # Lehman Brothers / Housing crisis
                    '2010-04-01',  # Dodd-Frank discussions
                    '2020-04-01',  # Mortgage forbearance programs
                ],
                'type': 'housing_policy'
            }
        }

        # Identify Fed rate change events from data
        if self.fed_series is not None:
            fed_changes = self.fed_series.diff().dropna()

            # Significant increases (>0.25%)
            increases = fed_changes[fed_changes > 0.25].index
            events['fed_rate_increases']['dates'] = [date.strftime('%Y-%m-%d') for date in increases]

            # Significant decreases (>0.25%)
            decreases = fed_changes[fed_changes < -0.25].index
            events['fed_rate_decreases']['dates'] = [date.strftime('%Y-%m-%d') for date in decreases]

        # Convert string dates to datetime
        for event_type in events:
            events[event_type]['dates'] = [pd.to_datetime(date) for date in events[event_type]['dates']]

        print(f"Events identified:")
        for event_type, event_info in events.items():
            print(f"  {event_type}: {len(event_info['dates'])} events")

        return events

    def calculate_abnormal_returns(self, benchmark_window=60):
        """Calculate abnormal returns using market model"""
        print(f"Calculating abnormal returns using {benchmark_window}-month benchmark...")

        # Calculate normal (expected) returns using rolling mean
        normal_returns = self.return_series.rolling(window=benchmark_window, min_periods=30).mean()

        # Abnormal returns = actual - expected
        abnormal_returns = self.return_series - normal_returns

        self.abnormal_returns = {
            'abnormal_returns': abnormal_returns.dropna(),
            'normal_returns': normal_returns.dropna(),
            'benchmark_window': benchmark_window
        }

        return self.abnormal_returns

    def analyze_events(self):
        """Analyze each event category"""
        print("Analyzing events...")

        if not self.abnormal_returns:
            self.calculate_abnormal_returns()

        abnormal_returns = self.abnormal_returns['abnormal_returns']

        for event_type, event_info in self.events.items():
            event_dates = event_info['dates']

            if len(event_dates) == 0:
                continue

            event_analysis = {
                'pre_event_returns': [],
                'post_event_returns': [],
                'short_term_returns': [],
                'long_term_returns': [],
                'event_dates': event_dates,
                'cumulative_abnormal_returns': [],
                'individual_event_stats': []
            }

            for event_date in event_dates:
                try:
                    # Define windows around event
                    pre_start = event_date - pd.DateOffset(months=self.event_windows['pre_event'])
                    pre_end = event_date - pd.DateOffset(months=1)
                    post_start = event_date
                    post_end = event_date + pd.DateOffset(months=self.event_windows['post_event'])
                    short_end = event_date + pd.DateOffset(months=self.event_windows['short_term'])
                    long_end = event_date + pd.DateOffset(months=self.event_windows['long_term'])

                    # Extract returns for each window
                    pre_returns = self.return_series[(self.return_series.index >= pre_start) &
                                                     (self.return_series.index <= pre_end)]
                    post_returns = self.return_series[(self.return_series.index >= post_start) &
                                                      (self.return_series.index <= post_end)]
                    short_returns = self.return_series[(self.return_series.index >= post_start) &
                                                       (self.return_series.index <= short_end)]
                    long_returns = self.return_series[(self.return_series.index >= post_start) &
                                                      (self.return_series.index <= long_end)]

                    # Abnormal returns around event
                    abnormal_event = abnormal_returns[(abnormal_returns.index >= pre_start) &
                                                      (abnormal_returns.index <= post_end)]

                    # Store individual event statistics
                    if len(pre_returns) > 0 and len(post_returns) > 0:
                        event_stat = {
                            'event_date': event_date,
                            'pre_mean': pre_returns.mean(),
                            'post_mean': post_returns.mean(),
                            'short_mean': short_returns.mean() if len(short_returns) > 0 else np.nan,
                            'long_mean': long_returns.mean() if len(long_returns) > 0 else np.nan,
                            'difference': post_returns.mean() - pre_returns.mean(),
                            'cumulative_abnormal': abnormal_event.sum() if len(abnormal_event) > 0 else np.nan
                        }
                        event_analysis['individual_event_stats'].append(event_stat)

                        # Aggregate across events
                        event_analysis['pre_event_returns'].extend(pre_returns.tolist())
                        event_analysis['post_event_returns'].extend(post_returns.tolist())
                        event_analysis['short_term_returns'].extend(short_returns.tolist())
                        event_analysis['long_term_returns'].extend(long_returns.tolist())

                        if len(abnormal_event) > 0:
                            event_analysis['cumulative_abnormal_returns'].append(abnormal_event.sum())

                except Exception as e:
                    print(f"Warning: Could not analyze event {event_date} for {event_type}: {e}")

            self.event_results[event_type] = event_analysis

        return self.event_results

    def calculate_event_statistics(self):
        """Calculate statistical tests for event impacts"""
        print("Calculating event statistics...")

        for event_type, event_data in self.event_results.items():
            if len(event_data['pre_event_returns']) == 0 or len(event_data['post_event_returns']) == 0:
                continue

            pre_returns = np.array(event_data['pre_event_returns'])
            post_returns = np.array(event_data['post_event_returns'])
            short_returns = np.array(event_data['short_term_returns'])

            # T-test for difference in means
            t_stat, t_pvalue = stats.ttest_ind(post_returns, pre_returns)

            # Wilcoxon rank-sum test (non-parametric)
            try:
                w_stat, w_pvalue = stats.ranksums(post_returns, pre_returns)
            except:
                w_stat, w_pvalue = np.nan, np.nan

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(pre_returns) + np.var(post_returns)) / 2)
            cohens_d = (np.mean(post_returns) - np.mean(pre_returns)) / pooled_std if pooled_std > 0 else np.nan

            # Cumulative abnormal return statistics
            car_values = event_data['cumulative_abnormal_returns']
            car_mean = np.mean(car_values) if len(car_values) > 0 else np.nan
            car_tstat = np.mean(car_values) / (np.std(car_values) / np.sqrt(len(car_values))) if len(
                car_values) > 1 else np.nan

            statistics = {
                'pre_event_mean': np.mean(pre_returns),
                'post_event_mean': np.mean(post_returns),
                'short_term_mean': np.mean(short_returns) if len(short_returns) > 0 else np.nan,
                'difference': np.mean(post_returns) - np.mean(pre_returns),
                'pre_event_std': np.std(pre_returns),
                'post_event_std': np.std(post_returns),
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_pvalue': w_pvalue,
                'cohens_d': cohens_d,
                'car_mean': car_mean,
                'car_tstatistic': car_tstat,
                'n_events': len(event_data['individual_event_stats']),
                'n_pre_obs': len(pre_returns),
                'n_post_obs': len(post_returns),
                'significant': t_pvalue < 0.05 if not np.isnan(t_pvalue) else False
            }

            self.event_statistics[event_type] = statistics

        return self.event_statistics

    def plot_event_overview(self, figsize=(15, 10)):
        """Plot 1: Event overview and timeline"""
        if not self.event_results:
            self.analyze_events()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # 1. Return series with event markers
        ax1.plot(self.return_series.index, self.return_series.values,
                 color='black', linewidth=1, alpha=0.7, label='Housing Returns')

        # Add event markers
        colors = {'fed_rate_increases': 'red', 'fed_rate_decreases': 'green',
                  'major_crises': 'purple', 'policy_regime_changes': 'orange',
                  'housing_policy_events': 'blue'}

        markers = {'fed_rate_increases': '^', 'fed_rate_decreases': 'v',
                   'major_crises': 's', 'policy_regime_changes': 'D',
                   'housing_policy_events': 'o'}

        for event_type, event_data in self.event_results.items():
            if len(event_data['event_dates']) > 0:
                event_dates = event_data['event_dates']

                for i, event_date in enumerate(event_dates):
                    if event_date in self.return_series.index:
                        event_return = self.return_series.loc[event_date]
                    else:
                        # Find closest date
                        closest_date = min(self.return_series.index, key=lambda x: abs(x - event_date))
                        event_return = self.return_series.loc[closest_date]

                    ax1.scatter(event_date, event_return,
                                color=colors.get(event_type, 'black'),
                                marker=markers.get(event_type, 'o'),
                                s=100, alpha=0.8, zorder=5,
                                label=event_type.replace('_', ' ').title() if i == 0 else "")

        ax1.set_title('Housing Returns with Economic Events', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Returns')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)

        # 2. Event frequency by year
        event_counts_by_year = {}

        for event_type, event_data in self.event_results.items():
            for event_date in event_data['event_dates']:
                year = event_date.year
                if year not in event_counts_by_year:
                    event_counts_by_year[year] = 0
                event_counts_by_year[year] += 1

        if event_counts_by_year:
            years = sorted(event_counts_by_year.keys())
            counts = [event_counts_by_year[year] for year in years]

            ax2.bar(years, counts, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.set_title('Economic Events Frequency by Year', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Number of Events')
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for year, count in zip(years, counts):
                ax2.text(year, count, str(count), ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def plot_event_analysis(self, figsize=(15, 8)):
        """Plot 2: Event impact analysis"""
        if not self.event_statistics:
            self.calculate_event_statistics()

        # Filter events with sufficient data
        valid_events = {k: v for k, v in self.event_statistics.items()
                        if v['n_events'] > 0 and not np.isnan(v['difference'])}

        if len(valid_events) == 0:
            print("No valid events for analysis")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 1. Pre vs Post event returns (box plot)
        event_names = list(valid_events.keys())
        pre_post_data = []
        labels = []

        for event_type in event_names:
            event_data = self.event_results[event_type]
            pre_returns = event_data['pre_event_returns']
            post_returns = event_data['post_event_returns']

            pre_post_data.extend([pre_returns, post_returns])
            labels.extend([f'{event_type.replace("_", " ")}\nPre', f'{event_type.replace("_", " ")}\nPost'])

        # Create box plot
        box_plot = ax1.boxplot(pre_post_data, patch_artist=True, labels=labels)

        # Color boxes alternately
        colors = ['lightblue', 'lightcoral'] * len(event_names)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title('Pre vs Post Event Returns Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Returns')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)

        # 2. Effect sizes (Cohen's d)
        cohens_d_values = [stats['cohens_d'] for stats in valid_events.values()]
        event_labels = [name.replace('_', '\n').title() for name in valid_events.keys()]

        colors_effect = ['green' if d > 0 else 'red' for d in cohens_d_values]
        bars = ax2.barh(event_labels, cohens_d_values, color=colors_effect, alpha=0.7)

        ax2.set_title('Effect Sizes (Cohen\'s d)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Effect Size')
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
        ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Medium Effect')
        ax2.axvline(0.8, color='gray', linestyle='--', alpha=0.9, label='Large Effect')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add value labels
        for bar, value in zip(bars, cohens_d_values):
            width = bar.get_width()
            ax2.text(width + (0.02 if width >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                     f'{value:.2f}', ha='left' if width >= 0 else 'right', va='center',
                     fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_cumulative_abnormal_returns(self, figsize=(14, 10)):
        """Plot 3: Cumulative abnormal returns analysis"""
        if not self.abnormal_returns:
            self.calculate_abnormal_returns()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Overall abnormal returns time series
        abnormal_returns = self.abnormal_returns['abnormal_returns']

        ax1.plot(abnormal_returns.index, abnormal_returns.values,
                 color='purple', linewidth=1, alpha=0.7, label='Abnormal Returns')
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)

        # Add 2-sigma bands
        ar_std = abnormal_returns.std()
        ax1.axhline(2 * ar_std, color='red', linestyle='--', alpha=0.7, label='±2σ')
        ax1.axhline(-2 * ar_std, color='red', linestyle='--', alpha=0.7)

        ax1.set_title('Abnormal Returns Time Series', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Abnormal Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative abnormal returns by event type
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (event_type, event_data) in enumerate(self.event_results.items()):
            if len(event_data['cumulative_abnormal_returns']) > 0:
                car_values = event_data['cumulative_abnormal_returns']
                event_dates = event_data['event_dates'][:len(car_values)]

                ax2.scatter(range(len(car_values)), car_values,
                            color=colors[i % len(colors)], alpha=0.7, s=60,
                            label=event_type.replace('_', ' ').title())

        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Cumulative Abnormal Returns by Event', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Event Number')
        ax2.set_ylabel('Cumulative Abnormal Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Event window analysis (average pattern)
        # Create average event pattern
        if len(self.event_results) > 0:
            # Combine all events for average pattern
            all_event_patterns = []

            for event_type, event_data in self.event_results.items():
                for event_date in event_data['event_dates']:
                    try:
                        # Extract returns around event (-6 to +12 months)
                        start_date = event_date - pd.DateOffset(months=6)
                        end_date = event_date + pd.DateOffset(months=12)

                        event_returns = self.return_series[(self.return_series.index >= start_date) &
                                                           (self.return_series.index <= end_date)]

                        if len(event_returns) >= 12:  # Ensure sufficient data
                            # Align to event date (time 0)
                            event_pattern = []
                            for month_offset in range(-6, 13):
                                target_date = event_date + pd.DateOffset(months=month_offset)
                                closest_return = self._find_closest_return(target_date, event_returns)
                                event_pattern.append(closest_return)

                            if len(event_pattern) == 19:  # -6 to +12 months
                                all_event_patterns.append(event_pattern)

                    except Exception as e:
                        continue

            if len(all_event_patterns) > 0:
                # Calculate average pattern
                avg_pattern = np.nanmean(all_event_patterns, axis=0)
                std_pattern = np.nanstd(all_event_patterns, axis=0)

                months = list(range(-6, 13))

                ax3.plot(months, avg_pattern, 'b-', linewidth=2, label='Average Return')
                ax3.fill_between(months, avg_pattern - std_pattern, avg_pattern + std_pattern,
                                 alpha=0.3, color='blue', label='±1 Std Dev')

                ax3.axvline(0, color='red', linestyle='--', alpha=0.7, label='Event Date')
                ax3.axhline(0, color='black', linestyle='-', alpha=0.5)

                ax3.set_title('Average Event Pattern', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Months Relative to Event')
                ax3.set_ylabel('Average Return')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for average pattern',
                         ha='center', va='center', transform=ax3.transAxes)

        # 4. Abnormal returns distribution
        ax4.hist(abnormal_returns.values, bins=30, alpha=0.7, color='lightcoral',
                 edgecolor='black', density=True)

        # Overlay normal distribution
        x = np.linspace(abnormal_returns.min(), abnormal_returns.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, abnormal_returns.mean(), abnormal_returns.std()),
                 'r-', linewidth=2, label='Normal Distribution')

        # Add statistics
        skewness = abnormal_returns.skew()
        kurtosis = abnormal_returns.kurtosis()

        stats_text = f'Skewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}'
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax4.set_title('Abnormal Returns Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Abnormal Returns')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _find_closest_return(self, target_date, return_series):
        """Find return closest to target date"""
        if target_date in return_series.index:
            return return_series.loc[target_date]
        else:
            # Find closest date
            closest_date = min(return_series.index, key=lambda x: abs(x - target_date))
            return return_series.loc[closest_date]

    def plot_all_results(self, show_plots=True, save_plots=False, save_path='./'):
        """Generate all plots for event study analysis"""
        plots = {}

        print("Generating Event Study analysis plots...")

        # Run all analyses
        self.calculate_abnormal_returns()
        self.analyze_events()
        self.calculate_event_statistics()

        plots['event_overview'] = self.plot_event_overview()
        print("✓ Plot 1: Event Overview and Timeline")

        plots['event_analysis'] = self.plot_event_analysis()
        print("✓ Plot 2: Event Impact Analysis")

        plots['cumulative_abnormal_returns'] = self.plot_cumulative_abnormal_returns()
        print("✓ Plot 3: Cumulative Abnormal Returns Analysis")

        # Save plots if requested
        if save_plots:
            for plot_name, fig in plots.items():
                if fig is not None:
                    filename = f"{save_path}event_study_{plot_name}.png"
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"Saved: {filename}")

        # Show plots if requested
        if show_plots:
            plt.show()

        return plots

    def summary(self):
        """Print event study analysis summary"""
        print("Event Study Analysis Results")
        print("=" * 35)

        # Event counts
        total_events = sum(len(event_data['event_dates']) for event_data in self.event_results.values())
        print(f"Total events analyzed: {total_events}")

        for event_type, event_data in self.event_results.items():
            n_events = len(event_data['event_dates'])
            if n_events > 0:
                print(f"  {event_type.replace('_', ' ').title()}: {n_events} events")

        # Statistical results
        if self.event_statistics:
            print(f"\nEvent Impact Results:")
            significant_events = []

            for event_type, stats in self.event_statistics.items():
                if stats['n_events'] > 0:
                    print(f"  {event_type.replace('_', ' ').title()}:")
                    print(f"    Mean difference: {stats['difference']:.4f}")
                    print(f"    t-statistic: {stats['t_statistic']:.2f}")
                    print(f"    p-value: {stats['t_pvalue']:.4f}")
                    print(f"    Significant: {'Yes' if stats['significant'] else 'No'}")

                    if stats['significant']:
                        significant_events.append(event_type)

            if significant_events:
                print(f"\nStatistically significant events: {len(significant_events)}")
                for event in significant_events:
                    print(f"  - {event.replace('_', ' ').title()}")
            else:
                print(f"\nNo statistically significant event impacts found")

        # Abnormal returns summary
        if self.abnormal_returns:
            ar = self.abnormal_returns['abnormal_returns']
            print(f"\nAbnormal Returns Summary:")
            print(f"  Mean: {ar.mean():.6f}")
            print(f"  Std Dev: {ar.std():.6f}")
            print(f"  Min: {ar.min():.6f}")
            print(f"  Max: {ar.max():.6f}")

    def get_model_results(self):
        """Return model results for external use"""
        return {
            'model_object': self,
            'events': self.events,
            'event_results': self.event_results,
            'event_statistics': self.event_statistics,
            'abnormal_returns': self.abnormal_returns,
            'data': self.data,
            'target_variable': self.target_variable
        }


# Convenience function
def analyze_housing_events(target_variable='shiller_return', event_windows=None):
    """
    Convenience function for event study analysis

    Parameters:
    -----------
    target_variable : str, default 'shiller_return'
        Target variable for analysis
    event_windows : dict, optional
        Event window specifications

    Returns:
    --------
    model : EventStudyHousingModel
        Event study analysis model
    """
    print("Running Event Study Analysis...")

    # Initialize and run analysis
    model = EventStudyHousingModel(
        target_variable=target_variable,
        event_windows=event_windows
    )

    # Run all analyses
    model.calculate_abnormal_returns()
    model.analyze_events()
    model.calculate_event_statistics()

    # Print summary
    model.summary()

    return model


if __name__ == "__main__":
    # Example usage
    print("Event Study Analysis - Housing Market")
    print("Simple before/after event analysis baseline")

    model = analyze_housing_events()

    # Generate plots
    print("\nGenerating analysis plots...")
    plots = model.plot_all_results(show_plots=True, save_plots=False)
    print(f"\nGenerated {len([p for p in plots.values() if p is not None])} plots:")
    print("1. Event Overview and Timeline")
    print("2. Event Impact Analysis")
    print("3. Cumulative Abnormal Returns Analysis")

    print("\nEvent Study analysis complete!")