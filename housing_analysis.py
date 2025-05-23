import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import warnings

warnings.filterwarnings('ignore')


class HousingDataProcessor:
    """
    Data processor for housing datasets with GitHub URL integration
    Loads data directly from public GitHub repository
    """

    def __init__(self):
        self.shiller_data = None
        self.zillow_data = None
        self.fed_data = None
        self.merged_data = None

        # GitHub raw file URLs
        self.github_urls = {
            'shiller': 'https://raw.githubusercontent.com/CuteSurtr/housing-market-econometrics/main/CSUSHPINSA.csv',
            'fed': 'https://raw.githubusercontent.com/CuteSurtr/housing-market-econometrics/main/fed_rate_clean_2000_2024(1).csv',
            'zillow': 'https://raw.githubusercontent.com/CuteSurtr/housing-market-econometrics/main/housing_data_filtered_regions.csv'
        }

    def load_data_from_url(self, url, encoding='utf-8'):
        """Load CSV data from GitHub URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()

            if encoding:
                response.encoding = encoding

            csv_content = StringIO(response.text)
            df = pd.read_csv(csv_content)

            print(f"Successfully loaded data from URL: {len(df)} rows")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error loading data from URL: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def load_case_shiller_data(self, url=None):
        """Load Case-Shiller Housing Price Index data from GitHub"""
        if url is None:
            url = self.github_urls['shiller']

        print("Loading Case-Shiller data from GitHub...")
        self.shiller_data = self.load_data_from_url(url)

        if self.shiller_data is None:
            return None

        # Clean column names and convert date
        self.shiller_data.columns = ['date', 'shiller_index']
        self.shiller_data['date'] = pd.to_datetime(self.shiller_data['date'])

        # Normalize to first of month
        self.shiller_data['date'] = self.shiller_data['date'].dt.to_period('M').dt.start_time
        self.shiller_data.set_index('date', inplace=True)

        # Calculate returns
        self.shiller_data['shiller_return'] = np.log(
            self.shiller_data['shiller_index'] / self.shiller_data['shiller_index'].shift(1)
        )

        print(f"Case-Shiller data processed: {len(self.shiller_data)} observations")
        print(
            f"Date range: {self.shiller_data.index.min().strftime('%Y-%m')} to {self.shiller_data.index.max().strftime('%Y-%m')}")

        return self.shiller_data

    def load_zillow_data(self, url=None, region='United States'):
        """Load Zillow housing data from GitHub and extract national series"""
        if url is None:
            url = self.github_urls['zillow']

        print("Loading Zillow data from GitHub...")
        zillow_raw = self.load_data_from_url(url)

        if zillow_raw is None:
            return None

        # Filter for US national data
        us_data = zillow_raw[zillow_raw['RegionName'] == region].copy()

        if len(us_data) == 0:
            print(f"Warning: {region} not found. Available regions:")
            print(zillow_raw['RegionName'].unique()[:10])
            return None

        # Extract date columns
        date_cols = [col for col in zillow_raw.columns
                     if len(str(col)) == 10 and str(col).count('-') == 2
                     and str(col)[:4].isdigit()]

        # Reshape from wide to long format
        zillow_long = []
        us_row = us_data.iloc[0]

        for date_col in date_cols:
            try:
                date_value = pd.to_datetime(date_col)
                # Normalize to first of month
                date_value = date_value.to_period('M').start_time
                price_value = us_row[date_col]

                if pd.notna(price_value):
                    zillow_long.append({
                        'date': date_value,
                        'zillow_index': price_value
                    })
            except:
                continue

        self.zillow_data = pd.DataFrame(zillow_long)
        self.zillow_data.set_index('date', inplace=True)
        self.zillow_data.sort_index(inplace=True)

        # Calculate returns
        self.zillow_data['zillow_return'] = np.log(
            self.zillow_data['zillow_index'] / self.zillow_data['zillow_index'].shift(1)
        )

        print(f"Zillow data processed: {len(self.zillow_data)} observations")
        print(
            f"Date range: {self.zillow_data.index.min().strftime('%Y-%m')} to {self.zillow_data.index.max().strftime('%Y-%m')}")

        return self.zillow_data

    def load_fed_data(self, url=None):
        """Load Federal Funds Rate data from GitHub"""
        if url is None:
            url = self.github_urls['fed']

        print("Loading Federal Funds data from GitHub...")
        self.fed_data = self.load_data_from_url(url)

        if self.fed_data is None:
            return None

        # Identify date and rate columns
        if 'Date' in self.fed_data.columns:
            date_col = 'Date'
        elif 'date' in self.fed_data.columns:
            date_col = 'date'
        else:
            date_col = self.fed_data.columns[0]

        if 'FedRate' in self.fed_data.columns:
            rate_col = 'FedRate'
        elif 'fed_rate' in self.fed_data.columns:
            rate_col = 'fed_rate'
        else:
            rate_col = self.fed_data.columns[1]

        # Process data
        self.fed_data['date'] = pd.to_datetime(self.fed_data[date_col])

        # Normalize to first of month
        self.fed_data['date'] = self.fed_data['date'].dt.to_period('M').dt.start_time

        self.fed_data = self.fed_data[['date', rate_col]].copy()
        self.fed_data.columns = ['date', 'fed_rate']
        self.fed_data.set_index('date', inplace=True)

        # Calculate derived variables
        self.fed_data['fed_change'] = self.fed_data['fed_rate'].diff()
        self.fed_data['fed_level'] = self.fed_data['fed_rate']
        self.fed_data['fed_vol'] = self.fed_data['fed_change'].rolling(6).std()

        median_rate = self.fed_data['fed_rate'].median()
        self.fed_data['high_rate_regime'] = (self.fed_data['fed_rate'] > median_rate).astype(int)

        self.fed_data['fed_trend'] = self.fed_data['fed_rate'].rolling(24, center=True).mean()
        self.fed_data['fed_cycle'] = self.fed_data['fed_rate'] - self.fed_data['fed_trend']

        print(f"Fed data processed: {len(self.fed_data)} observations")
        print(
            f"Date range: {self.fed_data.index.min().strftime('%Y-%m')} to {self.fed_data.index.max().strftime('%Y-%m')}")

        return self.fed_data

    def merge_all_data(self):
        """Merge all three datasets on common dates"""
        if any(data is None for data in [self.shiller_data, self.zillow_data, self.fed_data]):
            raise ValueError("All datasets must be loaded first")

        print("Merging all datasets...")

        # Debug: Check date ranges before merging
        print(f"Shiller dates: {self.shiller_data.index.min()} to {self.shiller_data.index.max()}")
        print(f"Zillow dates: {self.zillow_data.index.min()} to {self.zillow_data.index.max()}")
        print(f"Fed dates: {self.fed_data.index.min()} to {self.fed_data.index.max()}")

        # Start with Shiller data as base
        merged = self.shiller_data.copy()
        print(f"After starting with Shiller: {len(merged)} observations")

        # Merge Zillow data
        merged = merged.join(self.zillow_data, how='inner')
        print(f"After adding Zillow: {len(merged)} observations")

        # Merge Fed data
        merged = merged.join(self.fed_data, how='inner')
        print(f"After adding Fed data: {len(merged)} observations")

        # Create additional features if we have data
        if len(merged) > 0:
            merged = self.create_model_features(merged)

            print(f"Merged dataset created: {len(merged)} observations")
            print(
                f"Common date range: {merged.index.min().strftime('%Y-%m')} to {merged.index.max().strftime('%Y-%m')}")
        else:
            print("Failed to create merged dataset - no common dates found")
            return None

        self.merged_data = merged
        return merged

    def create_model_features(self, data, max_lags=12):
        """Create additional features for econometric modeling"""
        print("Creating model features and lagged variables...")
        df = data.copy()

        # Create lagged variables
        for lag in range(1, max_lags + 1):
            df[f'shiller_return_lag{lag}'] = df['shiller_return'].shift(lag)
            df[f'zillow_return_lag{lag}'] = df['zillow_return'].shift(lag)
            df[f'fed_change_lag{lag}'] = df['fed_change'].shift(lag)

        # Volatility measures
        df['shiller_vol_12m'] = df['shiller_return'].rolling(12).std()
        df['zillow_vol_12m'] = df['zillow_return'].rolling(12).std()

        # Price momentum indicators
        df['shiller_ma_12m'] = df['shiller_index'].rolling(12).mean()
        df['zillow_ma_12m'] = df['zillow_index'].rolling(12).mean()

        # Interaction terms
        df['shiller_zillow_spread'] = df['shiller_return'] - df['zillow_return']
        df['fed_shiller_interaction'] = df['fed_change'] * df['shiller_return'].shift(1)
        df['fed_zillow_interaction'] = df['fed_change'] * df['zillow_return'].shift(1)

        print(f"Created {len(df.columns) - len(data.columns)} additional features")
        return df

    def get_analysis_ready_data(self, target='shiller_return', dropna=True):
        """Get clean dataset ready for econometric analysis"""
        if self.merged_data is None:
            raise ValueError("Must merge data first using merge_all_data()")

        analysis_data = self.merged_data.copy()

        if dropna:
            initial_size = len(analysis_data)
            analysis_data = analysis_data.dropna()
            print(f"Removed {initial_size - len(analysis_data)} incomplete observations")

        print(f"Analysis-ready dataset: {len(analysis_data)} complete observations")
        print(f"Target variable: {target}")

        return analysis_data

    def summary_statistics(self):
        """Generate comprehensive summary statistics"""
        if self.merged_data is None:
            raise ValueError("Must merge data first")

        # Key variables for summary
        key_vars = ['shiller_return', 'zillow_return', 'fed_rate', 'fed_change',
                    'shiller_index', 'zillow_index']

        available_vars = [var for var in key_vars if var in self.merged_data.columns]

        summary = self.merged_data[available_vars].describe()

        print("\nSUMMARY STATISTICS")
        print("=" * 60)
        print(summary.round(4))

        # Additional statistics
        print(f"\nCORRELATION MATRIX")
        print("=" * 30)
        corr_matrix = self.merged_data[available_vars].corr()
        print(corr_matrix.round(3))

        return summary, corr_matrix

    def plot_data_overview(self, figsize=(15, 12)):
        """Create comprehensive data visualization"""
        if self.merged_data is None:
            raise ValueError("Must merge data first")

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1. Housing price indices
        axes[0, 0].plot(self.merged_data.index, self.merged_data['shiller_index'],
                        label='Case-Shiller', linewidth=2, color='blue')
        axes[0, 0].plot(self.merged_data.index, self.merged_data['zillow_index'],
                        label='Zillow', linewidth=2, alpha=0.8, color='orange')
        axes[0, 0].set_title('Housing Price Indices', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Housing returns
        axes[0, 1].plot(self.merged_data.index, self.merged_data['shiller_return'],
                        label='Case-Shiller Returns', alpha=0.7, color='blue')
        axes[0, 1].plot(self.merged_data.index, self.merged_data['zillow_return'],
                        label='Zillow Returns', alpha=0.7, color='orange')
        axes[0, 1].set_title('Housing Returns', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Fed funds rate
        axes[1, 0].plot(self.merged_data.index, self.merged_data['fed_rate'],
                        color='red', linewidth=2)
        axes[1, 0].set_title('Federal Funds Rate', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Rate (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Fed rate changes
        axes[1, 1].plot(self.merged_data.index, self.merged_data['fed_change'],
                        color='darkred', alpha=0.7)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Fed Funds Rate Changes', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Change (pp)')
        axes[1, 1].grid(True, alpha=0.3)

        # 5. Rolling volatility
        if 'shiller_vol_12m' in self.merged_data.columns:
            axes[2, 0].plot(self.merged_data.index, self.merged_data['shiller_vol_12m'],
                            label='Case-Shiller Vol', color='blue')
            axes[2, 0].plot(self.merged_data.index, self.merged_data['zillow_vol_12m'],
                            label='Zillow Vol', color='orange')
            axes[2, 0].set_title('12-Month Rolling Volatility', fontsize=12, fontweight='bold')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)

        # 6. Scatter plot: Fed changes vs Housing returns
        axes[2, 1].scatter(self.merged_data['fed_change'], self.merged_data['shiller_return'],
                           alpha=0.6, label='Case-Shiller', color='blue', s=20)
        axes[2, 1].scatter(self.merged_data['fed_change'], self.merged_data['zillow_return'],
                           alpha=0.6, label='Zillow', color='orange', s=20)
        axes[2, 1].set_xlabel('Fed Rate Change')
        axes[2, 1].set_ylabel('Housing Return')
        axes[2, 1].set_title('Fed Changes vs Housing Returns', fontsize=12, fontweight='bold')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig