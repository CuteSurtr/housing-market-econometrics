import pandas as pd
import numpy as np
import requests
from io import StringIO


def test_github_data_loading():
    """Test loading data from your GitHub repository"""

    # GitHub URLs for your data
    urls = {
        'shiller': 'https://raw.githubusercontent.com/CuteSurtr/housing-market-econometrics/main/CSUSHPINSA.csv',
        'fed': 'https://raw.githubusercontent.com/CuteSurtr/housing-market-econometrics/main/fed_rate_clean_2000_2024(1).csv',
        'zillow': 'https://raw.githubusercontent.com/CuteSurtr/housing-market-econometrics/main/housing_data_filtered_regions.csv'
    }

    for name, url in urls.items():
        try:
            response = requests.get(url)
            df = pd.read_csv(StringIO(response.text))
            print(f"✓ {name.title()} data: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")

    print("Data loading test complete!")


if __name__ == "__main__":
    test_github_data_loading()