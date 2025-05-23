import os

os.environ['PYTENSOR_FLAGS'] = 'cxx='

# Import your data processor
from housing_data_processor import HousingDataProcessor  # Your existing class

# Import all econometric models
from gjr_garch_model import fit_gjr_garch_housing
from regime_switching_model import fit_regime_switching_housing
from merton_jump_model import fit_merton_model_housing
from transfer_function_model import fit_transfer_function_housing


def run_complete_analysis():
    # Load data
    processor = HousingDataProcessor()
    processor.load_case_shiller_data()
    processor.load_zillow_data()
    processor.load_fed_data()
    processor.merge_all_data()
    data = processor.get_analysis_ready_data()

    # Run each model
    print("Running GJR-GARCH...")
    gjr_model = fit_gjr_garch_housing(
        data['shiller_return'],
        data[['fed_change', 'fed_vol']]
    )

    print("\nRunning Regime Switching...")
    regime_model = fit_regime_switching_housing(
        data['shiller_return'],
        data[['fed_change', 'fed_level']]
    )

    print("\nRunning Jump-Diffusion...")
    merton_model = fit_merton_model_housing(data['shiller_index'])

    print("\nRunning Transfer Function...")
    tf_model = fit_transfer_function_housing(
        data['shiller_return'],
        data['fed_change']
    )

    return gjr_model, regime_model, merton_model, tf_model


if __name__ == "__main__":
    models = run_complete_analysis()