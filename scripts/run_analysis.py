#!/usr/bin/env python3
"""
Script to run the complete housing market econometric analysis pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main_analysis import run_complete_analysis
from housing_data_processor import HousingDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the complete analysis pipeline."""
    try:
        logger.info("Starting housing market econometric analysis...")
        
        # Run the complete analysis
        models = run_complete_analysis()
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Generated {len(models)} models:")
        
        for i, model_name in enumerate(['GJR-GARCH', 'Regime Switching', 'Jump Diffusion', 'Transfer Function']):
            logger.info(f"  {i+1}. {model_name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 