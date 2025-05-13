import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Ensure the scripts can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model training and forecasting modules
try:
    from chlorophyll_model_training import (
        load_and_preprocess_data, engineer_features, select_features,
        train_models, evaluate_models, analyze_feature_importance,
        save_models, main as train_main
    )
    from chlorophyll_forecasting import ChlorophyllForecaster
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure chlorophyll_model_training.py and chlorophyll_forecasting.py are in the same directory.")
    sys.exit(1)


def train_model(data_path, no_plots=False):
    """
    Train the chlorophyll-a prediction model using the data at data_path

    Parameters:
    -----------
    data_path : str
        Path to the Excel file with water quality data
    no_plots : bool
        Whether to suppress plot creation

    Returns:
    --------
    model_info : dict
        Dictionary with model information
    """
    print("\n===== TRAINING CHLOROPHYLL-A PREDICTION MODEL =====")
    print(f"Using data from: {data_path}")

    if no_plots:
        # Suppress plots
        plt.ioff()
        old_show = plt.show
        plt.show = lambda: None

    # Call the main training function
    model_info = train_main()

    if no_plots:
        # Restore plot functionality
        plt.show = old_show
        plt.ion()

    return model_info


def generate_forecast(forecast_months=18, no_plots=False):
    """
    Generate chlorophyll-a forecasts for all stations

    Parameters:
    -----------
    forecast_months : int
        Number of months to forecast
    no_plots : bool
        Whether to suppress plot creation

    Returns:
    --------
    forecasts : DataFrame
        DataFrame with forecasts for all stations
    """
    print(f"\n===== GENERATING {forecast_months}-MONTH CHLOROPHYLL-A FORECASTS =====")

    # Initialize the forecaster
    forecaster = ChlorophyllForecaster(
        model_path='chlorophyll_gb_model.pkl',
        features_path='selected_features.pkl',
        metadata_path='chlorophyll_gb_model_metadata.pkl',
        original_data_path='merged_stations.xlsx'
    )

    # Run the full forecasting process
    forecasts = forecaster.run_full_forecast(
        forecast_months=forecast_months,
        save_results=True,
        create_plots=not no_plots,
        save_plots=True
    )

    return forecasts


def end_to_end_process(data_path, forecast_months=18, train=True, forecast=True, no_plots=False):
    """
    Run the end-to-end process from data to forecasts

    Parameters:
    -----------
    data_path : str
        Path to the Excel file with water quality data
    forecast_months : int
        Number of months to forecast
    train : bool
        Whether to train the model
    forecast : bool
        Whether to generate forecasts
    no_plots : bool
        Whether to suppress plot creation

    Returns:
    --------
    results : dict
        Dictionary with model and forecast information
    """
    results = {}

    # Train the model if requested
    if train:
        model_info = train_model(data_path, no_plots)
        results['model_info'] = model_info

    # Generate forecasts if requested
    if forecast:
        forecasts = generate_forecast(forecast_months, no_plots)
        results['forecasts'] = forecasts

    return results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Chlorophyll-a Prediction and Forecasting System')

    parser.add_argument('--data', '-d', dest='data_path', default='merged_stations.xlsx',
                        help='Path to the Excel file with water quality data')

    parser.add_argument('--months', '-m', dest='forecast_months', type=int, default=18,
                        help='Number of months to forecast')

    parser.add_argument('--train-only', dest='train_only', action='store_true',
                        help='Only train the model, do not generate forecasts')

    parser.add_argument('--forecast-only', dest='forecast_only', action='store_true',
                        help='Only generate forecasts, do not train the model')

    parser.add_argument('--no-plots', dest='no_plots', action='store_true',
                        help='Suppress plot creation')

    parser.add_argument('--output', '-o', dest='output_dir', default='.',
                        help='Directory to save output files')

    return parser.parse_args()


def main():
    """Main function to run the command-line interface"""
    # Parse command line arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Change to the output directory
    os.chdir(args.output_dir)

    # Determine which parts to run
    train = not args.forecast_only
    forecast = not args.train_only

    # Run the end-to-end process
    start_time = datetime.now()
    print(f"Starting chlorophyll-a prediction and forecasting at {start_time}")

    results = end_to_end_process(
        data_path=args.data_path,
        forecast_months=args.forecast_months,
        train=train,
        forecast=forecast,
        no_plots=args.no_plots
    )

    end_time = datetime.now()
    print(f"Process completed at {end_time}")
    print(f"Total runtime: {end_time - start_time}")

    # Print summary of results
    if 'model_info' in results:
        print("\nModel training completed.")
        if results['model_info'] and 'gb_r2' in results['model_info']:
            print(f"Gradient Boosting R²: {results['model_info']['gb_r2']:.4f}")

    if 'forecasts' in results:
        print("\nForecasting completed.")
        if results['forecasts'] is not None:
            print(
                f"Generated {len(results['forecasts'])} forecast points across {results['forecasts']['Station'].nunique()} stations")

    return results


if __name__ == "__main__":
    main()