import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle
import warnings

warnings.filterwarnings('ignore')


class ChlorophyllForecaster:
    """
    Class for generating chlorophyll-a forecasts based on trained models
    """

    def __init__(self, model_path=None, features_path=None, metadata_path=None, original_data_path=None):
        """
        Initialize the forecaster with model paths and data

        Parameters:
        -----------
        model_path : str
            Path to the saved model pickle file
        features_path : str
            Path to the saved selected features pickle file
        metadata_path : str
            Path to the saved model metadata pickle file
        original_data_path : str
            Path to the original data Excel file
        """
        self.model_path = model_path or 'chlorophyll_gb_model.pkl'
        self.features_path = features_path or 'selected_features.pkl'
        self.metadata_path = metadata_path or 'chlorophyll_gb_model_metadata.pkl'
        self.original_data_path = original_data_path or 'merged_stations.xlsx'

        self.model = None
        self.selected_features = None
        self.metadata = None
        self.original_df = None
        self.stations = None
        self.station_monthly_avgs = {}
        self.station_encoder = None
        self.target = 'Chlorophyll-a (ug/L)'

        # Base features used in the model
        self.base_features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                              'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature']

    def load_resources(self):
        """
        Load all necessary resources: model, features, metadata, and original data
        """
        print("\n=== Loading resources ===")

        # Load model
        try:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

        # Load selected features
        try:
            with open(self.features_path, 'rb') as file:
                self.selected_features = pickle.load(file)
            print(f"Selected features loaded from {self.features_path}")
        except Exception as e:
            print(f"Error loading selected features: {e}")
            return False

        # Load metadata
        try:
            with open(self.metadata_path, 'rb') as file:
                self.metadata = pickle.load(file)
            print(f"Model metadata loaded from {self.metadata_path}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            # Continue even if metadata cannot be loaded

        # Load original data
        try:
            self.original_df = pd.read_excel(self.original_data_path)
            print(f"Original data loaded from {self.original_data_path}")

            # Convert Date to datetime
            if 'Date' in self.original_df.columns:
                self.original_df['Date'] = pd.to_datetime(self.original_df['Date'])
            else:
                print("Warning: No 'Date' column found in the Excel file. Using index as date.")
                self.original_df['Date'] = pd.date_range(start='2020-01-01', periods=len(self.original_df), freq='D')

            # Convert all numeric columns to float, handling errors
            print("Converting columns to appropriate data types...")
            for col in self.base_features + [self.target]:
                if col in self.original_df.columns:
                    try:
                        self.original_df[col] = pd.to_numeric(self.original_df[col], errors='coerce')
                        print(f"  - Converted '{col}' to numeric type")
                    except Exception as e:
                        print(f"  - Warning: Could not convert '{col}' to numeric: {e}")

            # Extract stations
            if 'Station' in self.original_df.columns:
                self.stations = self.original_df['Station'].unique()
                print(f"Found {len(self.stations)} unique stations: {self.stations}")

                # Initialize the station encoder - using sparse_output instead of sparse for newer sklearn versions
                try:
                    # Try the newer parameter name first
                    self.station_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    self.station_encoder.fit(self.original_df[['Station']])
                    print(f"Station encoder initialized with categories: {self.station_encoder.categories_}")
                except TypeError:
                    # Fall back to the older parameter name
                    self.station_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    self.station_encoder.fit(self.original_df[['Station']])
                    print(f"Station encoder initialized with categories: {self.station_encoder.categories_}")
            else:
                print("Error: 'Station' column not found in the Excel file.")
                return False

        except Exception as e:
            print(f"Error loading original data: {e}")
            return False

        return True

    def calculate_station_monthly_averages(self):
        """
        Calculate monthly averages for each feature by station
        """
        print("\n=== Calculating station-specific monthly averages ===")

        for station in self.stations:
            station_df = self.original_df[self.original_df['Station'] == station]

            # Skip if station has no data
            if len(station_df) == 0:
                print(f"Warning: No data for station {station}, skipping...")
                continue

            self.station_monthly_avgs[station] = {}

            # Calculate monthly averages for features and target for this station
            for feature in self.base_features + [self.target]:
                if feature in station_df.columns:
                    try:
                        # Calculate monthly averages for this station, skipping NaN values
                        monthly_avg = station_df.dropna(subset=[feature]).groupby(station_df['Date'].dt.month)[
                            feature].mean()
                        self.station_monthly_avgs[station][feature] = monthly_avg

                        # If there are months with no data, fill them using overall average for that month
                        if len(monthly_avg) < 12:
                            print(f"  - Warning: Station {station} missing data for some months for {feature}")
                            # Get overall monthly average across all stations
                            overall_monthly_avg = self.original_df.groupby(self.original_df['Date'].dt.month)[
                                feature].mean(skipna=True)

                            # Fill missing months with overall average
                            for month in range(1, 13):
                                if month not in monthly_avg.index:
                                    if month in overall_monthly_avg.index:
                                        self.station_monthly_avgs[station][feature][month] = overall_monthly_avg[month]
                                    else:
                                        # If still no data, use overall average
                                        self.station_monthly_avgs[station][feature][month] = station_df[feature].mean(
                                            skipna=True)
                    except Exception as e:
                        print(f"  - Error calculating monthly averages for {station}, {feature}: {e}")
                        print("    Trying alternative approach...")
                        # Use a more robust approach
                        try:
                            # Create a month column
                            station_df_month = station_df.copy()
                            station_df_month['Month'] = station_df['Date'].dt.month

                            # Group manually and calculate mean
                            monthly_values = {}
                            for month in range(1, 13):
                                month_data = station_df_month[station_df_month['Month'] == month][feature].dropna()
                                if len(month_data) > 0:
                                    monthly_values[month] = month_data.astype(float).mean()

                            # Convert to Series
                            if monthly_values:
                                monthly_avg = pd.Series(monthly_values)
                                self.station_monthly_avgs[station][feature] = monthly_avg
                            else:
                                # Fallback to overall mean
                                overall_mean = self.original_df[feature].dropna().astype(float).mean()
                                self.station_monthly_avgs[station][feature] = pd.Series(
                                    {month: overall_mean for month in range(1, 13)})
                        except Exception as e2:
                            print(f"    Second approach also failed: {e2}")
                            # Use a constant value as last resort
                            try:
                                overall_mean = self.original_df[feature].dropna().astype(float).mean()
                                if pd.isna(overall_mean):
                                    overall_mean = 0
                            except:
                                overall_mean = 0
                            self.station_monthly_avgs[station][feature] = pd.Series(
                                {month: overall_mean for month in range(1, 13)})

        return True

    def generate_forecasts(self, forecast_months=18, frequency='M'):
        """
        Generate chlorophyll-a forecasts for all stations

        Parameters:
        -----------
        forecast_months : int
            Number of months to forecast
        frequency : str
            Frequency of forecasts ('M' for monthly, 'W' for weekly, 'D' for daily)

        Returns:
        --------
        combined_forecast : DataFrame
            DataFrame with forecasts for all stations
        """
        # Verify resources are loaded
        if self.model is None or self.selected_features is None or self.stations is None:
            print("Error: Resources not loaded. Run load_resources() first.")
            return None

        print(f"\n=== Generating {forecast_months} month forecasts for all stations ===")

        # Calculate station monthly averages if not already done
        if not self.station_monthly_avgs:
            self.calculate_station_monthly_averages()

        # Find the last date in our dataset
        last_date = self.original_df['Date'].max()
        print(f"Last date in dataset: {last_date.date()}")

        # Generate dates for the forecast period
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=forecast_months, freq=frequency)
        print(f"Forecasting from {forecast_dates.min().date()} to {forecast_dates.max().date()}")

        all_forecasts = []

        # For each station
        for station in self.stations:
            print(f"\nGenerating forecasts for station: {station}")

            # Create a future DataFrame for this station with all dates
            future_station_df = pd.DataFrame({
                'Date': forecast_dates,
                'Station': station
            })

            # Number of predictions for this station
            num_predictions = len(future_station_df)
            print(f"Creating {num_predictions} predictions for station {station}")

            # Extract month and seasonal features from the future dates
            future_station_df['Month'] = future_station_df['Date'].dt.month
            future_station_df['DayOfYear'] = future_station_df['Date'].dt.dayofyear
            future_station_df['Season_sin'] = np.sin(future_station_df['DayOfYear'] * (2 * np.pi / 365))
            future_station_df['Season_cos'] = np.cos(future_station_df['DayOfYear'] * (2 * np.pi / 365))

            # Create month dummy variables
            month_dummies = pd.get_dummies(future_station_df['Month'], prefix='month')
            future_station_df = pd.concat([future_station_df, month_dummies], axis=1)

            # Transform the Station categorical feature using one-hot encoding
            try:
                # First, create a DataFrame with just the Station column
                station_data = future_station_df[['Station']].copy()

                # Apply one-hot encoding
                station_encoded = self.station_encoder.transform(station_data)

                # Get the feature names - handle both older and newer sklearn versions
                try:
                    # Newer sklearn versions with get_feature_names_out
                    station_feature_names = self.station_encoder.get_feature_names_out(['Station'])
                except AttributeError:
                    # Older sklearn versions
                    station_feature_names = [f'station_{cat}' for cat in self.station_encoder.categories_[0]]

                # Create a DataFrame with the encoded features
                station_encoded_df = pd.DataFrame(station_encoded, columns=station_feature_names,
                                                  index=future_station_df.index)

                # Add the encoded station features to the future dataframe
                future_station_df = pd.concat([future_station_df, station_encoded_df], axis=1)
                print(f"  - Station encoded as categorical feature with {len(station_feature_names)} categories")
            except Exception as e:
                print(f"  - Error encoding station feature: {e}")
                # Create a fallback encoding method using pandas get_dummies for greater compatibility
                print("    Falling back to pandas get_dummies for station encoding")
                station_dummies = pd.get_dummies(future_station_df['Station'], prefix='station')
                future_station_df = pd.concat([future_station_df, station_dummies], axis=1)
                print(f"  - Station encoded with pandas get_dummies method: {station_dummies.shape[1]} categories")

            # Apply monthly averages to the future dataframe for this station
            for feature in self.base_features:
                if feature in self.station_monthly_avgs[station]:
                    future_station_df[feature] = future_station_df['Month'].map(
                        self.station_monthly_avgs[station][feature])

            # Get most recent values from historical data for this station
            station_recent_df = self.original_df[self.original_df['Station'] == station].sort_values('Date')

            # If this station has data, get recent values
            recent_values = {}
            if len(station_recent_df) > 0:
                for feature in self.base_features + [self.target]:
                    if feature in station_recent_df.columns:
                        # Convert to numeric and drop NaN values
                        feature_values = pd.to_numeric(station_recent_df[feature], errors='coerce').dropna()
                        if len(feature_values) >= 3:
                            recent_values[feature] = feature_values.iloc[-3:].values  # Get last 3 values
                        elif len(feature_values) > 0:
                            # Use whatever data we have (less than 3 points)
                            recent_values[feature] = feature_values.iloc[-len(feature_values):].values
                        else:
                            # Use zeros as fallback
                            recent_values[feature] = np.zeros(3)
                    else:
                        # Use zeros as fallback
                        recent_values[feature] = np.zeros(3)
            else:
                # If no data for this station, initialize with zeros
                for feature in self.base_features + [self.target]:
                    recent_values[feature] = np.zeros(3)

            # Initialize array to store predictions for this station
            model_future_preds = np.zeros(len(future_station_df))

            # Iterative forecasting - one step at a time
            print(f"Generating step-by-step forecasts for station {station}...")
            for i in range(len(future_station_df)):
                # Create lag features based on recent predictions or historical values
                for feature in self.base_features + [self.target]:
                    if feature in self.original_df.columns and feature in self.station_monthly_avgs[station]:
                        # For the first few predictions, use historical data
                        for lag in [1, 2, 3]:
                            lag_col = f'{feature}_lag{lag}'
                            if lag <= i:
                                # Use previously predicted values for future steps
                                if feature == self.target:
                                    future_station_df.loc[future_station_df.index[i], lag_col] = model_future_preds[
                                        i - lag]
                                else:
                                    # Use the corresponding month from previous year as an approximation
                                    month_val = self.station_monthly_avgs[station][feature][
                                        future_station_df.loc[future_station_df.index[i], 'Month']]
                                    future_station_df.loc[future_station_df.index[i], lag_col] = month_val
                            else:
                                # Use historical values for initial lags
                                idx = -lag + i
                                if idx >= -len(recent_values[feature]):
                                    future_station_df.loc[future_station_df.index[i], lag_col] = recent_values[feature][
                                        idx]
                                else:
                                    # Not enough history, use monthly average
                                    month_val = self.station_monthly_avgs[station][feature][
                                        future_station_df.loc[future_station_df.index[i], 'Month']]
                                    future_station_df.loc[future_station_df.index[i], lag_col] = month_val

                # Create rolling means and stds
                # For simplicity, we'll approximate these with the monthly averages initially
                for feature in self.base_features + [self.target]:
                    if feature in self.original_df.columns and feature in self.station_monthly_avgs[station]:
                        for window in [3, 7, 14]:
                            mean_col = f'{feature}_roll_mean{window}'
                            std_col = f'{feature}_roll_std{window}'

                            # For initial predictions, use average of historical values for that month
                            month_val = self.station_monthly_avgs[station][feature][
                                future_station_df.loc[future_station_df.index[i], 'Month']]
                            future_station_df.loc[future_station_df.index[i], mean_col] = month_val

                            # For std, use the std of that feature in the historical dataset for the station & month
                            month_data = pd.to_numeric(station_recent_df[station_recent_df['Date'].dt.month ==
                                                                         future_station_df.loc[
                                                                             future_station_df.index[i], 'Month']][
                                                           feature],
                                                       errors='coerce').dropna()

                            if len(month_data) > 1:
                                month_std = month_data.std()
                                future_station_df.loc[future_station_df.index[i], std_col] = month_std if not np.isnan(
                                    month_std) else 0
                            else:
                                # Not enough data for std calculation, use overall std
                                all_data = pd.to_numeric(station_recent_df[feature], errors='coerce').dropna()
                                future_station_df.loc[future_station_df.index[i], std_col] = all_data.std() if len(
                                    all_data) > 1 else 0

                # Create interaction features
                for j, feat1 in enumerate(self.base_features):
                    if feat1 in future_station_df.columns:
                        for feat2 in self.base_features[j + 1:]:
                            if feat2 in future_station_df.columns:
                                interaction_name = f'{feat1.split(" ")[0]}_{feat2.split(" ")[0]}_interact'
                                future_station_df.loc[future_station_df.index[i], interaction_name] = \
                                    future_station_df.loc[future_station_df.index[i], feat1] * \
                                    future_station_df.loc[future_station_df.index[i], feat2]

                # Create polynomial features
                for feature in self.base_features:
                    if feature in future_station_df.columns:
                        future_station_df.loc[future_station_df.index[i], f'{feature}_squared'] = \
                            future_station_df.loc[future_station_df.index[i], feature] ** 2

                # Create a feature vector for this future date
                row_idx = future_station_df.index[i]
                # First, create dict with all selected features initialized to NaN
                current_features = {col: np.nan for col in self.selected_features}

                # Then fill in the values we have
                for col in self.selected_features:
                    if col in future_station_df.columns:
                        current_features[col] = future_station_df.loc[row_idx, col]

                # Convert to DataFrame with the right structure
                X_future_i = pd.DataFrame([current_features])

                # Fill any missing values with medians from training data or zeros
                for col in X_future_i.columns:
                    if X_future_i[col].isna().any():
                        X_future_i[col] = X_future_i[col].fillna(0)

                # Make predictions for this step
                try:
                    # Check that all required features are available
                    missing_features = [feat for feat in self.selected_features if feat not in X_future_i.columns]

                    if missing_features:
                        print(f"  - Warning: Missing required features: {missing_features}")
                        # Add the missing features with zeros to handle this gracefully
                        for feat in missing_features:
                            X_future_i[feat] = 0
                        print("    Added missing features with zero values")

                    # Ensure features are in the right order
                    X_features = X_future_i[self.selected_features]

                    # Make prediction
                    model_pred = self.model.predict(X_features)[0]

                    # Convert back from log if needed
                    if self.metadata and self.metadata.get('use_log_transform', False):
                        model_pred = np.expm1(model_pred)

                    # Cap negative predictions at 0 (can't have negative chlorophyll)
                    model_pred = max(0, model_pred)

                    # Store predictions
                    model_future_preds[i] = model_pred

                except Exception as e:
                    print(f"Error making prediction for station {station}, step {i + 1}: {e}")
                    # Use last valid prediction or 0
                    model_pred = model_future_preds[i - 1] if i > 0 else 0
                    model_future_preds[i] = model_pred

            # Create a DataFrame with the results for this station
            forecast_results = pd.DataFrame({
                'Date': future_station_df['Date'],
                'Station': station,
                'Month': future_station_df['Date'].dt.month,
                'Month_Name': future_station_df['Date'].dt.month_name(),
                'Predicted_Chlorophyll': model_future_preds
            })

            print(f"Completed forecasts for station {station}")
            all_forecasts.append(forecast_results)

        # Combine all station forecasts
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)

        print("\nAll Station Forecasts Summary:")
        summary = combined_forecast.groupby(['Month_Name', 'Station'])['Predicted_Chlorophyll'].mean().unstack()
        print(summary)

        return combined_forecast

    def save_forecasts(self, combined_forecast, base_filename='chlorophyll_forecast'):
        """
        Save forecasts to CSV files

        Parameters:
        -----------
        combined_forecast : DataFrame
            DataFrame with forecasts for all stations
        base_filename : str
            Base filename for the CSV output
        """
        # Save the forecast to CSV
        combined_forecast.to_csv(f'{base_filename}.csv', index=False)
        print(f"Forecast saved to '{base_filename}.csv'")

        # Create a wide format version (one row per date, columns for each station)
        wide_forecast = combined_forecast.pivot(index='Date', columns='Station', values='Predicted_Chlorophyll')
        wide_forecast.reset_index(inplace=True)
        wide_forecast.to_csv(f'{base_filename}_wide.csv', index=False)
        print(f"Wide format forecast saved to '{base_filename}_wide.csv'")

    def plot_forecasts(self, combined_forecast, save_plots=False, plot_dir='forecast_plots'):
        """
        Create visualizations of the forecasts

        Parameters:
        -----------
        combined_forecast : DataFrame
            DataFrame with forecasts for all stations
        save_plots : bool
            Whether to save the plots to files
        plot_dir : str
            Directory to save plots if save_plots is True
        """
        import os

        # Create directory for plots if saving
        if save_plots and not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Plot the historical data and the forecast for each station
        for station in self.stations:
            plt.figure(figsize=(15, 8))

            # Get historical data for this station
            station_hist = self.original_df[self.original_df['Station'] == station]

            # Get forecast for this station
            station_forecast = combined_forecast[combined_forecast['Station'] == station]

            if len(station_hist) > 0:  # Only plot if we have historical data
                # Plot historical data
                plt.plot(station_hist['Date'], station_hist[self.target], 'b.-',
                         label='Historical Data', alpha=0.7)

                # Plot forecast
                plt.plot(station_forecast['Date'], station_forecast['Predicted_Chlorophyll'], 'r.--',
                         label='Forecast', markersize=8, alpha=0.7)

                # Add shaded area to indicate forecast region
                if len(station_hist) > 0 and len(station_forecast) > 0:
                    plt.axvspan(station_hist['Date'].max(), station_forecast['Date'].max(),
                                alpha=0.2, color='gray', label='Forecast Period')

                plt.title(f'Station {station}: Chlorophyll-a Historical Data and Forecast', fontsize=16)
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Chlorophyll-a (ug/L)', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)

                # Improve x-axis date formatting
                plt.gcf().autofmt_xdate()

                plt.tight_layout()

                if save_plots:
                    plt.savefig(f"{plot_dir}/station_{station.replace(' ', '_')}_forecast.png")

                plt.show()

        # Create a heatmap visualization of forecasts across all stations
        pivot_data = combined_forecast.pivot_table(
            index='Date',
            columns='Station',
            values='Predicted_Chlorophyll',
            aggfunc='mean'
        )

        plt.figure(figsize=(16, 10))
        sns.heatmap(pivot_data, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=.5)
        plt.title('Forecasted Chlorophyll-a Levels Across All Stations', fontsize=16)
        plt.ylabel('Date', fontsize=14)
        plt.xlabel('Station', fontsize=14)
        plt.tight_layout()

        if save_plots:
            plt.savefig(f"{plot_dir}/heatmap_all_stations.png")

        plt.show()

        # For each station, plot a line
        plt.figure(figsize=(16, 10))

        # For each station, plot a line
        for station in self.stations:
            station_data = combined_forecast[combined_forecast['Station'] == station]
            if len(station_data) > 0:
                plt.plot(station_data['Date'], station_data['Predicted_Chlorophyll'],
                         'o-', label=f'Station {station}', linewidth=2, markersize=4)

        plt.title('Comparison of Forecasted Chlorophyll-a Across All Stations', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Chlorophyll-a (ug/L)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        if save_plots:
            plt.savefig(f"{plot_dir}/line_all_stations.png")

        plt.show()

    def save_forecast_package(self, combined_forecast, last_date, forecast_dates,
                              filename='chlorophyll_station_forecast_package.pkl'):
        """
        Save the complete forecast package for future use

        Parameters:
        -----------
        combined_forecast : DataFrame
            DataFrame with forecasts for all stations
        last_date : datetime
            Last date in the historical dataset
        forecast_dates : DatetimeIndex
            DatetimeIndex of forecast dates
        filename : str
            Filename for the forecast package
        """
        print("\n=== Saving forecast package ===")

        forecast_package = {
            'model': self.model,
            'selected_features': self.selected_features,
            'forecast_results': combined_forecast,
            'last_training_date': last_date,
            'forecast_dates': forecast_dates,
            'use_log_transform': self.metadata.get('use_log_transform', False) if self.metadata else False,
            'station_monthly_stats': self.station_monthly_avgs,
            'station_encoder': self.station_encoder
        }

        with open(filename, 'wb') as file:
            pickle.dump(forecast_package, file)

        print(f"Forecast package saved as '{filename}'")

    def run_full_forecast(self, forecast_months=18, save_results=True, create_plots=True, save_plots=False):
        """
        Run the complete forecasting process from start to finish

        Parameters:
        -----------
        forecast_months : int
            Number of months to forecast
        save_results : bool
            Whether to save the forecast results to CSV
        create_plots : bool
            Whether to create plots of the forecasts
        save_plots : bool
            Whether to save the plots to files

        Returns:
        --------
        combined_forecast : DataFrame
            DataFrame with forecasts for all stations
        """
        # Load all resources
        if not self.load_resources():
            print("Error loading resources. Aborting forecast.")
            return None

        # Calculate station monthly averages
        self.calculate_station_monthly_averages()

        # Generate forecasts
        combined_forecast = self.generate_forecasts(forecast_months=forecast_months)

        if combined_forecast is None:
            print("Error generating forecasts. Aborting.")
            return None

        # Save forecasts if requested
        if save_results:
            self.save_forecasts(combined_forecast, base_filename=f'chlorophyll_{forecast_months}month_station_forecast')

        # Create plots if requested
        if create_plots:
            self.plot_forecasts(combined_forecast, save_plots=save_plots)

        # Save forecast package
        if save_results:
            last_date = self.original_df['Date'].max()
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                           periods=forecast_months, freq='M')
            self.save_forecast_package(combined_forecast, last_date, forecast_dates)

        print("\n=== Forecasting process complete! ===")

        return combined_forecast


# Example usage
if __name__ == "__main__":
    # Initialize the forecaster
    forecaster = ChlorophyllForecaster(
        model_path='chlorophyll_gb_model.pkl',
        features_path='selected_features.pkl',
        metadata_path='chlorophyll_gb_model_metadata.pkl',
        original_data_path='merged_stations.xlsx'
    )

    # Run the full forecasting process
    forecasts = forecaster.run_full_forecast(forecast_months=18, save_results=True, create_plots=True, save_plots=True)