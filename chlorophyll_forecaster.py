import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings


warnings.filterwarnings('ignore')

# ============= LOAD AND PREPARE DATA =============
def load_existing_model_and_features():
    """
    Load the saved model, features, and metadata
    """
    print("Loading saved model and features...")

    # Load the model
    with open('pkl/chlorophyll_gb_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load selected features
    with open('pkl/selected_features.pkl', 'rb') as file:
        selected_features = pickle.load(file)

    # Load model metadata
    with open('pkl/chlorophyll_gb_model_metadata.pkl', 'rb') as file:
        metadata = pickle.load(file)

    print(f"Loaded model (RÂ² = {metadata['r2_score']:.4f}, RMSE = {metadata['rmse']:.4f})")
    return model, selected_features, metadata


def load_and_preprocess_data(data, features=None, target='Chlorophyll-a (ug/L)'):
    """
    Re-implementation of the original load_and_preprocess_data function
    Modified to accept either a file path or a DataFrame
    """
    print("\n=== Loading and preprocessing data ===")

    # Load Excel file or use provided DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        print("Using provided DataFrame")
    else:
        # Load from file path
        df = pd.read_excel(data)
        print("Loaded data from file")

    # Display DataFrame dimensions
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    # Remove columns with too many missing values (using a threshold of 20%)
    threshold = len(df) * 0.2
    df = df.drop(columns=[col for col in df.columns if
                          "Unnamed" in col or
                          df[col].isnull().sum() > threshold])

    # Handle pH conversion
    if 'pH (units)' in df.columns:
        df['pH (units)'] = pd.to_numeric(df['pH (units)'], errors='coerce')
        # Check for invalid pH values (outside range 0-14)
        invalid_ph = ((df['pH (units)'] < 0) | (df['pH (units)'] > 14)).sum()
        if invalid_ph > 0:
            print(f"Detected {invalid_ph} invalid pH values outside range 0-14")
            # Replace invalid values with median
            valid_ph = df.loc[(df['pH (units)'] >= 0) & (df['pH (units)'] <= 14), 'pH (units)']
            if not valid_ph.empty:
                median_ph = valid_ph.median()
                df.loc[(df['pH (units)'] < 0) | (df['pH (units)'] > 14), 'pH (units)'] = median_ph

    # Only drop rows with missing target values
    df = df.dropna(subset=[target])

    # Extract station information
    if 'Station' in df.columns:
        print("Processing station information...")
        # One-hot encode the Station column
        station_dummies = pd.get_dummies(df['Station'], prefix='station')
        df = pd.concat([df, station_dummies], axis=1)
        print(f"Created {station_dummies.shape[1]} station dummy variables")

    # Ensure data is sorted by date if a date column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        print("Sorted data by Date")

    # Set features if not provided
    if features is None:
        features = [col for col in df.select_dtypes(include=np.number).columns
                    if col != target]

    # Time-aware missing value handling
    print("\n=== Handling missing values with time-aware methods ===")
    cols_with_na = df.columns[df.isna().any()].tolist()
    print(f"Columns with missing values: {cols_with_na}")

    # If date column exists, use time-based interpolation methods
    if 'Date' in df.columns:
        # Create a temporary copy with Date as index for interpolation
        temp_df = df.set_index('Date')

        # For time series variables, use interpolation and fill methods
        for col in temp_df.select_dtypes(include=np.number).columns:
            if temp_df[col].isna().any():
                # Store original NaN locations and create a copy to track changes
                na_mask_original = temp_df[col].isna()
                na_count_before = na_mask_original.sum()

                # Create a copy of the column with NaNs to track which method fills each value
                fill_method_col = pd.Series(index=temp_df.index, dtype='object')
                fill_method_col[na_mask_original] = 'unfilled'

                # 1. Linear interpolation
                temp_df[col] = temp_df[col].interpolate(method='linear')
                # Mark values that were filled by interpolation
                interp_filled_mask = na_mask_original & ~temp_df[col].isna()
                fill_method_col[interp_filled_mask] = 'linear interpolation'

                # 2. Forward fill for any remaining NaNs
                na_before_ffill = temp_df[col].isna()
                temp_df[col] = temp_df[col].fillna(method='ffill')
                # Mark values that were filled by forward fill
                ffill_mask = na_before_ffill & ~temp_df[col].isna()
                fill_method_col[ffill_mask] = 'forward fill'

                # 3. Backward fill for any remaining NaNs
                na_before_bfill = temp_df[col].isna()
                temp_df[col] = temp_df[col].fillna(method='bfill')
                # Mark values that were filled by backward fill
                bfill_mask = na_before_bfill & ~temp_df[col].isna()
                fill_method_col[bfill_mask] = 'backward fill'

                # 4. Use median for any still-missing values
                remaining_na = temp_df[col].isna().sum()
                if remaining_na > 0:
                    na_before_median = temp_df[col].isna()
                    median_val = temp_df[col].median()
                    temp_df[col] = temp_df[col].fillna(median_val)
                    # Mark values that were filled by median
                    median_mask = na_before_median & ~temp_df[col].isna()
                    fill_method_col[median_mask] = 'median fill'

                # Count and print methods used
                method_counts = fill_method_col.value_counts().to_dict()
                if 'unfilled' in method_counts:  # Should not happen, but just in case
                    del method_counts['unfilled']

                print(f"Imputed {na_count_before} missing values in '{col}':")
                for method, count in method_counts.items():
                    print(f"  - {count} values using {method}")

        # Copy the imputed values back to the original dataframe
        for col in temp_df.columns:
            df.loc[:, col] = temp_df[col].values

        # Reset Date as a regular column
        df.reset_index(inplace=True)
    else:
        # Without dates, fall back to median imputation for now
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isna().any():
                na_count = df[col].isna().sum()
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"Imputed {na_count} missing values in '{col}' with median: {median_val:.4f}")

    # Drop all remaining object/string columns that can't be converted to numeric
    print("\nDropping remaining non-numeric columns:")
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Date' and col != 'Station':  # Keep Date and Station
            print(f"  - Dropping: {col}")
            df = df.drop(columns=[col])

    return df


def engineer_features(df, features, target):
    """
    Re-implementation of the original engineer_features function
    """
    print("\n=== Creating enhanced features for peak detection ===")

    # Create lagged features with more lag options
    for feature in features + [target]:
        for lag in [1, 2, 3, 5, 9]:  # Added 3, 5, and 9 for capturing longer patterns
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

    # Create rolling window features with various statistics
    for feature in features + [target]:
        for window in [3, 7, 14]:
            # Mean captures trends
            df[f'{feature}_roll_mean{window}'] = df[feature].rolling(window=window).mean()
            # Max helps identify recent peaks
            df[f'{feature}_roll_max{window}'] = df[feature].rolling(window=window).max()
            # Standard deviation captures volatility
            df[f'{feature}_roll_std{window}'] = df[feature].rolling(window=window).std()

    # Create rate-of-change features - crucial for detecting rapid changes
    for feature in features + [target]:
        # First-order differences (daily change)
        df[f'{feature}_diff1'] = df[feature].diff()
        # Percentage change often better captures relative movements
        df[f'{feature}_pct_change'] = df[feature].pct_change()
        # 3-sample change rate
        df[f'{feature}_diff3'] = df[feature].diff(3)
        # 9-sample change rate (approximately monthly for your data)
        df[f'{feature}_diff9'] = df[feature].diff(9)

    # Create interaction features between key water quality parameters
    # Algal blooms often result from combinations of factors
    for i, feat1 in enumerate(features[:min(3, len(features))]):
        for feat2 in features[i + 1:min(i + 3, len(features))]:
            df[f'{feat1}_{feat2}_interact'] = df[feat1] * df[feat2]

    # Enhanced seasonality if Date column exists
    if 'Date' in df.columns:
        print("Creating enhanced seasonal features...")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['DayOfMonth'] = df['Date'].dt.day
        df['Year'] = df['Date'].dt.year

        # Annual cycle
        df['Season_annual_sin'] = np.sin(df['DayOfYear'] * (2 * np.pi / 365.25))
        df['Season_annual_cos'] = np.cos(df['DayOfYear'] * (2 * np.pi / 365.25))

        # Quarterly cycle (approximately 91 days)
        df['Season_quarterly_sin'] = np.sin(df['DayOfYear'] * (2 * np.pi / 91))
        df['Season_quarterly_cos'] = np.cos(df['DayOfYear'] * (2 * np.pi / 91))

        # Monthly cycle
        df['Season_monthly_sin'] = np.sin(df['DayOfMonth'] * (2 * np.pi / 30.44))
        df['Season_monthly_cos'] = np.cos(df['DayOfMonth'] * (2 * np.pi / 30.44))

        # If you have 9 samples per month, create sample sequence features
        try:
            # Create a month-year identifier
            df['MonthYear'] = df['Date'].dt.to_period('M')

            # Group by month-year and add sample sequence
            df['SampleInMonthSeq'] = df.groupby('MonthYear').cumcount() + 1

            # Normalize to 0-1 range and create trigonometric features
            df['SampleInMonth_sin'] = np.sin(df['SampleInMonthSeq'] * (2 * np.pi / 9))
            df['SampleInMonth_cos'] = np.cos(df['SampleInMonthSeq'] * (2 * np.pi / 9))

            # Drop the helper columns
            df = df.drop(columns=['MonthYear'])

            print("Added sample sequence features for 9 samples per month")
        except Exception as e:
            print(f"Could not create sample sequence features: {e}")

        # One-hot encode month for better representation of seasonality
        month_dummies = pd.get_dummies(df['Month'], prefix='month')
        df = pd.concat([df, month_dummies], axis=1)

    # Drop rows with NaN due to lag/rolling features
    original_rows = len(df)
    df = df.dropna()
    print(f"Dropped {original_rows - len(df)} rows due to NaN values from feature engineering")

    return df


def load_full_dataset(file_path, features=None, target='Chlorophyll-a (ug/L)'):
    """
    Load the full dataset and apply all preprocessing and feature engineering
    """
    print("\n=== Loading and preparing full dataset ===")

    # Load the raw Excel file first
    raw_df = pd.read_excel(file_path)
    print(f"Loaded raw data with {raw_df.shape[0]} rows and {raw_df.shape[1]} columns")

    # Convert date column to datetime
    if 'Date' in raw_df.columns:
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])

    # Check for missing target values
    target_missing = raw_df[target].isna()
    if target_missing.any():
        print(f"Found {target_missing.sum()} rows with missing {target} values")

        # Find the last date with a valid target value
        last_valid_date = raw_df.loc[~target_missing, 'Date'].max()
        print(f"Last date with valid {target} value: {last_valid_date}")

        # Filter to only include rows up to the last valid target date
        valid_df = raw_df[raw_df['Date'] <= last_valid_date].copy()
        print(f"Using {valid_df.shape[0]} rows with valid {target} values")
    else:
        valid_df = raw_df.copy()

    # Now process this filtered dataframe
    df = load_and_preprocess_data(valid_df, features, target)

    # Define features if not provided
    if features is None:
        features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                    'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature', 'Phytoplankton']

    # Apply the same feature engineering as in the original code
    df = engineer_features(df, features, target)

    # Find the last date in the dataset with valid target values
    last_date = df['Date'].max()
    print(f"Last date with valid target value after processing: {last_date}")
    print(f"Prepared dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    return df, last_date


# ============= FUTURE PREDICTION FUNCTIONS =============
def generate_future_dates(last_date, months_ahead=18, num_stations=9):
    """
    Generate future dates for prediction, starting from the last date in the dataset.
    """
    future_dates = []
    current_date = last_date.replace(day=1)  # Ensure we start at the first of the month

    for i in range(months_ahead):
        # For each station, add an entry for the current month
        for _ in range(num_stations):
            future_dates.append(current_date)
        # Move to the next month
        current_date = current_date + relativedelta(months=1)

    return pd.DataFrame({'Date': future_dates})


def prepare_future_features(df, future_dates_df, features, target, selected_features, enable_extremity_handling=False):
    """
    Prepare future feature dataframe with the necessary columns
    Enhanced to handle extreme values when enable_extremity_handling=True
    """
    if enable_extremity_handling:
        print("\n=== Preparing future features with improved extremity handling ===")
    else:
        print("\n=== Preparing future features ===")

    # Step 1: Create a copy of the dataframe with historical data
    # Use a full year of data to ensure we capture seasonal extremes if extremity handling is enabled
    if enable_extremity_handling:
        lookback_months = 12  # Use a full year for better seasonal patterns
    else:
        lookback_months = 6  # Original 6-month lookback

    last_n_months = df[df['Date'] >= (df['Date'].max() - pd.DateOffset(months=lookback_months))].copy()
    print(f"Using last {len(last_n_months)} samples as base for prediction")

    # Step 2: Get unique station configurations
    station_cols = [col for col in df.columns if 'station_' in col]

    # Get unique station configurations from the data
    if station_cols:
        print(f"Found {len(station_cols)} station columns")
        unique_station_configs = df[station_cols].drop_duplicates().reset_index(drop=True)
        num_unique_stations = len(unique_station_configs)
        print(f"Found {num_unique_stations} unique station configurations")

        # Assign station configurations to future dates
        if len(future_dates_df) > num_unique_stations:
            repeat_count = len(future_dates_df) // num_unique_stations
            if len(future_dates_df) % num_unique_stations != 0:
                repeat_count += 1

            repeated_stations = pd.concat([unique_station_configs] * repeat_count)
            repeated_stations = repeated_stations.iloc[:len(future_dates_df)].reset_index(drop=True)

            for col in station_cols:
                future_dates_df[col] = repeated_stations[col].values

    # Step 3: Append future dates to historical data
    future_df = pd.concat([last_n_months, future_dates_df], ignore_index=True)

    # Step 4: Create seasonal features for all dates
    print("Creating seasonal features...")
    future_df['Month'] = future_df['Date'].dt.month
    future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
    future_df['DayOfMonth'] = future_df['Date'].dt.day
    future_df['Year'] = future_df['Date'].dt.year

    # Annual cycle
    future_df['Season_annual_sin'] = np.sin(future_df['DayOfYear'] * (2 * np.pi / 365.25))
    future_df['Season_annual_cos'] = np.cos(future_df['DayOfYear'] * (2 * np.pi / 365.25))

    # Quarterly cycle
    future_df['Season_quarterly_sin'] = np.sin(future_df['DayOfYear'] * (2 * np.pi / 91))
    future_df['Season_quarterly_cos'] = np.cos(future_df['DayOfYear'] * (2 * np.pi / 91))

    # Monthly cycle
    future_df['Season_monthly_sin'] = np.sin(future_df['DayOfMonth'] * (2 * np.pi / 30.44))
    future_df['Season_monthly_cos'] = np.cos(future_df['DayOfMonth'] * (2 * np.pi / 30.44))

    # Month dummies - need to recreate all the dummies that existed in the training data
    month_dummies = pd.get_dummies(future_df['Month'], prefix='month')
    future_df = pd.concat([future_df, month_dummies], axis=1)

    # Step 5: Establish baseline values - with or without extremity handling
    if enable_extremity_handling:
        print("Establishing baseline values with improved extremity handling...")

        # If we have station columns, calculate station-specific stats
        if station_cols and len(station_cols) > 0:
            # For each unique station configuration
            for i, station_config in unique_station_configs.iterrows():
                # Create a mask for this station in the historical data
                station_mask = True
                for col, value in station_config.items():
                    station_mask = station_mask & (df[col] == value)

                # Calculate monthly statistics for this station
                station_monthly_stats = df[station_mask].groupby(df[station_mask]['Date'].dt.month)[
                    features + [target]].agg(['mean', 'max', 'min', 'std'])

                # Create a mask for this station in the future data
                future_station_mask = True
                for col, value in station_config.items():
                    future_station_mask = future_station_mask & (future_df[col] == value)

                # Apply baseline values for future dates for this station
                future_station_mask = future_station_mask & (future_df['Date'] >= future_dates_df['Date'].min())

                # For each feature, apply baseline values with extremity handling
                for j, row in future_df[future_station_mask].iterrows():
                    month = row['Date'].month

                    # For regular features, use means
                    for feature in features:
                        if month in station_monthly_stats[feature].index:
                            future_df.loc[j, feature] = station_monthly_stats[feature]['mean'].loc[month]
                        else:
                            overall_monthly_avgs = df.groupby(df['Date'].dt.month)[feature].mean()
                            if month in overall_monthly_avgs.index:
                                future_df.loc[j, feature] = overall_monthly_avgs.loc[month]
                            else:
                                future_df.loc[j, feature] = df[feature].mean()

                    # For the target (chlorophyll), occasionally insert extreme values
                    if month in station_monthly_stats[target].index:
                        # Get station-month statistics for this target
                        mean_val = station_monthly_stats[target]['mean'].loc[month]
                        max_val = station_monthly_stats[target]['max'].loc[month]
                        std_val = station_monthly_stats[target]['std'].loc[month] if 'std' in station_monthly_stats[
                            target] else 0

                        # Calculate the historical ratio (higher means more likely to see extremes)
                        historical_ratio = max_val / mean_val if mean_val > 0 else 1

                        # Calculate probability of extreme value based on historical ratio
                        extreme_prob = min(0.15, (historical_ratio - 1) * 0.1)

                        # Randomly decide whether to use an extreme value
                        if np.random.random() < extreme_prob and historical_ratio > 1.5:
                            # Use an extreme value (0.8-1.1Ã— historical max)
                            extremity_factor = 0.8 + (0.3 * np.random.random())
                            extreme_value = max_val * extremity_factor
                            future_df.loc[j, target] = extreme_value
                            print(
                                f"Injected extreme value of {extreme_value:.2f} ug/L for station at index {j} (month {month})")
                        else:
                            # Use a regular value with some randomness around the mean
                            noise_factor = np.random.normal(1, 0.15)  # 15% standard deviation
                            future_df.loc[j, target] = mean_val * max(0.5, noise_factor)  # Ensure stays positive
                    else:
                        # Fallback if no station-specific data for this month
                        overall_monthly_stats = df.groupby(df['Date'].dt.month)[target].agg(['mean', 'max', 'std'])
                        if month in overall_monthly_stats.index:
                            mean_val = overall_monthly_stats['mean'].loc[month]
                            noise_factor = np.random.normal(1, 0.15)
                            future_df.loc[j, target] = mean_val * max(0.5, noise_factor)
                        else:
                            future_df.loc[j, target] = df[target].mean() * max(0.5, np.random.normal(1, 0.15))
        else:
            # If no station columns, use overall monthly averages with extremity handling
            monthly_stats = df.groupby(df['Date'].dt.month)[features + [target]].agg(['mean', 'max', 'min', 'std'])

            # For each future date, fill in baseline values based on monthly statistics
            for i, row in future_df[future_df['Date'] >= future_dates_df['Date'].min()].iterrows():
                month = row['Date'].month

                # For each feature, apply the mean value
                for feature in features:
                    if month in monthly_stats[feature].index:
                        future_df.loc[i, feature] = monthly_stats[feature]['mean'].loc[month]
                    else:
                        future_df.loc[i, feature] = df[feature].mean()

                # For the target, occasionally inject extreme values
                if month in monthly_stats[target].index:
                    mean_val = monthly_stats[target]['mean'].loc[month]
                    max_val = monthly_stats[target]['max'].loc[month]

                    # Probabilistic approach for extreme values
                    if np.random.random() < 0.1 and max_val > mean_val * 1.5:
                        extremity_factor = 0.8 + (0.3 * np.random.random())
                        extreme_value = max_val * extremity_factor
                        future_df.loc[i, target] = extreme_value
                        print(f"Injected extreme value of {extreme_value:.2f} ug/L at index {i} (month {month})")
                    else:
                        noise_factor = np.random.normal(1, 0.15)
                        future_df.loc[i, target] = mean_val * max(0.5, noise_factor)
                else:
                    future_df.loc[i, target] = df[target].mean() * max(0.5, np.random.normal(1, 0.15))
    else:
        # Original approach without extremity handling
        print("Establishing baseline values for future predictions...")

        # If we have station columns, calculate averages by station and month
        if station_cols and len(station_cols) > 0:
            # For each unique station configuration
            for i, station_config in unique_station_configs.iterrows():
                # Create a mask for this station in the historical data
                station_mask = True
                for col, value in station_config.items():
                    station_mask = station_mask & (df[col] == value)

                # Calculate monthly averages for this station
                station_monthly_avgs = df[station_mask].groupby(df[station_mask]['Date'].dt.month)[
                    features + [target]].mean()

                # Create a mask for this station in the future data
                future_station_mask = True
                for col, value in station_config.items():
                    future_station_mask = future_station_mask & (future_df[col] == value)

                # Apply baseline values for future dates for this station
                future_station_mask = future_station_mask & (future_df['Date'] >= future_dates_df['Date'].min())
                for feature in features + [target]:
                    for j, row in future_df[future_station_mask].iterrows():
                        month = row['Date'].month
                        # If we have data for this month for this station, use it
                        if month in station_monthly_avgs.index:
                            future_df.loc[j, feature] = station_monthly_avgs.loc[month, feature]
                        else:
                            # Fallback to overall monthly average if we don't have station-specific data
                            overall_monthly_avgs = df.groupby(df['Date'].dt.month)[features + [target]].mean()
                            if month in overall_monthly_avgs.index:
                                future_df.loc[j, feature] = overall_monthly_avgs.loc[month, feature]
                            else:
                                # Fallback to overall average if we don't have month-specific data
                                future_df.loc[j, feature] = df[feature].mean()
        else:
            # If no station columns, use overall monthly averages
            monthly_avgs = df.groupby(df['Date'].dt.month)[features + [target]].mean()

            # For each future date, fill in baseline values based on the monthly averages
            for feature in features + [target]:
                for i, row in future_df[future_df['Date'] >= future_dates_df['Date'].min()].iterrows():
                    month = row['Date'].month
                    if month in monthly_avgs.index:
                        future_df.loc[i, feature] = monthly_avgs.loc[month, feature]
                    else:
                        future_df.loc[i, feature] = df[feature].mean()

    # Step 6: Handle NaN values that might exist in the historical part of future_df
    print("Handling any remaining NaN values...")
    for col in features + [target]:
        if future_df[col].isna().any():
            median_val = df[col].median()
            na_count = future_df[col].isna().sum()
            future_df[col] = future_df[col].fillna(median_val)
            print(f"Filled {na_count} NaN values in '{col}' with median: {median_val:.4f}")

    # Step 7: Apply log transformation to the target (if it was used in training)
    future_df[f'log_{target}'] = np.log1p(future_df[target])

    # Step 8: Now we need to generate all the time-based features
    # This is complex because we need to shift and roll forward
    print("Generating time-based features (lag, rolling, diff)...")

    # Create the full feature set
    # Lagged features
    for feature in features + [target]:
        for lag in [1, 2, 3, 5, 9]:
            future_df[f'{feature}_lag{lag}'] = future_df[feature].shift(lag)

    # Rolling window features
    for feature in features + [target]:
        for window in [3, 7, 14]:
            future_df[f'{feature}_roll_mean{window}'] = future_df[feature].rolling(window=window).mean()
            future_df[f'{feature}_roll_max{window}'] = future_df[feature].rolling(window=window).max()
            future_df[f'{feature}_roll_std{window}'] = future_df[feature].rolling(window=window).std()

    # Rate-of-change features
    for feature in features + [target]:
        future_df[f'{feature}_diff1'] = future_df[feature].diff()
        future_df[f'{feature}_pct_change'] = future_df[feature].pct_change()
        future_df[f'{feature}_diff3'] = future_df[feature].diff(3)
        future_df[f'{feature}_diff9'] = future_df[feature].diff(9)

    # Create interaction features
    for i, feat1 in enumerate(features[:min(3, len(features))]):
        for feat2 in features[i + 1:min(i + 3, len(features))]:
            future_df[f'{feat1}_{feat2}_interact'] = future_df[feat1] * future_df[feat2]

    # Step 9: Check that all required features exist in the dataframe
    # Create any missing features with default values
    for feature in selected_features:
        if feature not in future_df.columns:
            print(f"Warning: Feature '{feature}' not in future dataframe. Adding with zeros.")
            future_df[feature] = 0

    # Return only the future rows
    future_rows = future_df[future_df['Date'] >= future_dates_df['Date'].min()].copy()
    print(f"Prepared {len(future_rows)} future rows for prediction")

    return future_rows


def retrain_model_on_full_data(df, features, target, selected_features, model):
    """
    Retrain the model on the full dataset for future prediction
    """
    print("\n=== Retraining model on full dataset ===")

    # Prepare X and y from full dataset
    X = df[selected_features].copy()
    y = df[target].copy()

    # Check if log transform was used in the original model
    y_log = np.log1p(y)

    # Retrain the model on the full dataset
    print(f"Training on {len(df)} samples with {len(selected_features)} features")
    model.fit(X, y_log)

    # Calculate model performance on full dataset
    y_pred = model.predict(X)
    y_pred = np.expm1(y_pred)  # Transform back from log
    r2 = model.score(X, y_log)
    rmse = np.sqrt(((y_pred - y) ** 2).mean())
    print(f"Full dataset performance - RÂ²: {r2:.4f}, RMSE: {rmse:.4f}")

    return model, r2, rmse


def predict_future_values(model, future_df, selected_features, use_log=True):
    """
    Make predictions for future values
    """
    print("\n=== Making future predictions ===")

    # Prepare features for prediction
    X_future = future_df[selected_features].copy()

    # Fill any remaining NaN values with median from the columns
    for col in X_future.columns:
        if X_future[col].isna().any():
            col_median = X_future[col].median()
            # If median is NaN (all values are NaN), use 0 instead
            if pd.isna(col_median):
                print(f"Warning: All values in '{col}' are NaN. Using 0 as default.")
                col_median = 0
            na_count = X_future[col].isna().sum()
            X_future[col] = X_future[col].fillna(col_median)
            print(f"Filled {na_count} NaN values in '{col}' with median: {col_median}")

    # Double-check that there are no NaN values left
    if X_future.isna().any().any():
        print("WARNING: Some NaN values remain after filling. Replacing with zeros.")
        X_future = X_future.fillna(0)

    # Make predictions
    future_pred = model.predict(X_future)

    # Transform back if log was used
    if use_log:
        future_pred = np.expm1(future_pred)

    # Add predictions to future dataframe
    future_df['Predicted_Chlorophyll'] = future_pred

    return future_df


def plot_and_save_results(df, future_pred_df, target='Chlorophyll-a (ug/L)'):
    """
    Plot historical values and future predictions
    """
    print("\n=== Plotting results ===")

    # Create a figure for the overall plot
    plt.figure(figsize=(15, 8))

    # Plot historical data with light dots for individual samples
    plt.scatter(df['Date'], df[target], alpha=0.3, s=15, c='blue', label='_nolegend_')

    # Calculate and plot monthly averages for historical data
    monthly_hist = df.groupby(df['Date'].dt.strftime('%Y-%m-01'))[[target]].mean()
    monthly_hist.index = pd.to_datetime(monthly_hist.index)
    plt.plot(monthly_hist.index, monthly_hist[target], 'b-', linewidth=2, label='Historical Monthly Avg')

    # Plot future predictions with light dots for individual stations
    plt.scatter(future_pred_df['Date'], future_pred_df['Predicted_Chlorophyll'],
                alpha=0.3, s=15, c='red', label='_nolegend_')

    # Calculate and plot monthly averages for future predictions
    monthly_pred = future_pred_df.groupby(future_pred_df['Date'].dt.strftime('%Y-%m-01'))[
        ['Predicted_Chlorophyll']].mean()
    monthly_pred.index = pd.to_datetime(monthly_pred.index)
    plt.plot(monthly_pred.index, monthly_pred['Predicted_Chlorophyll'], 'r--',
             linewidth=2, label='Predicted Monthly Avg')

    # Add a vertical line to separate historical and future data
    last_date = df['Date'].max()
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    plt.text(last_date, plt.ylim()[1] * 0.9, ' Historical | Future ', backgroundcolor='w')

    # Add labels and title
    plt.title('Chlorophyll-a: Historical Data and Future Predictions (18 months)')
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll-a (ug/L)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Rotate x-axis dates for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig('chlorophyll_future_predictions.png', dpi=300)
    print("Plot saved as 'chlorophyll_future_predictions.png'")


    # Create station-specific plots
    station_cols = [col for col in future_pred_df.columns if 'station_' in col]
    if station_cols:
        # If we have station columns, try to identify station names
        station_names = []
        for col in station_cols:
            # Try to extract station name from column name (assuming format like 'station_Station_1_CWB')
            parts = col.split('_')
            if len(parts) > 2:  # Format should have at least "station" + "Station" + station specific part
                station_name = '_'.join(parts[2:])  # Join all parts after "station_Station_"
                station_names.append((col, station_name))

        # Create a multi-station plot
        plt.figure(figsize=(15, 10))

        # For each station configuration
        for i, (col, name) in enumerate(station_names):
            # Create a mask for this station
            station_mask = future_pred_df[col] == 1
            if station_mask.any():
                # Plot this station's predictions
                station_data = future_pred_df[station_mask]
                plt.plot(station_data['Date'], station_data['Predicted_Chlorophyll'],
                         marker='o', label=f'Station {name}')

        plt.title('Chlorophyll-a Predictions by Station')
        plt.xlabel('Date')
        plt.ylabel('Chlorophyll-a (ug/L)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the multi-station plot
        plt.savefig('chlorophyll_predictions_by_station.png', dpi=300)
        print("Station-specific plot saved as 'chlorophyll_predictions_by_station.png'")


    # Save the predictions to CSV
    future_pred_df.to_csv('chlorophyll_future_predictions.csv', index=False)
    print("Predictions saved to 'chlorophyll_future_predictions.csv'")

    # Print summary of predictions
    print("\n=== Summary of Future Predictions ===")

    # Group by date (year-month) and calculate statistics
    monthly_results = future_pred_df.groupby(future_pred_df['Date'].dt.strftime('%Y-%m'))[
        ['Predicted_Chlorophyll']].agg(['mean', 'min', 'max', 'std'])
    print(monthly_results)

    # If we have station columns, also show predictions by station
    if station_cols and len(station_names) > 0:
        print("\n=== Station-Specific Predictions ===")
        # Create a new dataframe with station names for better readability
        station_pred_df = future_pred_df.copy()

        # Add a Station column indicating which station each row belongs to
        station_pred_df['Station'] = 'Unknown'
        for col, name in station_names:
            station_pred_df.loc[station_pred_df[col] == 1, 'Station'] = name

        # Group by station and month and calculate average predictions
        station_monthly = station_pred_df.groupby(['Station', station_pred_df['Date'].dt.strftime('%Y-%m')])[
            ['Predicted_Chlorophyll']].mean()
        print(station_monthly)

        # Save station-specific predictions to CSV
        station_monthly.to_csv('CSV/chlorophyll_predictions_by_station.csv')
        print("Station-specific monthly predictions saved to 'chlorophyll_predictions_by_station.csv'")

    return monthly_results


# ============= MAIN FUNCTION =============
def main():
    """
    Enhanced main function to save both regular and extreme value predictions
    """
    # Step 1: Load the saved model, selected features, and metadata
    model, selected_features, metadata = load_existing_model_and_features()

    # Step 2: Define features and target
    features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature', 'Phytoplankton']
    target = 'Chlorophyll-a (ug/L)'

    # Step 3: Load and prepare the full dataset with all preprocessing and feature engineering
    file_path = 'CSV/merged_stations.xlsx'
    df, last_date = load_full_dataset(file_path, features, target)

    # Step 4: Retrain the model on the full dataset
    retrained_model, full_r2, full_rmse = retrain_model_on_full_data(
        df, features, target, selected_features, model
    )

    # Step 5: Analyze historical extremes
    print("\n=== Analyzing historical extreme values ===")
    station_cols = [col for col in df.columns if 'station_' in col]

    # For each station, find maximum values by month
    if station_cols:
        print("Historical maximum values by station and month:")
        unique_station_configs = df[station_cols].drop_duplicates().reset_index(drop=True)
        num_stations = len(unique_station_configs)

        for i, station_config in unique_station_configs.iterrows():
            # Create a mask for this station
            station_mask = True
            for col, value in station_config.items():
                station_mask = station_mask & (df[col] == value)

            # Get station name
            station_name = "Unknown"
            for col in station_cols:
                if station_config[col] == 1:
                    parts = col.split('_')
                    if len(parts) > 2:
                        station_name = '_'.join(parts[2:])

            # Calculate monthly statistics for this station
            monthly_max = df[station_mask].groupby(df[station_mask]['Date'].dt.month)[[target]].max()
            print(f"\nStation {station_name} historical monthly maximums:")
            print(monthly_max)

            # Calculate overall station maximum
            station_max = df[station_mask][target].max()
            print(f"Overall maximum for Station {station_name}: {station_max:.2f} ug/L")

    # Calculate overall maximum
    overall_max = df[target].max()
    print(f"\nOverall historical maximum chlorophyll: {overall_max:.2f} ug/L")

    # Step 6: Generate future dates for all stations
    future_dates_df = generate_future_dates(last_date, months_ahead=18, num_stations=num_stations)

    # Step 7A: Prepare features for future dates WITHOUT extremity handling (regular values)
    print("\n=== Generating predictions with regular values ===")
    future_df_regular = prepare_future_features(
        df, future_dates_df, features, target, selected_features, enable_extremity_handling=False
    )

    # Step 7B: Prepare features for future dates WITH extremity handling
    print("\n=== Generating predictions with extreme values ===")
    future_df_extreme = prepare_future_features(
        df, future_dates_df, features, target, selected_features, enable_extremity_handling=True
    )

    # Step 8A: Predict future values with regular data
    future_pred_df_regular = predict_future_values(
        retrained_model, future_df_regular, selected_features, use_log=True
    )

    # Step 8B: Predict future values with extreme data
    future_pred_df_extreme = predict_future_values(
        retrained_model, future_df_extreme, selected_features, use_log=True
    )

    # Step 9A: Plot and save results for regular predictions
    print("\n=== Plotting and saving regular predictions ===")
    plt.figure(figsize=(15, 8))
    plt.scatter(df['Date'], df[target], alpha=0.3, s=15, c='blue', label='_nolegend_')
    monthly_hist = df.groupby(df['Date'].dt.strftime('%Y-%m-01'))[[target]].mean()
    monthly_hist.index = pd.to_datetime(monthly_hist.index)
    plt.plot(monthly_hist.index, monthly_hist[target], 'b-', linewidth=2, label='Historical Monthly Avg')
    plt.scatter(future_pred_df_regular['Date'], future_pred_df_regular['Predicted_Chlorophyll'],
                alpha=0.3, s=15, c='green', label='_nolegend_')
    monthly_pred_regular = future_pred_df_regular.groupby(future_pred_df_regular['Date'].dt.strftime('%Y-%m-01'))[
        ['Predicted_Chlorophyll']].mean()
    monthly_pred_regular.index = pd.to_datetime(monthly_pred_regular.index)
    plt.plot(monthly_pred_regular.index, monthly_pred_regular['Predicted_Chlorophyll'], 'g--',
             linewidth=2, label='Regular Predicted Monthly Avg')
    last_date = df['Date'].max()
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    plt.text(last_date, plt.ylim()[1] * 0.9, ' Historical | Future ', backgroundcolor='w')
    plt.title('Chlorophyll-a: Historical Data and Regular Future Predictions (18 months)')
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll-a (ug/L)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('chlorophyll_future_predictions_regular.png', dpi=300)
    print("Regular predictions plot saved as 'chlorophyll_future_predictions_regular.png'")
    plt.close()

    # Step 9B: Plot and save results for extreme predictions
    print("\n=== Plotting and saving extreme predictions ===")
    summary_extreme = plot_and_save_results(df, future_pred_df_extreme, target)

    # Step 9C: Combined plot showing both predictions
    print("\n=== Creating combined plot with both prediction types ===")
    plt.figure(figsize=(15, 8))
    plt.scatter(df['Date'], df[target], alpha=0.3, s=15, c='blue', label='_nolegend_')
    plt.plot(monthly_hist.index, monthly_hist[target], 'b-', linewidth=2, label='Historical Monthly Avg')

    # Plot regular predictions
    plt.scatter(future_pred_df_regular['Date'], future_pred_df_regular['Predicted_Chlorophyll'],
                alpha=0.2, s=15, c='green', label='_nolegend_')
    plt.plot(monthly_pred_regular.index, monthly_pred_regular['Predicted_Chlorophyll'], 'g--',
             linewidth=2, label='Regular Predictions')

    # Plot extreme predictions
    monthly_pred_extreme = future_pred_df_extreme.groupby(future_pred_df_extreme['Date'].dt.strftime('%Y-%m-01'))[
        ['Predicted_Chlorophyll']].mean()
    monthly_pred_extreme.index = pd.to_datetime(monthly_pred_extreme.index)
    plt.scatter(future_pred_df_extreme['Date'], future_pred_df_extreme['Predicted_Chlorophyll'],
                alpha=0.2, s=15, c='red', label='_nolegend_')
    plt.plot(monthly_pred_extreme.index, monthly_pred_extreme['Predicted_Chlorophyll'], 'r--',
             linewidth=2, label='Extreme-Aware Predictions')

    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7)
    plt.text(last_date, plt.ylim()[1] * 0.9, ' Historical | Future ', backgroundcolor='w')
    plt.title('Chlorophyll-a: Comparison of Regular and Extreme-Aware Predictions')
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll-a (ug/L)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('chlorophyll_predictions_comparison.png', dpi=300)
    print("Comparison plot saved as 'chlorophyll_predictions_comparison.png'")
    plt.close()

    # Step 10: Save the retrained model
    print("\n=== Saving retrained model ===")
    with open('pkl/chlorophyll_gb_model_full_data.pkl', 'wb') as file:
        pickle.dump(retrained_model, file)

    # Create and save metadata
    full_metadata = {
        'r2_score': full_r2,
        'rmse': full_rmse,
        'use_log_transform': True,
        'model_parameters': retrained_model.named_steps['model'].get_params(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'last_historical_date': last_date.strftime('%Y-%m-%d'),
        'prediction_range': f"{future_dates_df['Date'].min().strftime('%Y-%m-%d')} to {future_dates_df['Date'].max().strftime('%Y-%m-%d')}",
        'num_stations': num_stations,
        'historical_max': overall_max
    }

    with open('pkl/chlorophyll_gb_model_full_metadata.pkl', 'wb') as file:
        pickle.dump(full_metadata, file)

    # Save the future predictions with regular values
    future_pred_df_regular.to_csv('chlorophyll_future_predictions_regular.csv', index=False)
    print("Regular predictions saved to 'chlorophyll_future_predictions_regular.csv'")

    # Save the future predictions with extremes
    future_pred_df_extreme.to_csv('chlorophyll_future_predictions_extreme.csv', index=False)
    print("Extreme-aware predictions saved to 'chlorophyll_future_predictions_extreme.csv'")

    # Save a combined dataframe with both predictions for comparison
    combined_df = future_pred_df_regular[['Date', 'Station']].copy() if 'Station' in future_pred_df_regular.columns else \
    future_pred_df_regular[['Date']].copy()
    combined_df['Regular_Predicted_Chlorophyll'] = future_pred_df_regular['Predicted_Chlorophyll']
    combined_df['Extreme_Predicted_Chlorophyll'] = future_pred_df_extreme['Predicted_Chlorophyll']
    combined_df['Difference'] = combined_df['Extreme_Predicted_Chlorophyll'] - combined_df[
        'Regular_Predicted_Chlorophyll']
    combined_df.to_csv('chlorophyll_predictions_comparison.csv', index=False)
    print("Comparison of predictions saved to 'chlorophyll_predictions_comparison.csv'")

    # Print comparison summary
    print("\n=== Comparison of Regular vs Extreme-Aware Predictions ===")
    print(f"Average difference: {combined_df['Difference'].mean():.4f} ug/L")
    print(f"Maximum difference: {combined_df['Difference'].max():.4f} ug/L")
    print(
        f"Percentage of samples with significant difference (>10%): {(abs(combined_df['Difference']) > 0.1 * combined_df['Regular_Predicted_Chlorophyll']).mean() * 100:.2f}%")

    print("\nPrediction process complete with both regular and extreme-aware predictions!")


if __name__ == "__main__":
    main()