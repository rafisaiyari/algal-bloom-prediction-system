import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
import pickle
from skopt import BayesSearchCV
from skopt.space import Real, Integer

warnings.filterwarnings('ignore')


# ============= LOAD AND PREPROCESS DATA =============
def load_and_preprocess_data(file_path, features=None, target='Chlorophyll-a (ug/L)'):
    """
    Load and preprocess water quality data with time-aware missing value handling
    """
    print("\n=== Loading and preprocessing data ===")
    # Load Excel file
    df = pd.read_excel(file_path)

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

# ============= FEATURE SELECTION =============
def select_features(X_train, y_train, use_log=False):
    """
    Select important features using Random Forest feature importance
    """
    print("\n=== Performing feature selection ===")

    # Use a simpler random forest model for feature selection
    feature_selector = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,  # Limit depth for less complexity
        min_samples_split=5,  # Require more samples per split
        min_samples_leaf=4,  # Require more samples per leaf
        random_state=42
    )
    feature_selector.fit(X_train, y_train)

    # Get feature importances
    importances = feature_selector.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Print top features
    print("Top 15 important features:")
    print(feature_importance.head(15))

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    # Select fewer features - keep top 20% to reduce model complexity
    num_features = max(int(len(X_train.columns) * 0.2), 10)
    num_features = min(num_features, len(X_train.columns))
    selected_features = feature_importance['feature'][:num_features].values
    print(f"Selected {len(selected_features)} out of {len(X_train.columns)} features")

    return selected_features, feature_importance

# ============= FEATURE ENGINEERING =============
def engineer_features(df, features, target):
    """
    Create improved features for time series prediction with focus on capturing peaks
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




# ============= MODEL TRAINING =============
def train_models(X_train, y_train, selected_features, tscv):
    """
    Train and tune Gradient Boosting models with simplified parameters
    """

    # 2. Gradient Boosting Hyperparameter Tuning using BayesSearchCV with simplified parameters
    print("\nTuning Gradient Boosting hyperparameters with BayesSearchCV...")

    # Simplified search spaces for Gradient Boosting
    # Optimized search spaces for Gradient Boosting
    search_spaces = {
        'n_estimators': Integer(800, 1200),  # Increase estimators to compensate for stronger regularization
        'max_depth': Integer(2, 4),  # Decrease max depth to limit tree complexity
        'learning_rate': Real(0.001, 0.02),  # Lower learning rate range for more stability
        'subsample': Real(0.6, 0.8),  # More aggressive subsampling to reduce overfitting
        'min_samples_split': Integer(5, 15),  # Require more samples per split
        'min_samples_leaf': Integer(4, 10),  # Require more samples per leaf
        'max_features': Real(0.5, 0.8),  # Limit features considered at each split
        'alpha': Real(0.1, 0.9)  # Add alpha parameter for Huber loss
    }

    # And update the estimator to use Huber loss:
    bayes_search = BayesSearchCV(
        estimator=GradientBoostingRegressor(
            random_state=42,
            validation_fraction=0.2,  # Increased validation fraction
            n_iter_no_change=20,  # More patience for convergence with stronger regularization
            tol=0.001,  # Higher tolerance for early stopping
            loss='huber'  # Switch to Huber loss for robustness to outliers
        ),
        search_spaces=search_spaces,
        n_iter=60,
        cv=tscv,
        scoring='r2',
        random_state=42,
        n_jobs=-1
    )

    # Fit the Bayesian search
    bayes_search.fit(X_train[selected_features], y_train)

    # Best parameters
    print("Best Gradient Boosting parameters found:")
    for param, value in bayes_search.best_params_.items():
        print(f"  - {param}: {value}")

    # Train GB with best parameters
    best_gb = bayes_search.best_estimator_
    print("\nTraining Gradient Boosting with best parameters...")

    # 3. Create Pipeline with Robust Scaling for Gradient Boosting
    gb_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', best_gb)
    ])

    # Fit the pipeline
    gb_pipeline.fit(X_train[selected_features], y_train)

    return gb_pipeline, bayes_search


# ============= MODEL EVALUATION =============
def evaluate_models(gb_pipeline, X_train, X_test, y_train, y_test, selected_features, use_log=False, df=None):
    """
    Evaluate models on training and test datasets with date-based outputs
    """
    print("\n=== Evaluating models ===")

    # Define y_log_train and y_log_test if using log transformation
    if use_log:
        # Assuming y_train and y_test are the original values
        y_log_train = np.log1p(y_train)
        y_log_test = np.log1p(y_test)
    else:
        y_log_train = y_train
        y_log_test = y_test

    # Calculate R2 on training data for Gradient Boosting
    gb_train_pred = gb_pipeline.predict(X_train[selected_features])
    if use_log:
        gb_train_pred = np.expm1(gb_train_pred)  # Convert back from log
    gb_train_r2 = r2_score(y_train, gb_train_pred)
    gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))

    # Print the training results
    print("\n===== TRAINING DATA PERFORMANCE =====")
    print(f"Gradient Boosting - Train R²: {gb_train_r2:.4f}, Train RMSE: {gb_train_rmse:.4f}")

    gb_pred = gb_pipeline.predict(X_test[selected_features])
    if use_log:
        gb_pred = np.expm1(gb_pred)  # Convert back from log

    # Evaluate models on test set
    print("\n===== TEST DATA PERFORMANCE =====")

    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    print(f"Gradient Boosting - Test R²: {gb_r2:.4f}, Test RMSE: {gb_rmse:.4f}")

    # Print date, predicted values, and actual values
    print("\n===== DATE, PREDICTED VALUES, AND ACTUAL VALUES =====")
    print_prediction_results(X_test, y_test, gb_pred, df)

    # Calculate and print the difference between train and test performance to check for overfitting
    print("\n===== TRAIN VS TEST COMPARISON =====")
    print(f"Gradient Boosting - R² difference (train-test): {gb_train_r2 - gb_r2:.4f}")
    print(f"Gradient Boosting - RMSE difference (test-train): {gb_rmse - gb_train_rmse:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(15, 8))

    # Create a common x-axis for comparison
    x_values = np.arange(len(y_test))

    # Plot actual values and predictions at the same x locations
    plt.plot(x_values, y_test, label='Actual', color='blue', linewidth=2)
    plt.plot(x_values, gb_pred, label=f'Gradient Boosting (R²={gb_r2:.3f})', linestyle='--', color='green')

    # If dates are available, create custom x-axis ticks and labels
    if df is not None and 'Date' in df.columns:
        test_dates = df.loc[X_test.index, 'Date']

        if isinstance(test_dates.iloc[0], pd.Timestamp):
            # Extract years from test dates
            years = test_dates.dt.year.unique()

            # Create tick positions - assuming 9 samples per year
            tick_positions = []
            tick_labels = []

            for i, year in enumerate(years):
                # Place the tick at the middle position of each year's samples
                year_indices = x_values[test_dates.dt.year == year]
                if len(year_indices) > 0:
                    # Place a tick at the middle of each year's data points
                    tick_pos = year_indices[len(year_indices) // 2]
                    tick_positions.append(tick_pos)
                    tick_labels.append(str(year))

                    # Optionally add vertical lines to separate years
                    if i > 0:  # Skip the first year boundary
                        first_idx = year_indices[0]
                        plt.axvline(x=first_idx, color='gray', linestyle=':', alpha=0.5)

            # Set custom ticks
            plt.xticks(tick_positions, tick_labels)

            # Update xlabel to reflect years
            plt.xlabel("Year")
        else:
            plt.xlabel("Sample Index")
    else:
        plt.xlabel("Sample Index")

    plt.title("Comparison of Predicted Values to Actual Values")
    plt.ylabel("Chlorophyll-a (ug/L)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Add cross-validation scoring
    from sklearn.model_selection import cross_val_score

    # 5-fold cross-validation for Gradient Boosting
    gb_cv_scores = cross_val_score(gb_pipeline, X_train[selected_features],
                                   y_train if not use_log else y_log_train,
                                   cv=5, scoring='r2')
    print(f"Gradient Boosting CV R² scores: {gb_cv_scores}")
    print(f"Gradient Boosting CV R² mean: {gb_cv_scores.mean():.4f}, std: {gb_cv_scores.std():.4f}")

    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    fold_numbers = np.arange(1, len(gb_cv_scores) + 1)

    # Bar plot for individual fold scores
    plt.bar(fold_numbers, gb_cv_scores, alpha=0.7, color='forestgreen', label='Individual Fold Scores')

    # Line for mean score
    mean_score = gb_cv_scores.mean()
    plt.axhline(y=mean_score, color='red', linestyle='-', label=f'Mean R² = {mean_score:.4f}')

    # Add standard deviation band
    std_score = gb_cv_scores.std()
    plt.axhline(y=mean_score + std_score, color='lightcoral', linestyle='--', alpha=0.7,
                label=f'±1 std ({std_score:.4f})')
    plt.axhline(y=mean_score - std_score, color='lightcoral', linestyle='--', alpha=0.7)

    # Fill the standard deviation band
    plt.fill_between(fold_numbers, mean_score - std_score, mean_score + std_score,
                     color='lightcoral', alpha=0.2)

    # Add text labels above each bar
    for i, score in enumerate(gb_cv_scores):
        plt.text(i + 1, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Cross-Validation R² Scores for Gradient Boosting Model')
    plt.xlabel('Fold Number')
    plt.ylabel('R² Score')
    plt.xticks(fold_numbers)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # You could also plot the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        gb_pipeline, X_train[selected_features],
        y_train if not use_log else y_log_train,
        cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5,
             label='Training R²')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5,
             label='Cross-validation R²')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                     alpha=0.15, color='green')

    plt.title('Learning Curve for Gradient Boosting Model')
    plt.xlabel('Training Examples')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return gb_pred, gb_r2, gb_rmse

    return  gb_pred, gb_r2, gb_rmse


def print_prediction_results(X_test, y_test, gb_pred, df=None):
    """
    Print the date (if available), predicted values, and actual values
    """
    # Create a results dataframe
    results = pd.DataFrame({
        'Actual': y_test.values,
        'GB_Predicted': gb_pred,
        'GB_Error': gb_pred - y_test.values
    })

    # Add date if available
    if df is not None and 'Date' in df.columns:
        # Assuming X_test has the same index as the corresponding rows in df
        # Get the dates that correspond to the test set
        test_dates = df.loc[X_test.index, 'Date']
        results['Date'] = test_dates.values
        # Reorder columns to put Date first
        results = results[['Date', 'Actual', 'GB_Predicted', 'GB_Error']]

    # Print the results
    print("Results for the first 20 test samples:")
    print(results.head(20).to_string(index=True, float_format='{:.4f}'.format))

    # Calculate and print the percentage error for the top 10 errors
    results['GB_Pct_Error'] = (results['GB_Error'] / results['Actual']) * 100

    print("\nTop 10 Gradient Boosting prediction errors (absolute):")
    top_errors = results.sort_values(by='GB_Error', key=abs, ascending=False)
    print(top_errors.head(10)[['Date', 'Actual', 'GB_Predicted', 'GB_Error', 'GB_Pct_Error']].to_string(
        index=True, float_format='{:.4f}'.format))

    # Export the full results to CSV
    results.to_csv('prediction_results.csv', index=False)
    print("\nFull prediction results exported to 'prediction_results.csv'")

# ============= SAVE MODELS =============
def save_models(gb_pipeline, selected_features, use_log, gb_r2, gb_rmse):
    """
    Save models, selected features, and metadata to disk
    """
    print("\n=== Saving models to disk ===")

    # Save the gradient boosting pipeline model
    with open('pkl/chlorophyll_gb_model.pkl', 'wb') as file:
        pickle.dump(gb_pipeline, file)
    print("Gradient Boosting pipeline saved as 'chlorophyll_gb_model.pkl'")

    # Save selected features for future reference
    with open('pkl/selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)
    print(f"Selected features list saved as 'selected_features.pkl'")

    # Optional: Save model metadata
    model_metadata = {
        'r2_score': gb_r2,
        'rmse': gb_rmse,
        'use_log_transform': use_log,
        'model_parameters': gb_pipeline.named_steps['model'].get_params(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    with open('pkl/chlorophyll_gb_model_metadata.pkl', 'wb') as file:
        pickle.dump(model_metadata, file)
    print("Model metadata saved as 'chlorophyll_gb_model_metadata.pkl'")


# ============= MAIN FUNCTION =============
def main():
    """
    Main function to run the entire model training pipeline
    """
    # Load data
    file_path = 'CSV/merged_stations.xlsx'
    df = load_and_preprocess_data(file_path)

    # Define features and target
    features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature']

    # Check if all features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        print(f"Warning: Some features not found. Using {len(available_features)} available features:")
        print(available_features)
        features = available_features

    target = 'Chlorophyll-a (ug/L)'

    # 1. Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], kde=True)
    plt.title(f'Distribution of {target}')
    plt.show()

    # Log transform target if skewed
    skewness = stats.skew(df[target])
    print(f"Skewness of {target}: {skewness}")

    # Always apply log transformation for environmental data
    print(f"Applying log transformation to target regardless of skewness.")
    # Add small constant to handle zeros
    df[f'log_{target}'] = np.log1p(df[target])
    use_log = True
    log_target = f'log_{target}'

    # Feature engineering with reduced complexity
    df = engineer_features(df, features, target)

    # All features including engineered ones
    # Exclude any non-numeric columns, Date, target and log_target
    exclude_cols = [col for col in df.columns if df[col].dtype == 'object'] + ['Date', target, log_target]
    all_features = [col for col in df.columns if col not in exclude_cols]

    # Verify all features are numeric
    for col in all_features:
        if not np.issubdtype(df[col].dtype, np.number):
            print(f"Warning: Non-numeric column detected: {col}, type: {df[col].dtype}")
            all_features.remove(col)

    # Print feature list
    print(f"Total features after engineering: {len(all_features)}")

    # Prepare data
    X = df[all_features]
    y = df[target]
    y_log = df[log_target] if use_log else y

    # Setup time series cross-validation with more splits
    print("\nSetting up time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)  # Keep 5 splits

    # Date-based train-test split with careful handling
    if 'Date' in df.columns:
        # Convert to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])

        # Get unique dates and sort
        unique_dates = df['Date'].sort_values().unique()

        # Calculate the split point - use the 80% mark instead of 70% to ensure enough training data
        split_idx = int(len(unique_dates) * 0.8)
        split_date = unique_dates[split_idx]

        print(f"Train-Test split date: {split_date}")

        # Create train and test sets based on dates
        train_mask = df['Date'] < split_date
        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]
        y_log_train = y_log[train_mask] if use_log else y_train
        y_log_test = y_log[~train_mask] if use_log else y_test
    else:
        # If no Date column, use standard split
        train_size = int(len(df) * 0.8)  # Use 80% for training
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        y_log_train, y_log_test = y_log.iloc[:train_size], y_log.iloc[train_size:]

    print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

    # Feature selection - use fewer features
    selected_features, feature_importance = select_features(X_train, y_log_train if use_log else y_train)

    # Train models with simplified parameters
    gb_pipeline, gb_random = train_models(
        X_train, y_log_train if use_log else y_train, selected_features, tscv
    )

    # Evaluate models - Pass the full dataframe to access dates
    gb_pred, gb_r2, gb_rmse = evaluate_models(
        gb_pipeline, X_train, X_test, y_train, y_test,
        selected_features, use_log, df
    )

    # Save final models and metadata
    save_models(gb_pipeline, selected_features, use_log, gb_r2, gb_rmse)

    # Print final performance summary
    print("\n===== SUMMARY OF RESULTS =====")
    print(f"Tuned Gradient Boosting: R² = {gb_r2:.4f}, RMSE = {gb_rmse:.4f}")


if __name__ == "__main__":
    main()

