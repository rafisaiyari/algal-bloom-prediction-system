import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
from scipy.stats import pearsonr
from skopt.space import Real, Integer

warnings.filterwarnings('ignore')


# ============= LOAD AND PREPROCESS DATA =============
def load_and_preprocess_data(file_path, features=None, target='Chlorophyll-a (ug/L)'):
    """
    Load and preprocess water quality data with time-aware missing value handling
    """
    print("Loading and preprocessing data...\n")
    # Load Excel file
    df = pd.read_excel(file_path)

    # Display DataFrame dimensions
    #print(f"Number of rows: {df.shape[0]}")
    #print(f"Number of columns: {df.shape[1]}")

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
        print("Processing station information...\n")
        # One-hot encode the Station column
        station_dummies = pd.get_dummies(df['Station'], prefix='station')
        df = pd.concat([df, station_dummies], axis=1)
        #print(f"Created {station_dummies.shape[1]} station dummy variables")

    # Ensure data is sorted by date if a date column exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        #print("Sorted data by Date")

    # Set features if not provided
    if features is None:
        features = [col for col in df.select_dtypes(include=np.number).columns
                    if col != target]

    # Time-aware missing value handling
    print("Handling missing values...\n")
    cols_with_na = df.columns[df.isna().any()].tolist()
    #print(f"Columns with missing values: {cols_with_na}")

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

               # print(f"Imputed {na_count_before} missing values in '{col}':")
               # for method, count in method_counts.items():
                   # print(f" - {count} values using {method}")

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
   # print("\nDropping remaining non-numeric columns:")
    for col in df.select_dtypes(include=['object']).columns:
        if col != 'Date' and col != 'Station':  # Keep Date and Station
           # print(f"  - Dropping: {col}")
            df = df.drop(columns=[col])

    return df

# ============= FEATURE SELECTION =============
def select_features(X_train, y_train, use_log=False):
    """
    Select important features using Random Forest feature importance
    """
    #print("\n=== Performing feature selection ===")

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
    #print("Top 15 important features:")
    #print(feature_importance.head(15))

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Feature Importance')
    plt.tight_layout()
    #plt.show()

    # Select fewer features - keep top 20% to reduce model complexity
    num_features = max(int(len(X_train.columns) * 0.2), 10)
    num_features = min(num_features, len(X_train.columns))
    selected_features = feature_importance['feature'][:num_features].values
    #print(f"Selected {len(selected_features)} out of {len(X_train.columns)} features")

    return selected_features, feature_importance

# ============= FEATURE ENGINEERING =============
def engineer_features(df, features, target):
    """
    Create improved features for time series prediction with focus on capturing peaks
    """
    print("Creating time-series enhanced features...\n")

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

            #print("Added sample sequence features for 9 samples per month")
        except Exception as e:
            print(f"Could not create sample sequence features: {e}")

        # One-hot encode month for better representation of seasonality
        month_dummies = pd.get_dummies(df['Month'], prefix='month')
        df = pd.concat([df, month_dummies], axis=1)

    # Drop rows with NaN due to lag/rolling features
    original_rows = len(df)
    df = df.dropna()
    #print(f"Dropped {original_rows - len(df)} rows due to NaN values from feature engineering")

    return df


# ============= MODEL TRAINING =============
def train_models(X_train, y_train, selected_features, tscv):
    """
    Train and tune Gradient Boosting models with simplified parameters
    """

    # 2. Gradient Boosting Hyperparameter Tuning using BayesSearchCV with simplified parameters
    print("\nTuning Gradient Boosting hyperparameters with BayesSearchCV...\n")

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
        n_jobs=1
    )

    # Fit the Bayesian search
    bayes_search.fit(X_train[selected_features], y_train)

    # Best parameters
   # print("Best Gradient Boosting parameters found:")
    #for param, value in bayes_search.best_params_.items():
       # print(f"  - {param}: {value}")

    # Train GB with best parameters
    best_gb = bayes_search.best_estimator_
    print("Model has found the best parameters and successfully trained the "
                                               "dataset!\n\n")

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
    #print("\n=== Evaluating models ===")

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
    #print("\n===== TRAINING DATA PERFORMANCE =====")
   #print(f"Gradient Boosting - Train R²: {gb_train_r2:.4f}, Train RMSE: {gb_train_rmse:.4f}")

    gb_pred = gb_pipeline.predict(X_test[selected_features])
    if use_log:
        gb_pred = np.expm1(gb_pred)  # Convert back from log

    # Evaluate models on test set
    #print("\n===== TEST DATA PERFORMANCE =====")

    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
   # print(f"Gradient Boosting - Test R²: {gb_r2:.4f}, Test RMSE: {gb_rmse:.4f}")

    # Print date, predicted values, and actual values
   # print("\n===== DATE, PREDICTED VALUES, AND ACTUAL VALUES =====")
   # print_prediction_results(X_test, y_test, gb_pred, df)

    # Calculate and print the difference between train and test performance to check for overfitting
   # print("\n===== TRAIN VS TEST COMPARISON =====")
   # print(f"Gradient Boosting - R² difference (train-test): {gb_train_r2 - gb_r2:.4f}")
   # print(f"Gradient Boosting - RMSE difference (test-train): {gb_rmse - gb_train_rmse:.4f}")

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
    #plt.show()

    # Add cross-validation scoring
    from sklearn.model_selection import cross_val_score

    # 5-fold cross-validation for Gradient Boosting
    gb_cv_scores = cross_val_score(gb_pipeline, X_train[selected_features],
                                   y_train if not use_log else y_log_train,
                                   cv=5, scoring='r2')
   # print(f"Gradient Boosting CV R² scores: {gb_cv_scores}")
   # print(f"Gradient Boosting CV R² mean: {gb_cv_scores.mean():.4f}, std: {gb_cv_scores.std():.4f}")

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
    #plt.show()

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
    #plt.show()

    return gb_pred, gb_r2, gb_rmse


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
   # print("\n=== Saving models to disk ===")

    # Save the gradient boosting pipeline model
    with open('pkl/chlorophyll_gb_model.pkl', 'wb') as file:
        pickle.dump(gb_pipeline, file)
 #   print("Gradient Boosting pipeline saved as 'chlorophyll_gb_model.pkl'")

    # Save selected features for future reference
    with open('pkl/selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)
   # print(f"Selected features list saved as 'selected_features.pkl'")

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
   # print("Model metadata saved as 'chlorophyll_gb_model_metadata.pkl'")


def plot_selected_features(X_train, feature_importance_df, selected_features, target_name='Target'):
    """
    Create comprehensive visualizations for selected features
    """

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Filter feature importance for selected features only
    selected_importance = feature_importance_df[
        feature_importance_df['feature'].isin(selected_features)
    ].copy()

    # ============= 1. SELECTED FEATURES IMPORTANCE BAR PLOT =============
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Horizontal bar plot
    sns.barplot(data=selected_importance, x='importance', y='feature', ax=ax1)
    ax1.set_title(f'Top {len(selected_features)} Selected Features - Importance Scores',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature Importance')
    ax1.tick_params(axis='y', labelsize=10)

    # Vertical bar plot for better readability of feature names
    plt.sca(ax2)
    bars = ax2.bar(range(len(selected_importance)), selected_importance['importance'])
    ax2.set_title(f'Feature Importance Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Feature Rank')
    ax2.set_ylabel('Importance Score')
    ax2.set_xticks(range(0, len(selected_importance), 5))
    ax2.set_xticklabels(range(1, len(selected_importance) + 1, 5))

    # Add value labels on bars for top 10
    for i, bar in enumerate(bars[:10]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    #plt.show()

    # ============= 2. FEATURE CATEGORIES ANALYSIS =============
    # Categorize features by type
    categories = {
        'Original': [],
        'Lagged': [],
        'Rolling': [],
        'Difference': [],
        'Seasonal': [],
        'Interaction': []
    }

    for feature in selected_features:
        if '_lag' in feature:
            categories['Lagged'].append(feature)
        elif '_roll_' in feature:
            categories['Rolling'].append(feature)
        elif '_diff' in feature or '_pct_change' in feature:
            categories['Difference'].append(feature)
        elif any(season in feature for season in ['Season', 'month', 'Day', 'Year']):
            categories['Seasonal'].append(feature)
        elif '_interact' in feature:
            categories['Interaction'].append(feature)
        else:
            categories['Original'].append(feature)

    # Plot category distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Count by category
    category_counts = {k: len(v) for k, v in categories.items() if len(v) > 0}

    # Pie chart
    ax1.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
    ax1.set_title('Selected Features by Category', fontsize=14, fontweight='bold')

    # Bar chart with feature names
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    bars = ax2.bar(category_counts.keys(), category_counts.values(), color=colors)
    ax2.set_title('Feature Count by Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Features')
    plt.xticks(rotation=45)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
   # plt.show()

    # Print category breakdown
    print("\n=== SELECTED FEATURES BY CATEGORY ===")
    for category, features in categories.items():
        if features:
            print(f"\n{category} Features ({len(features)}):")
            for i, feature in enumerate(features, 1):
                importance = selected_importance[selected_importance['feature'] == feature]['importance'].iloc[0]
                print(f"  {i:2d}. {feature:<40} (importance: {importance:.4f})")


def analyze_feature_pairs(X_train, selected_features, target_series=None, top_n=15):
    """
    Analyze and visualize top feature pairs based on correlation and interaction strength
    """

    print(f"\n=== ANALYZING TOP {top_n} FEATURE PAIRS ===")

    # Calculate correlation matrix for selected features
    X_selected = X_train[selected_features]
    corr_matrix = X_selected.corr().abs()

    # Get upper triangle of correlation matrix to avoid duplicates
    upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Extract feature pairs and their correlations
    feature_pairs = []
    for i in range(len(selected_features)):
        for j in range(i + 1, len(selected_features)):
            if upper_triangle[i, j]:
                correlation = corr_matrix.iloc[i, j]
                feature_pairs.append({
                    'feature1': selected_features[i],
                    'feature2': selected_features[j],
                    'correlation': correlation,
                    'pair_name': f"{selected_features[i]} + {selected_features[j]}"
                })

    # Sort by correlation strength
    feature_pairs_df = pd.DataFrame(feature_pairs)
    feature_pairs_df = feature_pairs_df.sort_values('correlation', ascending=False)

    # ============= 1. TOP CORRELATED PAIRS HEATMAP =============
    top_pairs = feature_pairs_df.head(top_n)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # Correlation heatmap for top pairs
    top_features_for_heatmap = list(set(
        list(top_pairs['feature1']) + list(top_pairs['feature2'])
    ))[:min(15, len(top_pairs) * 2)]  # Limit to prevent overcrowding

    corr_subset = X_selected[top_features_for_heatmap].corr()
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0,
                square=True, ax=ax1, fmt='.2f', cbar_kws={'shrink': 0.8})
    ax1.set_title('Correlation Heatmap - Top Feature Pairs', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', rotation=0, labelsize=10)

    # ============= 2. TOP PAIRS BAR CHART =============
    bars = ax2.barh(range(len(top_pairs)), top_pairs['correlation'])
    ax2.set_yticks(range(len(top_pairs)))
    ax2.set_yticklabels([f"{pair['feature1'][:20]}...\n+ {pair['feature2'][:20]}..."
                         if len(pair['feature1']) > 20 or len(pair['feature2']) > 20
                         else f"{pair['feature1']}\n+ {pair['feature2']}"
                         for _, pair in top_pairs.iterrows()], fontsize=8)
    ax2.set_xlabel('Absolute Correlation')
    ax2.set_title(f'Top {top_n} Most Correlated Feature Pairs', fontsize=14, fontweight='bold')

    # Add correlation values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{width:.3f}', ha='left', va='center', fontsize=8)

    # ============= 3. SCATTER PLOTS OF TOP 4 PAIRS =============
    top_4_pairs = top_pairs.head(4)

    for idx, (_, pair) in enumerate(top_4_pairs.iterrows()):
        if idx < 2:
            ax = ax3 if idx == 0 else ax4
        else:
            # Create additional subplots if needed
            continue

        feat1, feat2 = pair['feature1'], pair['feature2']

        # Create scatter plot
        ax.scatter(X_selected[feat1], X_selected[feat2], alpha=0.6, s=30)
        ax.set_xlabel(feat1[:30] + '...' if len(feat1) > 30 else feat1)
        ax.set_ylabel(feat2[:30] + '...' if len(feat2) > 30 else feat2)
        ax.set_title(f'Correlation: {pair["correlation"]:.3f}', fontsize=12)

        # Add trend line
        try:
            z = np.polyfit(X_selected[feat1].dropna(), X_selected[feat2].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(X_selected[feat1], p(X_selected[feat1]), "r--", alpha=0.8)
        except:
            pass

    # If we have less than 4 pairs, hide unused subplots
    if len(top_4_pairs) < 3:
        ax3.set_visible(False)
    if len(top_4_pairs) < 4:
        ax4.set_visible(False)

    plt.tight_layout()
    #plt.show()

    # ============= 4. FEATURE PAIR ANALYSIS WITH TARGET =============
    if target_series is not None:
        print("\n=== FEATURE PAIRS CORRELATION WITH TARGET ===")

        # Calculate correlation of each feature with target
        target_correlations = []
        for feature in selected_features:
            try:
                corr, _ = pearsonr(X_selected[feature].dropna(),
                                   target_series[X_selected[feature].dropna().index])
                target_correlations.append({
                    'feature': feature,
                    'target_correlation': abs(corr)
                })
            except:
                target_correlations.append({
                    'feature': feature,
                    'target_correlation': 0
                })

        target_corr_df = pd.DataFrame(target_correlations).sort_values(
            'target_correlation', ascending=False
        )

        # Plot target correlations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Top features correlated with target
        top_target_corr = target_corr_df.head(15)
        bars = ax1.barh(range(len(top_target_corr)), top_target_corr['target_correlation'])
        ax1.set_yticks(range(len(top_target_corr)))
        ax1.set_yticklabels([feat[:35] + '...' if len(feat) > 35 else feat
                             for feat in top_target_corr['feature']], fontsize=10)
        ax1.set_xlabel('Absolute Correlation with Target')
        ax1.set_title('Features Most Correlated with Target', fontsize=14, fontweight='bold')

        # Add correlation values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=9)

        # Distribution of target correlations
        ax2.hist(target_corr_df['target_correlation'], bins=15, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Absolute Correlation with Target')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Distribution of Target Correlations', fontsize=14, fontweight='bold')
        ax2.axvline(target_corr_df['target_correlation'].mean(), color='red',
                    linestyle='--', label=f'Mean: {target_corr_df["target_correlation"].mean():.3f}')
        ax2.legend()

        plt.tight_layout()
        #plt.show()

    # Print top pairs summary
    print(f"\nTop {min(10, len(feature_pairs_df))} Feature Pairs by Correlation:")
    for i, (_, pair) in enumerate(feature_pairs_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {pair['feature1']:<35} + {pair['feature2']:<35} | Corr: {pair['correlation']:.4f}")

    return feature_pairs_df, top_pairs


def analyze_non_chlorophyll_features(selected_features, feature_importance_df, X_train, y_train=None):
    """
    Specifically analyze non-chlorophyll features and their importance
    """
    print("\n" + "=" * 80)
    print("NON-CHLOROPHYLL FEATURES ANALYSIS")
    print("=" * 80)

    # Define your specific parameters for accurate categorization
    # Original parameters from your dataset:
    # pH (units), Ammonia (mg/L), Nitrate (mg/L), Inorganic Phosphate (mg/L),
    # Dissolved Oxygen (mg/L), Temperature, Chlorophyll-a (ug/L), Station,
    # Solar Mean, Solar Max, Solar Min, Phytoplankton

    chlorophyll_keywords = ['chlorophyll', 'chl', 'chla', 'chl_a']

    # Separate chlorophyll and non-chlorophyll features
    chlorophyll_features = []
    non_chlorophyll_features = []

    for feature in selected_features:
        is_chlorophyll = any(keyword.lower() in feature.lower() for keyword in chlorophyll_keywords)
        if is_chlorophyll:
            chlorophyll_features.append(feature)
        else:
            non_chlorophyll_features.append(feature)

    print(f"Total Selected Features: {len(selected_features)}")
    print(f"Chlorophyll-related Features: {len(chlorophyll_features)}")
    print(f"Non-Chlorophyll Features: {len(non_chlorophyll_features)}")

    # Get importance scores for non-chlorophyll features
    non_chl_importance = feature_importance_df[
        feature_importance_df['feature'].isin(non_chlorophyll_features)
    ].copy().sort_values('importance', ascending=False)

    chl_importance = feature_importance_df[
        feature_importance_df['feature'].isin(chlorophyll_features)
    ].copy().sort_values('importance', ascending=False)

    # ============= VISUALIZATION =============
    fig = plt.figure(figsize=(24, 16))

    # 1. Comparison bar chart
    ax1 = plt.subplot(2, 3, 1)
    categories = ['Non-Chlorophyll', 'Chlorophyll']
    counts = [len(non_chlorophyll_features), len(chlorophyll_features)]
    colors = ['skyblue', 'lightgreen']
    bars = ax1.bar(categories, counts, color=colors)
    ax1.set_title('Feature Count: Non-Chlorophyll vs Chlorophyll', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Features')

    # Add count labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # 2. Non-chlorophyll features importance
    ax2 = plt.subplot(2, 3, 2)
    if len(non_chl_importance) > 0:
        top_non_chl = non_chl_importance.head(15)
        bars = ax2.barh(range(len(top_non_chl)), top_non_chl['importance'], color='skyblue')
        ax2.set_yticks(range(len(top_non_chl)))
        ax2.set_yticklabels([feat[:30] + '...' if len(feat) > 30 else feat
                             for feat in top_non_chl['feature']], fontsize=9)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title(f'Top {min(15, len(non_chl_importance))} Non-Chlorophyll Features',
                      fontsize=12, fontweight='bold')

        # Add importance values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No non-chlorophyll features found',
                 ha='center', va='center', transform=ax2.transAxes)

    # 3. Chlorophyll features importance
    ax3 = plt.subplot(2, 3, 3)
    if len(chl_importance) > 0:
        bars = ax3.barh(range(len(chl_importance)), chl_importance['importance'], color='lightgreen')
        ax3.set_yticks(range(len(chl_importance)))
        ax3.set_yticklabels([feat[:30] + '...' if len(feat) > 30 else feat
                             for feat in chl_importance['feature']], fontsize=9)
        ax3.set_xlabel('Feature Importance')
        ax3.set_title(f'Chlorophyll-Related Features ({len(chl_importance)})',
                      fontsize=12, fontweight='bold')

        # Add importance values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{width:.3f}', ha='left', va='center', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No chlorophyll features found',
                 ha='center', va='center', transform=ax3.transAxes)

    # 4. Importance distribution comparison
    ax4 = plt.subplot(2, 3, 4)
    if len(non_chl_importance) > 0 and len(chl_importance) > 0:
        ax4.hist(non_chl_importance['importance'], bins=10, alpha=0.7,
                 label='Non-Chlorophyll', color='skyblue', edgecolor='black')
        ax4.hist(chl_importance['importance'], bins=10, alpha=0.7,
                 label='Chlorophyll', color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Feature Importance')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Importance Distribution Comparison', fontsize=12, fontweight='bold')
        ax4.legend()

    # 5. Categorize non-chlorophyll features by type based on your specific parameters
    ax5 = plt.subplot(2, 3, 5)
    non_chl_categories = {
        'Chemical Parameters': [],
        'Physical Parameters': [],
        'Solar/Light': [],
        'Location/Station': [],
        'Temporal Features': [],
        'Engineered Features': [],
        'Other': []
    }

    # Your specific parameter keywords
    chemical_keywords = ['ph', 'ammonia', 'nitrate', 'phosphate', 'oxygen', 'dissolved']
    physical_keywords = ['temperature', 'temp']
    solar_keywords = ['solar', 'light', 'radiation']
    location_keywords = ['station', 'site', 'location']
    temporal_keywords = ['month', 'day', 'year', 'season', 'time', 'date', 'dayofyear', 'dayofmonth']
    engineered_keywords = ['lag', 'roll', 'diff', 'pct_change', 'interact', 'mean', 'std', 'max', 'min']

    for feature in non_chlorophyll_features:
        feature_lower = feature.lower()
        if any(kw in feature_lower for kw in engineered_keywords):
            non_chl_categories['Engineered Features'].append(feature)
        elif any(kw in feature_lower for kw in temporal_keywords):
            non_chl_categories['Temporal Features'].append(feature)
        elif any(kw in feature_lower for kw in solar_keywords):
            non_chl_categories['Solar/Light'].append(feature)
        elif any(kw in feature_lower for kw in location_keywords):
            non_chl_categories['Location/Station'].append(feature)
        elif any(kw in feature_lower for kw in physical_keywords):
            non_chl_categories['Physical Parameters'].append(feature)
        elif any(kw in feature_lower for kw in chemical_keywords):
            non_chl_categories['Chemical Parameters'].append(feature)
        else:
            non_chl_categories['Other'].append(feature)

    # Plot category distribution
    category_counts = {k: len(v) for k, v in non_chl_categories.items() if len(v) > 0}
    if category_counts:
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
        wedges, texts, autotexts = ax5.pie(category_counts.values(), labels=category_counts.keys(),
                                           autopct='%1.1f%%', colors=colors)
        ax5.set_title('Non-Chlorophyll Features by Category', fontsize=12, fontweight='bold')

    # 6. Top non-chlorophyll features with target correlation (if available)
    ax6 = plt.subplot(2, 3, 6)
    if y_train is not None and len(non_chlorophyll_features) > 0:
        # Calculate correlations with target
        target_corrs = []
        for feature in non_chlorophyll_features:
            try:
                if feature in X_train.columns:
                    corr, _ = pearsonr(X_train[feature].dropna(),
                                       y_train[X_train[feature].dropna().index])
                    target_corrs.append({
                        'feature': feature,
                        'correlation': abs(corr)
                    })
            except:
                continue

        if target_corrs:
            target_corr_df = pd.DataFrame(target_corrs).sort_values('correlation', ascending=False)
            top_corr = target_corr_df.head(10)

            bars = ax6.barh(range(len(top_corr)), top_corr['correlation'], color='orange')
            ax6.set_yticks(range(len(top_corr)))
            ax6.set_yticklabels([feat[:25] + '...' if len(feat) > 25 else feat
                                 for feat in top_corr['feature']], fontsize=9)
            ax6.set_xlabel('Absolute Correlation with Target')
            ax6.set_title('Non-Chlorophyll Features:\nStrongest Target Correlation',
                          fontsize=12, fontweight='bold')

            # Add correlation values
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax6.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    #plt.show()

    # ============= DETAILED ANALYSIS PRINTOUT =============
    print(f"\n=== NON-CHLOROPHYLL FEATURES BREAKDOWN ===")

    if len(non_chl_importance) > 0:
        print(f"\nTop 10 Most Important Non-Chlorophyll Features:")
        for i, (_, row) in enumerate(non_chl_importance.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:<45} (importance: {row['importance']:.4f})")

        print(f"\nNon-Chlorophyll Feature Categories:")
        for category, features in non_chl_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                for feature in features[:3]:  # Show first 3 in each category
                    importance = non_chl_importance[non_chl_importance['feature'] == feature]['importance']
                    if len(importance) > 0:
                        print(f"    - {feature} (imp: {importance.iloc[0]:.4f})")
                if len(features) > 3:
                    print(f"    ... and {len(features) - 3} more")

        # Show which original parameters are most important
      #  print(f"\n=== ORIGINAL PARAMETER IMPORTANCE ===")
        original_params = {
            'pH': ['ph'],
            'Ammonia': ['ammonia'],
            'Nitrate': ['nitrate'],
            'Phosphate': ['phosphate'],
            'Dissolved Oxygen': ['oxygen', 'dissolved'],
            'Temperature': ['temperature', 'temp'],
            'Solar': ['solar'],
            'Station': ['station']
        }

        param_importance = {}
        for param_name, keywords in original_params.items():
            matching_features = []
            for feature in non_chlorophyll_features:
                if any(kw in feature.lower() for kw in keywords):
                    matching_features.append(feature)

            if matching_features:
                # Get average importance for this parameter type
                importances = []
                for feature in matching_features:
                    imp = non_chl_importance[non_chl_importance['feature'] == feature]['importance']
                    if len(imp) > 0:
                        importances.append(imp.iloc[0])

                if importances:
                    param_importance[param_name] = {
                        'avg_importance': np.mean(importances),
                        'max_importance': np.max(importances),
                        'feature_count': len(matching_features),
                        'top_feature': matching_features[np.argmax(importances)]
                    }

        # Sort by average importance
        sorted_params = sorted(param_importance.items(), key=lambda x: x[1]['avg_importance'], reverse=True)

       # print("Ranking of Original Parameters (by average feature importance):")
        #for i, (param, info) in enumerate(sorted_params, 1):
          #  print(f"  {i}. {param:<20} - Avg: {info['avg_importance']:.4f}, Max: {info['max_importance']:.4f}")
          #  print(f"     └─ {info['feature_count']} features, Top: {info['top_feature']}")

    #if len(chl_importance) > 0:
       # print(f"\n=== CHLOROPHYLL FEATURES FOR COMPARISON ===")
        #print(f"Top Chlorophyll-Related Features:")
        #for i, (_, row) in enumerate(chl_importance.head(5).iterrows(), 1):
            #print(f"  {i}. {row['feature']:<45} (importance: {row['importance']:.4f})")

    # Statistical comparison
   # if len(non_chl_importance) > 0 and len(chl_importance) > 0:
    #    print(f"\n=== STATISTICAL COMPARISON ===")
     #   print(f"Non-Chlorophyll Features - Mean Importance: {non_chl_importance['importance'].mean():.4f}")
      #  print(f"Chlorophyll Features - Mean Importance: {chl_importance['importance'].mean():.4f}")

      #  if non_chl_importance['importance'].mean() > chl_importance['importance'].mean():
       #     print("→ Non-chlorophyll features have higher average importance!")
       # else:
           # print("→ Chlorophyll features have higher average importance")

    return {
        'non_chlorophyll_features': non_chlorophyll_features,
        'chlorophyll_features': chlorophyll_features,
        'non_chl_importance': non_chl_importance,
        'chl_importance': chl_importance,
        'categories': non_chl_categories
    }


def create_feature_summary_report(selected_features, feature_importance_df, X_train):
    """
    Create a comprehensive summary report of selected features
    """
    print("\n" + "=" * 80)
    print("FEATURE SELECTION SUMMARY REPORT")
    print("=" * 80)

    selected_importance = feature_importance_df[
        feature_importance_df['feature'].isin(selected_features)
    ].copy()

    print(f"Total Features Available: {X_train.shape[1]}")
    print(f"Features Selected: {len(selected_features)} ({len(selected_features) / X_train.shape[1] * 100:.1f}%)")
    print(f"Dimensionality Reduction: {X_train.shape[1] - len(selected_features)} features removed")

    print(f"\nImportance Score Statistics:")
    print(f"  Mean: {selected_importance['importance'].mean():.4f}")
    print(f"  Std:  {selected_importance['importance'].std():.4f}")
    print(f"  Min:  {selected_importance['importance'].min():.4f}")
    print(f"  Max:  {selected_importance['importance'].max():.4f}")

    print(f"\nTop 5 Most Important Features:")
    for i, (_, row) in enumerate(selected_importance.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']:<40} ({row['importance']:.4f})")

    return selected_importance


# ============= MAIN EXECUTION FUNCTION =============
def run_feature_analysis(X_train, y_train, selected_features, feature_importance_df):
    """
    Run complete feature analysis and visualization

    Parameters:
    - X_train: Training features DataFrame
    - y_train: Training target Series
    - selected_features: List of selected feature names
    - feature_importance_df: DataFrame with 'feature' and 'importance' columns
    """

    print("Starting comprehensive feature analysis...")

    # 1. Plot selected features importance
    plot_selected_features(X_train, feature_importance_df, selected_features)

    # 2. Analyze feature pairs
    feature_pairs_df, top_pairs = analyze_feature_pairs(X_train, selected_features, y_train)

    # 3. Analyze non-chlorophyll features specifically
    non_chl_analysis = analyze_non_chlorophyll_features(selected_features, feature_importance_df, X_train, y_train)

    # 4. Create summary report
    summary = create_feature_summary_report(selected_features, feature_importance_df, X_train)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    return {
        'feature_pairs': feature_pairs_df,
        'top_pairs': top_pairs,
        'summary': summary,
        'non_chlorophyll_analysis': non_chl_analysis
    }


# ============= MAIN FUNCTION =============
def main():
    """
    Main function to run the entire model training pipeline
    """
    # Load data
    file_path = 'CSV/merged_stations_mag.xlsx'
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
    #plt.show()

    # Log transform target if skewed
    skewness = stats.skew(df[target])
    #print(f"Skewness of {target}: {skewness}")

    # Always apply log transformation for environmental data
    print("Applying log transformation to target...\n")
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
           # print(f"Warning: Non-numeric column detected: {col}, type: {df[col].dtype}")
            all_features.remove(col)

    # Print feature list
    #print(f"Total features after engineering: {len(all_features)}")

    # Prepare data
    X = df[all_features]
    y = df[target]
    y_log = df[log_target] if use_log else y

    # Setup time series cross-validation with more splits
    #print("\nSetting up time series cross-validation...")
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

        #print(f"Train-Test split date: {split_date}")

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

   #print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
    #print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

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
    print("\nModel analysis completed successfully!")
    print(f"Tuned Gradient Boosting: R² = {gb_r2:.4f}, RMSE = {gb_rmse:.4f}")
    print ("\nYou may now use the forecasting!")

    # Run the complete analysis:
    #results = run_feature_analysis(X_train, y_train, selected_features, feature_importance)

    # Access results:
   # feature_pairs_df = results['feature_pairs']
    #top_pairs = results['top_pairs']
    #summary_stats = results['summary']
    #non_chl_analysis = results['non_chlorophyll_analysis']

    # You can also run just the non-chlorophyll analysis separately:
    #non_chl_results = analyze_non_chlorophyll_features(selected_features, feature_importance, X_train, y_train)


if __name__ == "__main__":
    main()