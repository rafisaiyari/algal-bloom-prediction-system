import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
import pickle

warnings.filterwarnings('ignore')


# ============= LOAD AND PREPROCESS DATA =============
def load_and_preprocess_data(file_path):
    """
    Load and preprocess water quality data
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
    df = df.dropna(subset=['Chlorophyll-a (ug/L)'])

    # Extract station information
    if 'Station' in df.columns:
        print("Processing station information...")
        # One-hot encode the Station column
        station_dummies = pd.get_dummies(df['Station'], prefix='station')
        df = pd.concat([df, station_dummies], axis=1)
        print(f"Created {station_dummies.shape[1]} station dummy variables")

    # Impute missing values
    cols_with_na = df.columns[df.isna().any()].tolist()
    print(f"Columns with missing values: {cols_with_na}")

    # For numeric columns, impute with median (more robust than mean for skewed data)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isna().any():
            na_count = df[col].isna().sum()
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {na_count} missing values in '{col}' with median: {median_val:.4f}")

    # Drop all remaining object/string columns that can't be converted to numeric
    print("\nDropping remaining non-numeric columns:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"  - Dropping: {col}")
        df = df.drop(columns=[col])

    # Handle outliers
    numeric_columns = df.select_dtypes(include=np.number).columns
    target = 'Chlorophyll-a (ug/L)'

    for col in numeric_columns:
        if col != target:  # Handle target separately
            detect_and_handle_outliers(df, col, method='IQR', threshold=3)

    # Handle target outliers separately
    detect_and_handle_outliers(df, target, method='IQR', threshold=3)

    # Ensure data is sorted by date if a date column exists
    if 'Date' in df.columns:
        df = df.sort_values('Date')
        print("Sorted data by Date")

    return df


def detect_and_handle_outliers(df, column, method='IQR', threshold=3):
    """
    Detect and handle outliers in a column using either IQR or z-score method
    """
    if method == 'IQR':
        # IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        outliers = ((df[column] < lower_bound) | (df[column] > upper_bound))
    else:
        # Z-score method
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = df[column].notna() & pd.Series(z_scores > threshold, index=df[column].dropna().index)

    outlier_count = outliers.sum()

    if outlier_count > 0:
        print(f"Found {outlier_count} outliers in '{column}' using {method} method")
        # Cap outliers instead of removing them
        if method == 'IQR':
            df.loc[df[column] < lower_bound, column] = lower_bound
            df.loc[df[column] > upper_bound, column] = upper_bound
        else:
            # Calculate bounds for z-score method
            mean_val = df[column].mean()
            std_val = df[column].std()
            df.loc[df[column] < mean_val - threshold * std_val, column] = mean_val - threshold * std_val
            df.loc[df[column] > mean_val + threshold * std_val, column] = mean_val + threshold * std_val
        print(f"Capped outliers in '{column}'")

    return outlier_count


# ============= FEATURE ENGINEERING =============
def engineer_features(df, features, target):
    """
    Create advanced features for time series prediction
    """
    print("\n=== Creating advanced features ===")

    # Create lagged features
    for feature in features + [target]:
        for lag in [1, 2, 3]:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

    # Create rolling mean and standard deviation features
    for feature in features + [target]:
        for window in [3, 7, 14]:
            df[f'{feature}_roll_mean{window}'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_roll_std{window}'] = df[feature].rolling(window=window).std()

    # Create interaction terms between all pairs of features
    print("Creating interaction features...")
    for i, feat1 in enumerate(features):
        for feat2 in features[i + 1:]:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f'{feat1.split(" ")[0]}_{feat2.split(" ")[0]}_interact'
                df[interaction_name] = df[feat1] * df[feat2]
                print(f"  - Created {interaction_name}")

    # Create polynomial features
    print("Creating polynomial features...")
    for feature in features:
        if feature in df.columns:
            df[f'{feature}_squared'] = df[feature] ** 2
            print(f"  - Created {feature}_squared")

    # Month as a proxy for seasonality if Date column exists
    if 'Date' in df.columns:
        print("Creating seasonal features...")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfYear'] = df['Date'].dt.dayofyear

        # Create seasonal features using sine and cosine transforms
        df['Season_sin'] = np.sin(df['DayOfYear'] * (2 * np.pi / 365))
        df['Season_cos'] = np.cos(df['DayOfYear'] * (2 * np.pi / 365))

        # One-hot encode month for better representation of seasonality
        month_dummies = pd.get_dummies(df['Month'], prefix='month')
        df = pd.concat([df, month_dummies], axis=1)

    # Drop rows with NaN due to lag/rolling features
    original_rows = len(df)
    df = df.dropna()
    print(f"Dropped {original_rows - len(df)} rows due to NaN values from feature engineering")

    return df


# ============= FEATURE SELECTION =============
def select_features(X_train, y_train, use_log=False):
    """
    Select important features using Random Forest feature importance
    """
    print("\n=== Performing feature selection ===")

    # Use a more efficient approach than RFECV - Random Forest feature importance
    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
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

    # Select top features - keep top 30% or at least 20 features
    num_features = max(int(len(X_train.columns) * 0.3), 20)
    num_features = min(num_features, len(X_train.columns))  # Ensure we don't select more than we have
    selected_features = feature_importance['feature'][:num_features].values
    print(f"Selected {len(selected_features)} out of {len(X_train.columns)} features")

    return selected_features, feature_importance


# ============= MODEL TRAINING =============
def train_models(X_train, y_train, selected_features, tscv):
    """
    Train and tune Random Forest and Gradient Boosting models
    """
    print("\n=== Training and tuning models ===")

    # 1. Random Forest Hyperparameter Tuning
    print("\nTuning Random Forest hyperparameters...")
    # Parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Use RandomizedSearchCV for efficiency
    rf_random = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=rf_param_grid,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Fit the random search
    rf_random.fit(X_train[selected_features], y_train)

    # Best parameters
    print("Best Random Forest parameters found:")
    for param, value in rf_random.best_params_.items():
        print(f"  - {param}: {value}")

    # Train RF with best parameters
    best_rf = rf_random.best_estimator_
    print("\nTraining Random Forest with best parameters...")

    # 2. Gradient Boosting Hyperparameter Tuning
    print("\nTuning Gradient Boosting hyperparameters...")
    # Parameter grid for Gradient Boosting
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Use RandomizedSearchCV for efficiency
    gb_random = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions=gb_param_grid,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Fit the random search
    gb_random.fit(X_train[selected_features], y_train)

    # Best parameters
    print("Best Gradient Boosting parameters found:")
    for param, value in gb_random.best_params_.items():
        print(f"  - {param}: {value}")

    # Train GB with best parameters
    best_gb = gb_random.best_estimator_
    print("\nTraining Gradient Boosting with best parameters...")

    # 3. Create Pipeline with Robust Scaling for Gradient Boosting
    gb_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', best_gb)
    ])

    # Fit the pipeline
    gb_pipeline.fit(X_train[selected_features], y_train)

    return best_rf, gb_pipeline, rf_random, gb_random


# ============= MODEL EVALUATION =============
def evaluate_models(best_rf, gb_pipeline, X_train, X_test, y_train, y_test, selected_features, use_log=False):
    """
    Evaluate models on training and test datasets
    """
    print("\n=== Evaluating models ===")

    # Calculate R2 on training data for Random Forest
    rf_train_pred = best_rf.predict(X_train[selected_features])
    if use_log:
        rf_train_pred = np.expm1(rf_train_pred)  # Convert back from log
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))

    # Calculate R2 on training data for Gradient Boosting
    gb_train_pred = gb_pipeline.predict(X_train[selected_features])
    if use_log:
        gb_train_pred = np.expm1(gb_train_pred)  # Convert back from log
    gb_train_r2 = r2_score(y_train, gb_train_pred)
    gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))

    # Print the training results
    print("\n===== TRAINING DATA PERFORMANCE =====")
    print(f"Random Forest - Train R²: {rf_train_r2:.4f}, Train RMSE: {rf_train_rmse:.4f}")
    print(f"Gradient Boosting - Train R²: {gb_train_r2:.4f}, Train RMSE: {gb_train_rmse:.4f}")

    # Predictions on test data
    rf_pred = best_rf.predict(X_test[selected_features])
    if use_log:
        rf_pred = np.expm1(rf_pred)  # Convert back from log

    gb_pred = gb_pipeline.predict(X_test[selected_features])
    if use_log:
        gb_pred = np.expm1(gb_pred)  # Convert back from log

    # Evaluate models on test set
    print("\n===== TEST DATA PERFORMANCE =====")
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"Random Forest - Test R²: {rf_r2:.4f}, Test RMSE: {rf_rmse:.4f}")

    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    print(f"Gradient Boosting - Test R²: {gb_r2:.4f}, Test RMSE: {gb_rmse:.4f}")

    # Calculate and print the difference between train and test performance to check for overfitting
    print("\n===== TRAIN VS TEST COMPARISON =====")
    print(f"Random Forest - R² difference (train-test): {rf_train_r2 - rf_r2:.4f}")
    print(f"Random Forest - RMSE difference (test-train): {rf_rmse - rf_train_rmse:.4f}")
    print(f"Gradient Boosting - R² difference (train-test): {gb_train_r2 - gb_r2:.4f}")
    print(f"Gradient Boosting - RMSE difference (test-train): {gb_rmse - gb_train_rmse:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(15, 8))

    # Create a common x-axis for comparison
    x_values = np.arange(len(y_test))

    # Plot actual values and predictions at the same x locations
    plt.plot(x_values, y_test, label='Actual', color='blue', linewidth=2)
    plt.plot(x_values, rf_pred, label=f'Random Forest (R²={rf_r2:.3f})', linestyle='--', color='red')
    plt.plot(x_values, gb_pred, label=f'Gradient Boosting (R²={gb_r2:.3f})', linestyle='--', color='green')

    plt.title("Comparison of Tuned Models")
    plt.xlabel("Sample Index")
    plt.ylabel("Chlorophyll-a (ug/L)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return rf_pred, gb_pred, rf_r2, gb_r2, rf_rmse, gb_rmse


# ============= FEATURE IMPORTANCE ANALYSIS =============
def analyze_feature_importance(best_rf, gb_pipeline, selected_features):
    """
    Analyze and visualize feature importances
    """
    print("\n=== Analyzing feature importance ===")

    # Get feature importances for Random Forest
    rf_importances = pd.DataFrame({
        'feature': selected_features,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Random Forest feature importances:")
    print(rf_importances.head(15))

    # Get feature importances for Gradient Boosting
    gb_importances = pd.DataFrame({
        'feature': selected_features,
        'importance': gb_pipeline.named_steps['model'].feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Gradient Boosting feature importances:")
    print(gb_importances.head(15))

    # Plot feature importances
    plt.figure(figsize=(14, 10))
    plt.subplot(1, 2, 1)
    sns.barplot(x='importance', y='feature', data=rf_importances.head(15))
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    sns.barplot(x='importance', y='feature', data=gb_importances.head(15))
    plt.title('Gradient Boosting Feature Importance')
    plt.tight_layout()

    plt.show()

    return rf_importances, gb_importances


# ============= PLOT LEARNING CURVES =============
def plot_learning_curve(estimator, X, y, title, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a plot of the learning curve for an estimator
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (neg MSE)")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='neg_mean_squared_error')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    return None


# ============= SAVE MODELS =============
def save_models(best_model, gb_pipeline, selected_features, use_log, rf_r2, gb_r2, rf_rmse, gb_rmse, gb_importances):
    """
    Save models, selected features, and metadata to disk
    """
    print("\n=== Saving models to disk ===")

    # Save the gradient boosting pipeline model
    with open('chlorophyll_gb_model.pkl', 'wb') as file:
        pickle.dump(gb_pipeline, file)
    print("Gradient Boosting pipeline saved as 'chlorophyll_gb_model.pkl'")

    # Save selected features for future reference
    with open('selected_features.pkl', 'wb') as file:
        pickle.dump(selected_features, file)
    print(f"Selected features list saved as 'selected_features.pkl'")

    # Optional: Save model metadata
    model_metadata = {
        'r2_score': gb_r2,
        'rmse': gb_rmse,
        'use_log_transform': use_log,
        'feature_importances': gb_importances.to_dict(),
        'model_parameters': gb_pipeline.named_steps['model'].get_params(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    }

    with open('chlorophyll_gb_model_metadata.pkl', 'wb') as file:
        pickle.dump(model_metadata, file)
    print("Model metadata saved as 'chlorophyll_gb_model_metadata.pkl'")

    # Example of how to load the model later
    print("\nTo load this model in the future, use:")
    print("with open('chlorophyll_gb_model.pkl', 'rb') as file:")
    print("    loaded_model = pickle.load(file)")
    print("predictions = loaded_model.predict(X_new)")


# ============= MAIN FUNCTION =============
def main():
    """
    Main function to run the entire model training pipeline
    """
    # Load data
    file_path = 'merged_stations.xlsx'
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

    if abs(skewness) > 1:
        print(f"Target is skewed (skewness={skewness}). Applying log transformation.")
        # Add small constant to handle zeros
        df[f'log_{target}'] = np.log1p(df[target])
        use_log = True
        log_target = f'log_{target}'
    else:
        use_log = False
        log_target = target

    # Feature engineering
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

    # Double-check that X contains only numeric data
    print("X data types:")
    print(X.dtypes.value_counts())

    # Setup time series cross-validation
    print("\nSetting up time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    # Date-based train-test split
    if 'Date' in df.columns:
        # Convert to datetime if not already
        df['Date'] = pd.to_datetime(df['Date'])

        # Get unique dates and sort
        unique_dates = df['Date'].sort_values().unique()

        # Calculate the split point - use the 70% mark of unique dates
        split_idx = int(len(unique_dates) * 0.7)
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
        train_size = int(len(df) * 0.7)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        y_log_train, y_log_test = y_log.iloc[:train_size], y_log.iloc[train_size:]

    print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

    # Feature selection
    selected_features, feature_importance = select_features(X_train, y_log_train if use_log else y_train)

    # Train models
    best_rf, gb_pipeline, rf_random, gb_random = train_models(
        X_train, y_log_train if use_log else y_train, selected_features, tscv
    )

    # Evaluate models
    rf_pred, gb_pred, rf_r2, gb_r2, rf_rmse, gb_rmse = evaluate_models(
        best_rf, gb_pipeline, X_train, X_test, y_train, y_test,
        selected_features, use_log
    )

    # Analyze feature importance
    rf_importances, gb_importances = analyze_feature_importance(best_rf, gb_pipeline, selected_features)

    # Plot learning curves
    plot_learning_curve(
        best_rf, X_train[selected_features], y_train if not use_log else y_log_train,
        "Learning Curve - Random Forest", cv=tscv, n_jobs=-1
    )

    plot_learning_curve(
        gb_pipeline, X_train[selected_features], y_train if not use_log else y_log_train,
        "Learning Curve - Gradient Boosting", cv=tscv, n_jobs=-1
    )

    # Save final models and metadata
    save_models(best_rf, gb_pipeline, selected_features, use_log, rf_r2, gb_r2, rf_rmse, gb_rmse, gb_importances)

    # Print final performance summary
    print("\n===== SUMMARY OF RESULTS =====")
    print(f"Tuned Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}")
    print(f"Tuned Gradient Boosting: R² = {gb_r2:.4f}, RMSE = {gb_rmse:.4f}")

    # Return model data for potential use in other scripts
    return {
        'best_rf': best_rf,
        'gb_pipeline': gb_pipeline,
        'selected_features': selected_features,
        'use_log': use_log,
        'rf_importances': rf_importances,
        'gb_importances': gb_importances,
        'tscv': tscv,
        'X': X,
        'y': y
    }


if __name__ == "__main__":
    main()