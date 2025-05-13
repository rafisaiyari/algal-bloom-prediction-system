import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import stats
import warnings
import pickle

warnings.filterwarnings('ignore')


# ============= LOAD AND PREPROCESS DATA =============
def load_and_preprocess_data(file_path):
    """
    Load and preprocess water quality data with stations as categorical features
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

    # Keep Station as a categorical feature instead of one-hot encoding
    if 'Station' in df.columns:
        print("Processing station information...")
        # Check number of unique stations
        num_stations = df['Station'].nunique()
        print(f"Found {num_stations} unique stations: {df['Station'].unique()}")

        # Convert to categorical for more efficient memory usage
        df['Station'] = df['Station'].astype('category')

        # Create station category codes (0, 1, 2, etc.) for visualization
        df['StationCode'] = df['Station'].cat.codes
        print(f"Converted 'Station' to categorical feature with {num_stations} categories")

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

    # Handle outliers in numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    target = 'Chlorophyll-a (ug/L)'

    for col in numeric_columns:
        if col != target and col != 'StationCode':  # Skip target and station code
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

    # Separate categorical and numeric features
    categorical_features = []
    numeric_features = []

    for feature in features:
        if feature in df.columns:
            if pd.api.types.is_categorical_dtype(df[feature]) or feature == 'Station':
                categorical_features.append(feature)
                print(f"Identified categorical feature: {feature}")
            else:
                numeric_features.append(feature)
                print(f"Identified numeric feature: {feature}")

    print(f"Categorical features: {categorical_features}")
    print(f"Numeric features: {numeric_features}")

    # Create lagged features for numeric features only
    for feature in numeric_features + [target]:
        for lag in [1, 2, 3]:
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)

    # Create rolling mean and standard deviation features for numeric features only
    for feature in numeric_features + [target]:
        for window in [3, 7, 14]:
            df[f'{feature}_roll_mean{window}'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_roll_std{window}'] = df[feature].rolling(window=window).std()

    # Create interaction terms between all pairs of numeric features
    print("Creating interaction features...")
    for i, feat1 in enumerate(numeric_features):
        for feat2 in numeric_features[i + 1:]:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f'{feat1.split(" ")[0]}_{feat2.split(" ")[0]}_interact'
                df[interaction_name] = df[feat1] * df[feat2]
                print(f"  - Created {interaction_name}")

    # Create polynomial features for numeric columns only
    print("Creating polynomial features...")
    for feature in numeric_features:
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

        # Encode month as a categorical variable
        df['Month'] = df['Month'].astype('category')

    # Station-specific features - can help capture station-specific patterns
    if 'Station' in df.columns:
        print("Creating station-specific aggregated features...")
        # Group by Station and calculate aggregate statistics
        station_aggs = df.groupby('Station')[target].agg(['mean', 'std']).reset_index()
        station_aggs.columns = ['Station', f'Station_{target}_mean', f'Station_{target}_std']

        # Merge back to main dataframe
        df = pd.merge(df, station_aggs, on='Station', how='left')

        # Create station-target difference (how much a sample deviates from its station average)
        df[f'Station_{target}_diff'] = df[target] - df[f'Station_{target}_mean']

        print(f"  - Created station-specific aggregate features")

    # Drop rows with NaN due to lag/rolling features
    original_rows = len(df)
    df = df.dropna()
    print(f"Dropped {original_rows - len(df)} rows due to NaN values from feature engineering")

    return df


# ============= FEATURE SELECTION =============
def select_features(X_train, y_train, categorical_cols=None, use_log=False):
    """
    Select important features using Random Forest feature importance
    """
    print("\n=== Performing feature selection ===")

    # Create a copy of X_train for feature selection
    X_train_sel = X_train.copy()

    # Encode categorical columns for feature selection
    if categorical_cols:
        print(f"Encoding categorical columns for feature selection: {categorical_cols}")
        # Use OrdinalEncoder for feature selection
        ordinal_encoder = OrdinalEncoder()
        for col in categorical_cols:
            if col in X_train_sel.columns:
                X_train_sel[col] = ordinal_encoder.fit_transform(X_train_sel[[col]])
                print(f"  - Encoded {col} with {X_train_sel[col].nunique()} unique values")

    # Use a more efficient approach than RFECV - Random Forest feature importance
    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    feature_selector.fit(X_train_sel, y_train)

    # Get feature importances
    importances = feature_selector.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': X_train_sel.columns,
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
    num_features = max(int(len(X_train_sel.columns) * 0.3), 20)
    num_features = min(num_features, len(X_train_sel.columns))  # Ensure we don't select more than we have
    selected_features = feature_importance['feature'][:num_features].values
    print(f"Selected {len(selected_features)} out of {len(X_train_sel.columns)} features")

    # Make sure categorical columns are included in selected features
    if categorical_cols:
        for col in categorical_cols:
            if col not in selected_features:
                print(f"Adding categorical feature {col} to selected features")
                selected_features = np.append(selected_features, col)

    return selected_features, feature_importance


# ============= MODEL TRAINING =============
def train_models(X_train, y_train, selected_features, categorical_cols, tscv):
    """
    Train and tune Random Forest and Gradient Boosting models with preprocessing pipeline
    """
    print("\n=== Training and tuning models ===")

    # Separate numerical and categorical columns
    numeric_features = [col for col in selected_features if col not in categorical_cols]
    categorical_features = [col for col in selected_features if col in categorical_cols]

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # Create preprocessing pipelines for different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    # 1. Random Forest Pipeline with Preprocessing
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    # Parameter grid for Random Forest
    rf_param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 15, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    }

    # Use RandomizedSearchCV for efficiency
    print("\nTuning Random Forest hyperparameters...")
    rf_random = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=rf_param_grid,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Fit the random search
    rf_random.fit(X_train, y_train)

    # Best parameters
    print("Best Random Forest parameters found:")
    for param, value in rf_random.best_params_.items():
        print(f"  - {param}: {value}")

    # Train RF with best parameters
    best_rf = rf_random.best_estimator_
    print("\nTraining Random Forest with best parameters complete")

    # 2. Gradient Boosting Pipeline with Preprocessing
    gb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(random_state=42))
    ])

    # Parameter grid for Gradient Boosting
    gb_param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 4, 5, 6],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__subsample': [0.8, 0.9, 1.0]
    }

    # Use RandomizedSearchCV for efficiency
    print("\nTuning Gradient Boosting hyperparameters...")
    gb_random = RandomizedSearchCV(
        estimator=gb_pipeline,
        param_distributions=gb_param_grid,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )

    # Fit the random search
    gb_random.fit(X_train, y_train)

    # Best parameters
    print("Best Gradient Boosting parameters found:")
    for param, value in gb_random.best_params_.items():
        print(f"  - {param}: {value}")

    # Train GB with best parameters
    best_gb = gb_random.best_estimator_
    print("\nTraining Gradient Boosting with best parameters complete")

    return best_rf, best_gb, rf_random, gb_random


# ============= MODEL EVALUATION =============
def evaluate_models(best_rf, best_gb, X_train, X_test, y_train, y_test, use_log=False):
    """
    Evaluate models on training and test datasets
    """
    print("\n=== Evaluating models ===")

    # Calculate R2 on training data for Random Forest
    rf_train_pred = best_rf.predict(X_train)
    if use_log:
        rf_train_pred = np.expm1(rf_train_pred)  # Convert back from log
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))

    # Calculate R2 on training data for Gradient Boosting
    gb_train_pred = best_gb.predict(X_train)
    if use_log:
        gb_train_pred = np.expm1(gb_train_pred)  # Convert back from log
    gb_train_r2 = r2_score(y_train, gb_train_pred)
    gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))

    # Print the training results
    print("\n===== TRAINING DATA PERFORMANCE =====")
    print(f"Random Forest - Train R²: {rf_train_r2:.4f}, Train RMSE: {rf_train_rmse:.4f}")
    print(f"Gradient Boosting - Train R²: {gb_train_r2:.4f}, Train RMSE: {gb_train_rmse:.4f}")

    # Predictions on test data
    rf_pred = best_rf.predict(X_test)
    if use_log:
        rf_pred = np.expm1(rf_pred)  # Convert back from log

    gb_pred = best_gb.predict(X_test)
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
def analyze_feature_importance(best_rf, best_gb):
    """
    Analyze and visualize feature importances for models with preprocessing pipelines
    """
    print("\n=== Analyzing feature importance ===")

    # For models with ColumnTransformer preprocessing, feature importance extraction is more complex
    # We'll extract feature names and importances from the model inside the pipeline

    # Get feature names after preprocessing (this can be complex with OneHotEncoder)
    # For categorical features after one-hot encoding, we need to get the new feature names
    try:
        # Try to get feature names from the pipeline
        # This may vary depending on sklearn version
        rf_preprocessor = best_rf.named_steps['preprocessor']
        gb_preprocessor = best_gb.named_steps['preprocessor']

        # Get the transformed feature names (may need adjustment based on sklearn version)
        num_features = rf_preprocessor.transformers_[0][2]
        cat_features = rf_preprocessor.transformers_[1][2]

        # Get One-hot encoded feature names
        cat_encoder = rf_preprocessor.named_transformers_['cat']
        cat_features_encoded = []
        if hasattr(cat_encoder, 'get_feature_names_out'):  # sklearn 1.0+
            cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
            cat_features_encoded = list(cat_feature_names)
        else:  # sklearn < 1.0
            for i, cat in enumerate(cat_features):
                cat_values = cat_encoder.categories_[i]
                cat_features_encoded.extend([f"{cat}_{val}" for val in cat_values])

        # Combine numeric and encoded categorical feature names
        all_features = list(num_features) + cat_features_encoded

        # Extract importances from RF and GB models
        rf_importances = best_rf.named_steps['model'].feature_importances_
        gb_importances = best_gb.named_steps['model'].feature_importances_

        # Create DataFrames for visualization
        rf_importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': rf_importances
        }).sort_values('importance', ascending=False)

        gb_importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': gb_importances
        }).sort_values('importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(14, 10))
        plt.subplot(1, 2, 1)
        sns.barplot(x='importance', y='feature', data=rf_importance_df.head(15))
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        sns.barplot(x='importance', y='feature', data=gb_importance_df.head(15))
        plt.title('Gradient Boosting Feature Importance')
        plt.tight_layout()

        plt.show()

        return rf_importance_df, gb_importance_df

    except Exception as e:
        # If there's an error extracting preprocessed feature names, use a simpler approach
        print(f"Error extracting feature importances from pipeline: {e}")
        print("Using simplified feature importance analysis...")

        # For pipeline models, plot feature importances directly from the model
        plt.figure(figsize=(14, 10))
        plt.subplot(1, 2, 1)
        importances = best_rf.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.title('Random Forest Feature Importance (indices)')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), indices)
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        importances = best_gb.named_steps['model'].feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.title('Gradient Boosting Feature Importance (indices)')
        plt.bar(range(len(indices)), importances[indices])
        plt.xticks(range(len(indices)), indices)
        plt.tight_layout()

        plt.show()

        # Return simplified importance DataFrames
        rf_importance_df = pd.DataFrame({
            'feature_index': range(len(best_rf.named_steps['model'].feature_importances_)),
            'importance': best_rf.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)

        gb_importance_df = pd.DataFrame({
            'feature_index': range(len(best_gb.named_steps['model'].feature_importances_)),
            'importance': best_gb.named_steps['model'].feature_importances_
        }).sort_values('importance', ascending=False)

        return rf_importance_df, gb_importance_df


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
def save_models(best_rf, best_gb, selected_features, categorical_cols, use_log, rf_r2, gb_r2, rf_rmse, gb_rmse,
                gb_importances):
    """
    Save models, selected features, and metadata to disk
    """
    print("\n=== Saving models to disk ===")

    # Save the best model (usually gradient boosting performs better)
    with open('chlorophyll_gb_model.pkl', 'wb') as file:
        pickle.dump(best_gb, file)
    print("Gradient Boosting pipeline saved as 'chlorophyll_gb_model.pkl'")

    # Save selected features and categorical columns for future reference
    feature_info = {
        'selected_features': selected_features,
        'categorical_columns': categorical_cols
    }
    with open('feature_info.pkl', 'wb') as file:
        pickle.dump(feature_info, file)
    print(f"Feature information saved as 'feature_info.pkl'")

    # Optional: Save model metadata
    model_metadata = {
        'r2_score': gb_r2,
        'rmse': gb_rmse,
        'use_log_transform': use_log,
        'feature_importances': gb_importances.to_dict(),
        'model_parameters': best_gb.named_steps['model'].get_params(),
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


# ============= STATION ANALYSIS =============
def analyze_stations(df, target):
    """
    Analyze station-specific patterns and relationships
    """
    print("\n=== Analyzing station-specific patterns ===")

    if 'Station' not in df.columns:
        print("Station column not found in dataframe")
        return

    # Count samples per station
    station_counts = df['Station'].value_counts()
    print("\nSample count by station:")
    print(station_counts)

    # Calculate target variable statistics by station
    station_stats = df.groupby('Station')[target].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print("\nTarget variable statistics by station:")
    print(station_stats)

    # Plot boxplot of target variable by station
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Station', y=target, data=df)
    plt.title(f"{target} Distribution by Station")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Time series plot by station if Date column exists
    if 'Date' in df.columns:
        plt.figure(figsize=(16, 10))
        for station in df['Station'].unique():
            station_data = df[df['Station'] == station]
            plt.plot(station_data['Date'], station_data[target], label=station, marker='o', linestyle='-')

        plt.title(f"{target} Time Series by Station")
        plt.xlabel("Date")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Calculate correlations between features and target for each station
    print("\nTop correlations with target by station:")
    for station in df['Station'].unique():
        station_data = df[df['Station'] == station]
        numeric_cols = station_data.select_dtypes(include=np.number).columns

        # Skip if there are too few samples
        if len(station_data) < 10:
            print(f"Station {station}: Not enough samples for correlation analysis")
            continue

        # Calculate correlations with target
        corrs = station_data[numeric_cols].corr()[target].sort_values(ascending=False)

        # Print top 5 correlations (excluding self-correlation)
        print(f"\nStation {station} (n={len(station_data)}):")
        print(corrs.drop(target).head(5))

    return station_stats


# ============= STATION EFFECT INTERPRETATION =============
def interpret_station_effects(best_model, df, target, categorical_cols):
    """
    Interpret the station effects captured by the model
    """
    print("\n=== Interpreting Station Effects ===")

    if 'Station' not in categorical_cols or 'Station' not in df.columns:
        print("Station not found as a categorical feature")
        return

    try:
        # Create a reference dataset for prediction with only station varying
        unique_stations = df['Station'].unique()
        num_stations = len(unique_stations)

        print(f"Analyzing effects of {num_stations} unique stations")

        # Identify all required columns for the model
        # We need to check what columns the model expects
        if hasattr(best_model, 'feature_names_in_'):
            required_columns = best_model.feature_names_in_
        elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps.get('model', None),
                                                            'feature_names_in_'):
            # For pipelines, get feature names from the final estimator
            required_columns = best_model.named_steps['model'].feature_names_in_
        else:
            # If we can't determine exact columns, use a heuristic approach
            # This might be needed for older scikit-learn versions
            print("Could not determine exact feature names, using columns from training data")
            # Exclude target and other non-feature columns
            exclude_cols = [target]
            if 'log_' + target in df.columns:
                exclude_cols.append('log_' + target)
            if 'Date' in df.columns:
                exclude_cols.append('Date')

            required_columns = [col for col in df.columns if col not in exclude_cols]

        print(f"Model requires {len(required_columns)} features")

        # Create a baseline dataset using median values for numeric features
        baseline_data = {}

        # First add all required features with default values
        for col in required_columns:
            if col in df.columns:
                if col == 'Station' or col in categorical_cols:
                    continue  # We'll handle categorical columns separately
                elif pd.api.types.is_numeric_dtype(df[col]):
                    baseline_data[col] = [df[col].median()] * num_stations
                else:
                    # For other columns, use the most common value
                    baseline_data[col] = [df[col].mode()[0]] * num_stations
            else:
                print(f"Warning: Required column {col} not found in training data. Using zeros.")
                baseline_data[col] = [0] * num_stations

        # Add Station column with each unique station
        baseline_data['Station'] = unique_stations

        # Add any other categorical columns that might be required
        for col in categorical_cols:
            if col != 'Station' and col in required_columns:
                # Use the most frequent value for other categorical columns
                if col in df.columns:
                    mode_value = df[col].mode()[0]
                    baseline_data[col] = [mode_value] * num_stations
                    print(f"Using {mode_value} for categorical column {col}")
                else:
                    print(f"Warning: Categorical column {col} not found. Using placeholder value.")
                    baseline_data[col] = ["unknown"] * num_stations

        # Check for any missing required columns and add them with zeros or placeholders
        for col in required_columns:
            if col not in baseline_data:
                if col.endswith(('_mean', '_std', '_min', '_max')):
                    # For statistical columns, use zeros
                    baseline_data[col] = [0] * num_stations
                    print(f"Adding placeholder zeros for missing column: {col}")
                else:
                    # For other columns, use a placeholder string (for categorical) or zero (for numeric)
                    baseline_data[col] = ["unknown"] * num_stations
                    print(f"Adding placeholder 'unknown' for missing column: {col}")

        # Create DataFrame
        baseline_df = pd.DataFrame(baseline_data)

        # Ensure categorical columns have the right dtype
        for col in categorical_cols:
            if col in baseline_df.columns:
                baseline_df[col] = baseline_df[col].astype('category')

        # Print the baseline dataframe structure
        print("\nBaseline dataframe structure:")
        print(f"Shape: {baseline_df.shape}")
        print(f"Columns: {baseline_df.columns.tolist()}")

        # Make predictions
        predictions = best_model.predict(baseline_df)

        # Create results DataFrame
        results = pd.DataFrame({
            'Station': unique_stations,
            'Predicted_Chlorophyll': predictions
        })

        # Sort by predicted values
        results = results.sort_values('Predicted_Chlorophyll', ascending=False)

        print("\nStation effect on predictions (all other variables held constant):")
        print(results)

        # Visualize station effects
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Station', y='Predicted_Chlorophyll', data=results)
        plt.title('Station Effect on Chlorophyll-a Predictions')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        # Calculate overall station effect range
        effect_range = results['Predicted_Chlorophyll'].max() - results['Predicted_Chlorophyll'].min()
        effect_percent = (effect_range / df[target].mean()) * 100

        print(f"\nStation effect range: {effect_range:.2f} units")
        print(f"Station effect as percentage of mean target: {effect_percent:.1f}%")

        return results

    except Exception as e:
        print(f"Error interpreting station effects: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============= MAIN FUNCTION =============
def main():
    """
    Main function to run the entire model training pipeline with categorical station features
    """
    # Load data
    file_path = 'merged_stations.xlsx'
    df = load_and_preprocess_data(file_path)

    # Define features and target
    numeric_features = ['pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)',
                        'Inorganic Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Temperature']

    # Add Station as a categorical feature
    categorical_features = []
    if 'Station' in df.columns:
        categorical_features.append('Station')
        print(f"Using Station as a categorical feature")

    # Add Month if available
    if 'Month' in df.columns:
        categorical_features.append('Month')
        print(f"Using Month as a categorical feature")

    # Combine all features
    features = numeric_features + categorical_features

    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    print(f"All features: {features}")

    # Check if all features exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        print(f"Warning: Some features not found. Using {len(available_features)} available features:")
        print(available_features)
        features = available_features

    target = 'Chlorophyll-a (ug/L)'

    # Analyze station-specific patterns
    if 'Station' in df.columns:
        station_stats = analyze_stations(df, target)

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

    # Feature engineering - with separate handling for categorical features
    df = engineer_features(df, features, target)

    # All features including engineered ones
    # Exclude any non-numeric columns, Date, target and log_target
    exclude_cols = ['Date', target, log_target]

    # Keep categorical columns in features list
    all_features = [col for col in df.columns if col not in exclude_cols]

    # Ensure categorical columns are marked as categories
    for col in categorical_features:
        if col in df.columns and col in all_features:
            df[col] = df[col].astype('category')
            print(f"Ensured {col} is properly typed as a category")

    # Verify all features are valid
    for col in all_features:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe. Removing from features list.")
            all_features.remove(col)

    # Print feature list
    print(f"Total features after engineering: {len(all_features)}")

    # Update categorical_features list to include any new categorical columns
    # (like Month if it was created during feature engineering)
    categorical_cols = [col for col in df.columns if
                        pd.api.types.is_categorical_dtype(df[col]) or
                        col in categorical_features]
    print(f"Categorical columns for modeling: {categorical_cols}")

    # Prepare data
    X = df[all_features]
    y = df[target]
    y_log = df[log_target] if use_log else y

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
    selected_features, feature_importance = select_features(
        X_train, y_log_train if use_log else y_train,
        categorical_cols=categorical_cols
    )

    # Train models
    best_rf, best_gb, rf_random, gb_random = train_models(
        X_train, y_log_train if use_log else y_train,
        selected_features, categorical_cols, tscv
    )

    # Evaluate models
    rf_pred, gb_pred, rf_r2, gb_r2, rf_rmse, gb_rmse = evaluate_models(
        best_rf, best_gb, X_train, X_test, y_train, y_test, use_log
    )

    # Analyze feature importance
    rf_importances, gb_importances = analyze_feature_importance(best_rf, best_gb)

    # Plot learning curves
    plot_learning_curve(
        best_rf, X_train, y_train if not use_log else y_log_train,
        "Learning Curve - Random Forest", cv=tscv, n_jobs=-1
    )

    plot_learning_curve(
        best_gb, X_train, y_train if not use_log else y_log_train,
        "Learning Curve - Gradient Boosting", cv=tscv, n_jobs=-1
    )

    # Save final models and metadata
    save_models(best_rf, best_gb, selected_features, categorical_cols,
                use_log, rf_r2, gb_r2, rf_rmse, gb_rmse, gb_importances)

    # Print final performance summary
    print("\n===== SUMMARY OF RESULTS =====")
    print(f"Tuned Random Forest: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.4f}")
    print(f"Tuned Gradient Boosting: R² = {gb_r2:.4f}, RMSE = {gb_rmse:.4f}")

    # Interpret station effects (using the better-performing model)
    best_model = best_gb if gb_r2 > rf_r2 else best_rf
    station_effects = interpret_station_effects(best_model, df, target, categorical_cols)

    # Return model data for potential use in other scripts
    return {
        'best_rf': best_rf,
        'best_gb': best_gb,
        'selected_features': selected_features,
        'categorical_cols': categorical_cols,
        'use_log': use_log,
        'rf_importances': rf_importances,
        'gb_importances': gb_importances,
        'tscv': tscv,
        'X': X,
        'y': y
    }


if __name__ == "__main__":
    main()