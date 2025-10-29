<div align="center">

# 🔬 BloomSentry

## Algal Bloom Prediction System for Laguna Lake, Philippines

[![Status](https://img.shields.io/badge/Status-Production%20Ready-blue)](README.md)
[![Model](https://img.shields.io/badge/Model-GBR%2C%20RF%20%2B%20BayesSearchCV-orange)](#-prediction--modeling)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Deployment](https://img.shields.io/badge/Deployment-Python%20IDE-informational)](#-tech-stack--requirements)

<br>

A data-driven system leveraging **Gradient Boosting Regression (GBR)** to provide **monthly predictions** of algal bloom indicators. Designed for proactive monitoring and informed decision-making by water quality managers and stakeholders in Laguna Lake.

-----

## 📖 Overview

The core objective of BloomSentry is to assess the severity and occurrence of algal blooms by forecasting key indicators. It integrates two critical datasets—**water quality (LLDA)** and **meteorological (PAGASA)**—to power a fine-tuned Machine Learning model, resulting in a system that transforms complex environmental data into actionable insights and visual representations.

### Why BloomSentry?

  - **🧠 Gradient Boosting Regression Core**: Uses an optimized GBR model with **BayesSearchCV** and **Huber loss** for robust, high-accuracy, monthly prediction of bloom indicators.
  - **🇵🇭 Region-Specific Data**: Integrates authoritative datasets from the **Laguna Lake Development Authority (LLDA)** and **PAGASA** for local relevance and precision.
  - **📊 Prediction & Monitoring Tool**: Forecasts vital bloom indicators (**Chlorophyll-a, Phytoplankton Count, Nitrogen, Phosphate**) and visualizes the results on a geospatial heatmap.
  - **🔒 Secure Access**: Implements a secure login and signup system for user authentication and managing access control.

-----

## ✨ Features

### Prediction & Modeling

  - **📅 Monthly Forecasting**: Provides a predictive outlook on the occurrence and severity of algal bloom indicators.
  - **⚠️ Indicator Forecasting**: Predicts crucial parameters: **Chlorophyll-a concentration**, **Phytoplankton Count**, **Nitrogen levels**, and **Phosphate concentration**.
  - **📍 Bloom Location Prediction**: Utilizes station data, wind speed, and wind direction to predict the potential location/distribution of a bloom.
  - **⚙️ Optimized Model**: Employs a **Gradient Boosting Regressor (GBR)** model whose hyperparameters are fine-tuned using **BayesSearchCV** for peak performance.

### Data & Visualization

  - **🗺️ Spatial Heatmap Visualization**: Overlays a color-coded heatmap on a map of Laguna Lake to indicate the **intensity and distribution** of predicted bloom occurrences.
  - **📜 Historical Data Visualization**: Presents long-term trends using **bar graphs and line graphs** to illustrate temporal fluctuations and seasonal patterns of water quality.
  - **🚦 DENR Classification Integration**: Classifies water quality parameters using a **color-coded system** based on compliance with **DENR Administrative Order No. 2016-08** regulatory standards.

-----

## 🛠️ Tech Stack & Requirements

### Machine Learning Core

  - **Primary Model**: **Gradient Boosting Regression (GBR)**
  - **Optimization**: **BayesSearchCV** for hyperparameter tuning.
  - **Loss Function**: **Huber Loss** to reduce sensitivity to outliers in the time-series data.
  - **Libraries**: **Scikit-learn**, **Scikit-optimize**, **NumPy**, **Pandas**, **Matplotlib**.

### Data Sources

  - **Water Quality**: **LLDA Water Quality Monitoring Dataset** (Nitrate, Phosphate, Chlorophyll-a, Temperature, DO, etc.)
  - **Meteorological**: **PAGASA Weather Data** (Wind Speed/Direction, Solar Radiation, Rainfall).

### Infrastructure

  - **Development Environment**: Python **IDE** (e.g., Spyder or PyCharm).
  - **Computational Requirement**: Personal computer or laptop with **sufficient processing power** for ML algorithm execution.
  - **Authentication**: Secure **Login and Signup system** for user access control.

-----

## 🚀 Getting Started

### Prerequisites

You will need a personal computer/laptop with adequate processing power and the following installed:

  - Python (v3.x)
  - Python IDE (e.g., Spyder or PyCharm)
  - The required ML and data science libraries.

### Installation & Setup

1.  **Clone the repository**

    ```bash
    git clone https://github.com/rafisaiyari/algal-bloom-prediction-system.git
    cd algal-bloom-prediction-system
    ```

2.  **Install Python Libraries**

    ```bash
    pip install scikit-learn scikit-optimize numpy pandas matplotlib
    ```

3.  **Data Acquisition**
    Obtain and format the **LLDA** and **PAGASA** datasets (expected to be in an Excel file format for initial processing). Place the data file in the designated project folder.

4.  **Run the System**
    Launch your Python IDE, open the main script, and run the model training and prediction pipeline.

-----

## ⚙️ Data Pipeline & Modeling

### Data Processing Steps

| Step | Technique | Purpose |
| :--- | :--- | :--- |
| **Data Cleaning** | Linear Interpolation, Forward/Backward Filling, Median Imputation | Handle inconsistencies and missing values in the time-series data. |
| **Data Transformation** | **RobustScaler**, Log Transformation | Normalize features to reduce outlier influence; normalize skewed Chl-a distribution. |
| **Feature Engineering** | Lagged Features, Rolling Statistics, Rate-of-Change Metrics | Capture crucial **temporal patterns** and rapid changes in water quality conditions. |
| **Data Profiling** | Time-based 80/20 Train/Test Split | Ensure the model is evaluated on unseen future data. |
| **Feature Selection** | **RandomForestRegressor** | Identify and select the **top 20%** of the most important predictors. |
| **Model Optimization** | **BayesSearchCV** | Systematically fine-tune GBR hyperparameters for peak accuracy and efficiency. |

### Prediction Workflow

1.  Read and clean the LLDA and PAGASA datasets.
2.  Perform feature engineering to create new time-series features.
3.  Train the **BayesSearchCV-optimized GBR model** using **Time Series Cross-Validation**.
4.  Generate monthly predictions for the target indicators (Chl-a, etc.).
5.  Visualize the predictions as a **Spatial Heatmap** and integrate with historical trends.

-----

## 👥 Authors

- **Rafi Saiyari** - *Initial work* - [@rafisaiyari](https://github.com/rafisaiyari)
- **Matt Rias** - [https://github.com/mattrias](https://github.com/mattrias)
- **Benjamin Africano** - [https://github.com/FBAfricano](https://github.com/FBAfricano)
- **Beau Sison** - [https://github.com/BeauSison](https://github.com/BeauSison)

-----

\<div align="center"\>

**[⬆ back to top](#bloomsentry-algal-bloom-prediction-system)**

A predictive tool for sustainable water management.

\</div\>
