<div align="center">

# 🔬 BloomSentry

## Algal Bloom Prediction System for Laguna Lake, Philippines

[![Status](https://img.shields.io/badge/Status-Production%20Ready-blue)](README.md)
[![Model](https://img.shields.io/badge/Model-GBR%2C%20RF%20%2B%20BayesSearchCV-orange)](#-prediction--modeling)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<br>

A data-driven system leveraging **Gradient Boosting Regression (GBR)** to provide **monthly predictions** of algal bloom indicators. Designed for proactive monitoring and informed decision-making by water quality managers and stakeholders in Laguna Lake.

-----

[📖 Overview](#-overview) • [✨ Features](#-features) • [🛠️ Tech Stack](#-tech-stack--requirements) • [🚀 Getting Started](#-getting-started) • [⚙️ Data Pipeline](#-data-pipeline--modeling)

</div>

---

## 📖 Overview

The core objective of BloomSentry is to assess the severity and occurrence of algal blooms by forecasting key indicators. It integrates two critical datasets—**water quality (LLDA)** and **meteorological (PAGASA)**—to power a fine-tuned Machine Learning model, transforming complex environmental data into actionable insights and visual representations.

### Key Advantages

* **🧠 Optimized ML Core**: Uses an optimized **Gradient Boosting Regressor (GBR)** model with **BayesSearchCV** and **Huber loss** for robust, high-accuracy, monthly prediction of bloom indicators.
* **🇵🇭 Region-Specific Authority**: Integrates authoritative, local datasets from the **Laguna Lake Development Authority (LLDA)** and **PAGASA** for maximum relevance and precision.
* **📊 Actionable Intelligence**: Forecasts vital bloom indicators (**Chlorophyll-a, Phytoplankton Count, Nitrogen, Phosphate**) and visualizes results on a **geospatial heatmap**.
* **🔒 Secure & Controlled Access**: Implements a secure login and signup system for user authentication and managing access control.

---

## ✨ Features

### Prediction & Modeling
| Feature | Description |
| :--- | :--- |
| **📅 Monthly Forecasting** | Provides a forward-looking predictive outlook on the occurrence and severity of algal bloom indicators. |
| **⚠️ Indicator Prediction** | Forecasts crucial parameters: **Chlorophyll-a concentration**, **Phytoplankton Count**, **Nitrogen levels**, and **Phosphate concentration**. |
| **📍 Bloom Location** | Utilizes station data, wind speed, and wind direction to predict the potential distribution of a bloom. |
| **⚙️ Optimized Model** | Employs a **Gradient Boosting Regressor** fine-tuned using **BayesSearchCV** for peak performance. |

### Data & Visualization
* **🗺️ Spatial Heatmap Visualization**: Overlays a color-coded heatmap on a map of Laguna Lake to indicate the predicted **intensity and distribution** of bloom occurrences.
* **📜 Historical Trend Analysis**: Presents long-term data using **bar graphs and line graphs** to illustrate temporal fluctuations and seasonal patterns of water quality.
* **🚦 Regulatory Classification**: Classifies water quality compliance using a **color-coded system** based on **DENR Administrative Order No. 2016-08** regulatory standards.

---

## 🛠️ Tech Stack & Requirements

### Machine Learning Core
* **Primary Model**: **Gradient Boosting Regression (GBR)**
* **Optimization**: **BayesSearchCV** (for hyperparameter tuning)
* **Loss Function**: **Huber Loss** (to reduce sensitivity to outliers)
* **Libraries**: `Scikit-learn`, `Scikit-optimize`, `NumPy`, `Pandas`, `Matplotlib`

### Data Sources
* **Water Quality**: **LLDA Water Quality Monitoring Dataset** (Chlorophyll-a, Temperature, DO, Nutrients, etc.)
* **Meteorological**: **PAGASA Weather Data** (Wind Speed/Direction, Solar Radiation, Rainfall)

### Infrastructure
* **Environment**: Python **IDE** (e.g., Spyder or PyCharm).
* **Requirements**: Personal computer/laptop with **sufficient processing power** for ML execution.
* **Authentication**: Secure **Login and Signup system** for user access control.

---

## 🚀 Getting Started

### Prerequisites

You will need a personal computer/laptop with adequate processing power, **Python (v3.x)**, a Python IDE, and the required ML libraries installed.

### Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/rafisaiyari/algal-bloom-prediction-system.git](https://github.com/rafisaiyari/algal-bloom-prediction-system.git)
    cd algal-bloom-prediction-system
    ```

2.  **Install Python Libraries**
    ```bash
    pip install scikit-learn scikit-optimize numpy pandas matplotlib
    ```

3.  **Data Acquisition**
    Obtain the **LLDA** and **PAGASA** datasets (expected in an Excel format) and place the data file in the designated project folder for processing.

4.  **Run the System**
    Launch your Python IDE, open the main script, and execute the model training and prediction pipeline.

---

## ⚙️ Data Pipeline & Modeling

This pipeline transforms raw environmental readings into reliable forecasts.

| Step | Technique | Purpose |
| :--- | :--- | :--- |
| **1. Data Cleaning** | Linear Interpolation, Filling, Median Imputation | Handles inconsistencies and **missing values** in time-series data. |
| **2. Data Transformation** | **RobustScaler**, Log Transformation | Normalizes features and reduces the influence of **outliers** in Chl-a distribution. |
| **3. Feature Engineering** | Lagged Features, Rolling Statistics, Rate-of-Change | Captures crucial **temporal patterns** and rapid changes in water conditions. |
| **4. Data Profiling** | Time-based 80/20 Train/Test Split | Ensures the model is robustly evaluated on **unseen future data**. |
| **5. Feature Selection** | **RandomForestRegressor Importance** | Identifies and selects the **top 20%** of the most important predictors. |
| **6. Model Optimization** | **BayesSearchCV** | Systematically fine-tunes GBR hyperparameters for **peak accuracy and efficiency**. |

### Prediction Workflow
1.  Read and clean the LLDA and PAGASA datasets.
2.  Perform feature engineering to create new time-series features.
3.  Train the **BayesSearchCV-optimized GBR model** using **Time Series Cross-Validation**.
4.  Generate monthly predictions for the target indicators (Chl-a, etc.).
5.  Visualize the predictions as a **Spatial Heatmap** and integrate with historical trends.

---

## 👥 Authors

* **Rafi Saiyari** - *Initial work* - [@rafisaiyari](https://github.com/rafisaiyari)
* **Matt Rias** - [https://github.com/mattrias](https://github.com/mattrias)
* **Benjamin Africano** - [https://github.com/FBAfricano](https://github.com/FBAfricano)
* **Beau Sison** - [https://github.com/BeauSison](https://github.com/BeauSison)

---

<div align="center">

**[⬆ Back to Top](#bloomsentry-algal-bloom-prediction-system)**

A predictive tool for sustainable water management.

</div>
