from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.read_csv('CSV\\water_quality_dataset - Station_4_CentralB.csv')
print(data.head())
print(data.info())
print(data.describe())

# data visualization of csv

# plot 1
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Nitrate'], label='Nitrate', color='blue')
plt.plot(data['Date'], data['Phosphate'], label='Phosphate', color='red')
plt.title('Nitrate vs Phosphate')
plt.legend()
plt.show()

# plot 2
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Chlorophyll-a'], label='Chlorophyll-a', color='green')
plt.title('Chlorophyll-a over time')
plt.show()

numeric_data = data.select_dtypes(include=['float64'])

# plot 3 - check for correlations between features
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Features Correlation Heatmap')
plt.show()