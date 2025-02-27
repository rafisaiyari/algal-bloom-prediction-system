from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('CSV\\water_quality_dataset - Station_4_CentralB.csv')
print(data.head())
print(data.info())
print(data.describe())


# replace NaN
data.fillna(-1, inplace=True)

# data visualization of csv

# plot 1
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Nitrate'], label='Nitrate', color='blue')
plt.plot(data['Date'], data['Phosphate'], label='Phosphate', color='red')
plt.title('Nitrate vs Phosphate')
plt.legend()
#plt.show()

# plot 2
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Chlorophyll-a'], label='Chlorophyll-a', color='green')
plt.title('Chlorophyll-a over time')
#plt.show()

numeric_data = data.select_dtypes(include=['float64'])

# plot 3 - check for correlations between features
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Features Correlation Heatmap')
#plt.show()

# convert object to date
data['Date'] = pd.to_datetime(data['Date'])

prediction = data.loc[
    (data['Date'] > datetime(2016, 2, 1)) &
    (data['Date'] < datetime(2022, 12, 1))
]

plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Chlorophyll-a'], color='blue')
plt.xlabel('Date')
plt.ylabel('Chlorophyll-a')
plt.title('Chlorophyll-a over time')

chlorophyll_a = data.filter(['Chlorophyll-a'])

dataset = chlorophyll_a.values  #convert to numpy array

training_data_len = int(np.ceil(len(dataset) * 0.80))  #use for training data

# Preprocessing Stages
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[0:training_data_len]  # 80% of data

X_train, y_train = [], []

#create a sliding window
for i in range(30, len(training_data)):
    X_train.append(training_data[i-30:i, 0])
    y_train.append(training_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build model
model = keras.models.Sequential()

# masking
model.add(keras.layers.Masking(mask_value=-1, input_shape=(X_train.shape[1], 1)))

# First layer
model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# Second layer
model.add(keras.layers.LSTM(units=50, return_sequences=False))

# 3rd layer (dense) braincells/neurons to help with the correlation of features
model.add(keras.layers.Dense(100, activation='relu'))

# 4th layer dropout
model.add(keras.layers.Dropout(0.1))

# 5th layer (output)
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer='adam',
              loss='mae',
              metrics=[keras.metrics.RootMeanSquaredError()])

training = model.fit(X_train, y_train, epochs=50, batch_size=2, verbose=1)
#prep the test data
test_data = scaled_data[training_data_len - 30:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(30, len(test_data)):
    X_test.append(test_data[i-30:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# make a prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


# Plotting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12, 8))
plt.plot(train['Date'], train['Chlorophyll-a'], label="Train Actual", color='blue')
plt.plot(test['Date'], test['Chlorophyll-a'], label="Test Actual", color='green')
plt.plot(test['Date'], test['Predictions'], label="Test Actual", color='red')
plt.title('Actual vs Predictions')
plt.xlabel('Date')
plt.ylabel('Chlorophyll-a')
plt.legend()
plt.show()