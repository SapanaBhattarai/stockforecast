import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

# Load the data
df = pd.read_csv('aapl_us_d.csv')

# Detect columns automatically
columns = df.columns
print("Detected columns:", columns)

# Ensure column names are consistent
df.columns = df.columns.str.strip().str.lower()

# Check for required columns
required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"'{col}' column not found in the dataset")

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Preprocess data
df['close'] = df['close'].astype(float)
df['volume'] = df['volume'].astype(float)

# Create features and target
X = df[['open', 'high', 'low', 'volume']]
y = df['close']

# Split data into train and test sets
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define and compile the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f"Linear Regression - MAE: {mae}, RMSE: {rmse}")

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='True Values', color='blue')
plt.plot(y_test.index, y_pred, label='Predictions', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig('forecast_plot.png')  # Save the plot as an image file
plt.close()  # Close the plot to avoid displaying it in a non-interactive environment

