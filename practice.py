import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Load the dataset
data = pd.read_csv('C:/Users/Shree/Desktop/dataset_tk.csv')

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')  # Adjust format if necessary

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Create date features for the training set
features = data.drop(columns=['Punjab'])  # Features for regression
features['Year'] = features.index.year
features['Month'] = features.index.month
features['Day'] = features.index.day

# Use Punjab's energy consumption as the target variable (y)
target = data['Punjab']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and fit the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate model accuracy metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

# Print accuracy metrics
print(f'RMSE: {rmse}')
print(f'MAPE: {mape * 100:.2f}%')
print(f'R-squared: {r_squared}')

# Plotting the results
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Diagonal line
plt.title('Random Forest Predictions vs Actual')
plt.xlabel('Actual Energy Consumption')
plt.ylabel('Predicted Energy Consumption')
plt.grid()
plt.show()

# Create future dates for 30 days
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

# Prepare future features with the same columns as the original features
future_features = pd.DataFrame(index=future_dates)

# Add date features for the future predictions
future_features['Year'] = future_features.index.year
future_features['Month'] = future_features.index.month
future_features['Day'] = future_features.index.day

# Initialize other state features to 0 (or another reasonable value)
for column in features.columns:
    if column not in ['Year', 'Month', 'Day']:  # Skip date columns
        future_features[column] = 0  # Initialize other state features to 0

# Ensure future features match the training feature order
future_features = future_features[features.columns]  # Reorder columns to match training data

# Predict future energy consumption using the trained model
future_predictions = rf_model.predict(future_features)

# Print the future predictions
print("Future Predictions for Punjab:")
print(future_predictions)

# Plot the forecasted future values
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Punjab'], label='Actual', color='blue')
plt.plot(future_dates, future_predictions, label='Forecast', linestyle='--', color='orange')
plt.title('Punjab Energy Consumption Forecast using Random Forest')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (MW)')
plt.legend()
plt.grid(True)
plt.show()

# Print accuracy of the predictions
print("Accuracy (R-squared):", r_squared)
