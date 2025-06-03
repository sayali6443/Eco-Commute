# Import necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Public_Transport.csv')

df['Metropolitan train'] = df['Metropolitan train'].str.replace(',', '').fillna('0').astype(int)
df['Metropolitan tram'] = df['Metropolitan tram'].str.replace(',', '').fillna('0').astype(int)
df['Metropolitan bus'] = df['Metropolitan bus'].str.replace(',', '').fillna('0').astype(int)
df['Regional train'] = df['Regional train'].str.replace(',', '').fillna('0').astype(int)
df['Regional coach'] = df['Regional coach'].str.replace(',', '').fillna('0').astype(int)
df['Regional bus'] = df['Regional bus'].str.replace(',', '').fillna('0').astype(int)

# Create a new column for total passengers by summing all transport-related columns
df['Total passengers'] = (df['Metropolitan train'] + df['Metropolitan tram'] +
                          df['Metropolitan bus'] + df['Regional train'] +
                          df['Regional coach'] + df['Regional bus'])

# Limit the data
df = df[df['_id'] <= 78]

# Plotting the existing trend 
plt.figure(figsize=(10,6))
plt.plot(df['_id'], df['Total passengers'], marker='o', linestyle='-', color='b')
plt.title('Trend of Total Passengers Using Public Transport (2018 - 2024)')
plt.xlabel('Month')
plt.ylabel('Total Number of Passengers (in Millions)')

plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))
plt.xticks(df['_id'], df['Year'], rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Data for training the model
X = df[['Year']] 
y = df['Total passengers']

# Split data 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the trend for the test set
y_pred = model.predict(X_test)

# Print model performance metrics
print(f"R-squared: {model.score(X_test, y_test)}")

# Now, let's predict the future (2024-2029) with the introduction of an incentive program
# First, simulate the future months
future_months = pd.DataFrame({
    '_id': np.arange(79, 79 + 60),  # Next 60 months- 2024-2029 (5 years)
    'Year': np.repeat(np.arange(2024, 2029), 12)  # Repeat each year 12 times (assuming monthly data)
})

# Predict future public transport usage with the incentive program
# Assume a 1% increase in passengers each year due to incentives
incentive_factor = 1.01  # 1% increase each year

# Use linear regression predictions as baseline
future_predictions = model.predict(future_months[['Year']])

# Apply incentive factor to increase predictions
for i in range(1, len(future_predictions)):
    future_predictions[i] = future_predictions[i - 1] * incentive_factor

# Combine the future months and predictions into a DataFrame for visualization
future_months['Predicted Passengers'] = future_predictions

# Plot both the historical and predicted future trends
plt.figure(figsize=(10,6))

# Plot historical data
plt.plot(df['_id'], df['Total passengers'], marker='o', linestyle='-', color='b', label='Actual Passengers (2018-2024)')

# Plot future predictions (2024-2029)
plt.plot(future_months['_id'], future_months['Predicted Passengers'], marker='o', linestyle='--', color='r', label='Predicted Passengers (2024-2029 with Incentive)')

plt.title('Public Transport Usage: Historical Data and Future Predictions with Incentive Program')
plt.xlabel('Month')
plt.ylabel('Total Number of Passengers (in Millions)')
plt.xticks(list(df['_id']) + list(future_months['_id'].iloc[::12]), list(df['Year']) + list(future_months['Year'].iloc[::12]), rotation=45, ha='right')

plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display the predicted future values
print(future_months[['Year', 'Predicted Passengers']])