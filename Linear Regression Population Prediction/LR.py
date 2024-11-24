import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set seed for reproducibility
np.random.seed(0)

# Number of countries and years
num_countries = 10
num_years = 20

# Generate years and countries
years = np.arange(2000, 2000 + num_years)
countries = [f'Country_{i+1}' for i in range(num_countries)]

# Create a DataFrame with random population data
data = np.random.randint(1_000_000, 100_000_000, size=(num_years, num_countries))
df = pd.DataFrame(data, columns=countries, index=years)

print("Historical Population Data:")
print(df.head())

# Prepare the data
df_reset = df.reset_index()
df_melted = df_reset.melt(id_vars='index', var_name='Country', value_name='Population')
df_melted.rename(columns={'index': 'Year'}, inplace=True)
df_melted['Lag_1'] = df_melted.groupby('Country')['Population'].shift(1)
df_melted = df_melted.dropna()

print("\nPrepared Data:")
print(df_melted.head())

# Features and target
X = df_melted[['Lag_1']]
y = df_melted['Population']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

# Predict future population
future_year = df.index[-1] + 1
last_population = df.iloc[-1, :].values

# Predict future populations
future_populations = model.predict(last_population.reshape(-1, 1))

# Create a DataFrame for future predictions
future_df = pd.DataFrame(future_populations, index=countries, columns=[future_year])
print("\nPredicted Future Population for", future_year)
print(future_df)