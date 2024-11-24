import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Random Forest/delaney_solubility_with_descriptors.csv")
y = df['logS']
X = df.drop('logS', axis=1)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(max_depth=5, random_state=100)  # You may experiment with max_depth
rf.fit(X_train, y_train)

# Predictions
y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

# Evaluate performance (MSE and R2)
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

# Display the results
rf_results = pd.DataFrame([['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]])
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print(rf_results)

# Visualize predictions vs actual values for training data
plt.figure(figsize=(5, 5))
plt.scatter(x=y_train, y=y_rf_train_pred, c="#7CAE00", alpha=0.3)

# Fit a line for the scatter plot (linear regression line)
z = np.polyfit(y_train, y_rf_train_pred, 1)
p = np.poly1d(z)

# Plot the regression line
plt.plot(y_train, p(y_train), '#F8766D')

# Add labels and title
plt.title('Random Forest Regression: Actual vs Predicted LogS (Training Set)')
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')
plt.show()

# Feature importances - Understanding which features the model deems important
feature_importances = rf.feature_importances_
features = X.columns

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display feature importances
print("Feature Importances:")
print(feature_importance_df)
