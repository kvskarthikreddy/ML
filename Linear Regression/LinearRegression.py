import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Linear Regression/delaney_solubility_with_descriptors.csv")

# Target variable and features
y = df['logS']
X = df.drop('logS', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_lr_train_pred = lr.predict(X_train)
y_lr_test_pred = lr.predict(X_test)

# Evaluate the model performance
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

# Print the evaluation results
print('LR MSE (Train): ', lr_train_mse)
print('LR R2 (Train): ', lr_train_r2)
print('LR MSE (Test): ', lr_test_mse)
print('LR R2 (Test): ', lr_test_r2)

# Store the results in a DataFrame for better presentation
lr_results = pd.DataFrame({
    'Method': ['Linear Regression'],
    'Training MSE': [lr_train_mse],
    'Training R2': [lr_train_r2],
    'Test MSE': [lr_test_mse],
    'Test R2': [lr_test_r2]
})

print(lr_results)

# Plotting the results
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_lr_train_pred, color="#7CAE00", alpha=0.3, label="Training data")
plt.xlabel('Experimental LogS')
plt.ylabel('Predicted LogS')

# Fit and plot a line to show the relationship between the observed and predicted values
z = np.polyfit(y_train, y_lr_train_pred, 1)  # Linear fit
p = np.poly1d(z)  # Polynomial object for the fit line

plt.plot(y_train, p(y_train), color="#F8766D", label="Fitted line")

# Display the legend
plt.legend()

# Show the plot
plt.show()
