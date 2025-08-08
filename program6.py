#(6).Write a program to demonstrate linear regression using an appropriate dataset. 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and select 'MedInc' as feature
X, y = fetch_california_housing(return_X_y=True)
X = X[:, [0]]  # Median Income

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model and predict
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Display results
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Plot
idx = np.argsort(X_test[:, 0])
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test[idx], y_pred[idx], color='red', label='Predicted')
plt.xlabel('Median Income')
plt.ylabel('House Price')
plt.title('Linear Regression: Income vs Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()