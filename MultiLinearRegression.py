import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score
)


# =========================
# Generate Synthetic Dataset
# =========================

X, y = make_regression(
    n_samples=200,
    n_features=5,          # Multiple Features
    n_informative=5,
    noise=20,
    random_state=42
)


# =========================
# Create DataFrame
# =========================

df = pd.DataFrame({
    "f1": X[:, 0],
    "f2": X[:, 1],
    "f3": X[:, 2],
    "f4": X[:, 3],
    "f5": X[:, 4],
    "target": y
})


# =========================
# Dataset Information
# =========================

print(df.head())

print(df.describe())

print(df.info())


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    df[["f1", "f2", "f3", "f4", "f5"]],
    df["target"],
    test_size=0.2,
    random_state=42
)


# =========================
# Train Linear Regression
# =========================

linear_regressor = LinearRegression()

linear_regressor.fit(
    X=X_train,
    y=y_train
)


# =========================
# Predictions
# =========================

y_pred = linear_regressor.predict(X_test)

print("Predictions:")
print(y_pred)


# =========================
# Model Parameters
# =========================

w = linear_regressor.coef_
b = linear_regressor.intercept_

print("\nWeights (Coefficients):")
print(w)

print("\nBias (Intercept):")
print(b)


# =========================
# Evaluation Metrics
# =========================

mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMSE :", mse)

print("MAE :", mae)

print("RMSE :", rmse)

print("R2 Score :", r2)


# =========================
# Actual vs Predicted Plot
# =========================

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    y_pred,
    marker='x'
)

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r'
)

plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.show()


# =========================
# Feature Correlation Heatmap
# =========================

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")

plt.show()