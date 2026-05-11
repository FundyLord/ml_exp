import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.random.rand(100,1)*10
y = 5 + 3*X.flatten() + np.random.randn(100)*2

df = pd.DataFrame({
    'X' : X.flatten(),
    'y' : y
})

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE = ", rmse)

plt.figure(figsize=(6,8))
plt.scatter(X_test, y_test, color='blue', label='actual points')
plt.plot(X_test, y_pred, linewidth=2, color='red', label='Regression Line')
plt.legend()
plt.title("Simple Linear Regression")
plt.show()