import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from matplotlib.pyplot import figure
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X,y = make_regression(
    n_samples=100
    , n_features=1
    , noise=20
    , random_state=42
)

df = pd.DataFrame({
    "feature": X.flatten(),
    "target": y
})

print(df)

print(df.describe())

print(df.info())


X_train, X_test, y_train, y_test= train_test_split(df[["feature"]], df["target"] , test_size=0.2, random_state=42)
linear_regressor  = LinearRegression()
linear_regressor.fit(X = X_train, y = y_train , sample_weight=None)

y_pred = linear_regressor.predict(X_test)
print(y_pred)

w = linear_regressor.coef_
b = linear_regressor.intercept_
print("w:",w)
print("b:",b)

from sklearn.metrics import mean_squared_error , root_mean_squared_error,mean_absolute_error, r2_score

mse = mean_squared_error(y_test , y_pred)
mae = mean_absolute_error(y_test , y_pred)
rmse = root_mean_squared_error(y_test , y_pred)
r2_score = r2_score(y_test , y_pred)

print("MSE:",mse)
print("MAE:",mae)
print("RMSE:",rmse)
print("R2:",r2_score)

plt.figure(figsize=(6,5))
plt.scatter(X, y)
X_sorted = np.sort(X)
y_line = linear_regressor.predict(X_sorted)
plt.plot(X_sorted, y_line, 'r')
plt.xlabel("feature")
plt.ylabel("target")
plt.title("Regression Line")
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, marker="x")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()



