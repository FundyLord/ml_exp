import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = load_iris()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
print(df.head())

# model = PCA(n_components=2)

# model = TruncatedSVD(n_components=2)
# model = LinearDiscriminantAnalysis(n_components=2)

if isinstance(model, LinearDiscriminantAnalysis):
    X_new = model.fit_transform(X, y)
else:
    X_new = model.fit_transform(X)

print("Transformed Data:")
print(X_new[:5])

plt.figure(figsize=(8,6))
plt.scatter(X_new[:,0], X_new[:,1], c=y, cmap='viridis')
plt.title("Universal PCA / SVD / LDA Template")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()