import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
print(df.head())

model = KMeans(n_clusters=3, random_state=42)

# model = GaussianMixture(n_components=3, random_state=42)

if isinstance(model, KMeans):
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_

else:
    model.fit(X)
    labels = model.predict(X)
    centers = model.means_

print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)
plt.scatter(
    centers[:,0],
    centers[:,1],
    color='red',
    marker='X',
    s=200,
    label='Centers'
)

plt.title("Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()