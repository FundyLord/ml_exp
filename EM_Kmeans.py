import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score


# =========================
# Generate Clustered Dataset
# =========================

X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)


# =========================
# Generate Noise Points
# =========================

np.random.seed(42)

noise_points = np.random.uniform(
    low=-15,
    high=15,
    size=(200, 2)
)

noise_labels = np.full(200, -1)


# =========================
# Combine Data + Noise
# =========================

X_final = np.vstack((X, noise_points))

y_final = np.concatenate((y, noise_labels))


# =========================
# Create DataFrame
# =========================

df = pd.DataFrame(
    X_final,
    columns=['Feature1', 'Feature2']
)

df['TrueCluster'] = y_final

print(df.head())


# =========================
# Original Dataset Plot
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    df['Feature1'],
    df['Feature2'],
    color='yellow',
    edgecolors='black'
)

plt.title("Dataset with Noise Points")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.show()


# =========================
# Elbow Method (KMeans Only)
# =========================

k_values = np.arange(1, 11)

inertias = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    kmeans.fit(X_final)

    inertias.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertias,
    marker='o'
)

plt.xlabel("Number of Clusters (K)")

plt.ylabel("Inertia")

plt.title("Elbow Method")

plt.grid(True)

plt.show()


# =========================
# Choose Model
# =========================

# Uncomment ONE model

model = KMeans(
    n_clusters=3,
    random_state=42
)

# model = GaussianMixture(
#     n_components=3,
#     random_state=42
# )


# =========================
# Train Model
# =========================

if isinstance(model, KMeans):

    model.fit(X_final)

    labels = model.labels_

    centers = model.cluster_centers_

else:

    model.fit(X_final)

    labels = model.predict(X_final)

    centers = model.means_


# =========================
# Print Results
# =========================

print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)


# =========================
# Silhouette Score
# =========================

silhouette = silhouette_score(
    X_final,
    labels
)

print("\nSilhouette Score:", silhouette)


# =========================
# Final Visualization
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_final[:, 0],
    X_final[:, 1],
    c=labels,
    cmap='viridis',
    s=50
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color='red',
    marker='X',
    s=250,
    label='Centers'
)

plt.title("Clustering Visualization with Noise")

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.legend()

plt.show()