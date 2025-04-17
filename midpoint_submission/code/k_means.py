import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# === Load feature and label CSVs ===
features = pd.read_csv("features.csv")
labels = pd.read_csv("labels.csv")

# === Combine them into one DataFrame ===
X = features
y_true = labels['Cancer_Type']  # Use the Cancer_Type column as labels

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Apply PCA to reduce to 2D ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === Fit K-Means model ===
k = len(pd.unique(y_true))  # Number of unique cancer types
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)  # still cluster on full data

# === Evaluate clustering ===
ari_score = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index (ARI): {ari_score:.3f}")
overall_score = silhouette_score(X_scaled, y_pred)
print(f"Overall Silhouette Score: {overall_score:.3f}")

# === Visualize clusters using PCA-reduced data ===
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="viridis", alpha=0.6)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
            pca.transform(kmeans.cluster_centers_)[:, 1],
            s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering (PCA-Reduced to 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === Count how many samples are in each predicted cluster ===
unique_labels, counts = np.unique(y_pred, return_counts=True)

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=unique_labels, y=counts, palette="Blues_d")
plt.xlabel("Cluster Label")
plt.ylabel("Number of Samples")
plt.title("Cluster Membership Count")
plt.show()
