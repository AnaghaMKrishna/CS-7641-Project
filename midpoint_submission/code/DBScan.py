import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.manifold import TSNE

download_dir = os.path.expanduser('~/Downloads')
feature_dataset = os.path.join(download_dir, 'features.csv')
features_df = pd.read_csv(feature_dataset)

pca = PCA(n_components=1000)
data_with_pca = pca.fit_transform(features_df)

explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("explained variance: " + str(explained_variance[len(explained_variance) - 1]))

k_best_model = SelectKBest(score_func=f_regression, k=10)
k_best_features = k_best_model.fit(data_with_pca, data_with_pca[:, 0])
range_nums = np.arange(1000)
k_best_data_indices = range_nums[k_best_features.get_support()]
feature_selected_data = data_with_pca[:, k_best_data_indices]
k_nearest_neighbor_data = resample(feature_selected_data, n_samples=min(len(feature_selected_data), 1000), random_state=42)

nearest_neighbor_fit = NearestNeighbors(n_neighbors=4).fit(k_nearest_neighbor_data)
kneighbors = nearest_neighbor_fit.kneighbors(k_nearest_neighbor_data)
dists = np.sort(kneighbors[0], axis=0)[:, 1]


plt.figure()
plt.plot(dists)
plt.axhline(y=25, color='r', linestyle='--', label='Selected epsilon=25')
plt.xlabel('Points')
plt.ylabel('4th nearest neighbor distance')
plt.title('Elbow Graph')
plt.legend()
plt.savefig('DBScan Epsilon Elbow Graph')
plt.close()

epsilon = 25
sil_score_best = -2**31
min_points_best = -1

for i in range(1, 25):
    dbscan_model = DBSCAN(eps=epsilon, min_samples=i)
    predictions = dbscan_model.fit_predict(feature_selected_data)
    predictions_unique = np.unique(predictions)
    if(len(predictions_unique) > 1):
        if(len(predictions_unique) == 2 and -1 in predictions_unique):
            continue
        mask = (predictions != -1)
        mask_sum = np.sum(mask)
        if mask_sum > 1:
            sil_score = silhouette_score(feature_selected_data[mask], predictions[mask])
            if sil_score > sil_score_best:
                sil_score_best = sil_score
                min_points_best = i
                print("New Best Silhouette Score: " + str(sil_score_best) + ", minPoints = " + str(min_points_best))

final_dbscan_model = DBSCAN(eps=epsilon, min_samples=min_points_best)
predictions = final_dbscan_model.fit_predict(feature_selected_data)
predictions_unique = np.unique(predictions)
final_clusters = len(predictions_unique)
if(-1 in predictions_unique):
    final_clusters = len(predictions_unique) - 1

print("Output:\n")
print("Final Epsilon: " + str(epsilon))
print("Final Clusters: " + str(final_clusters))
print("Final Silhouette Score: " + str(sil_score_best))

tsne_model = TSNE(n_components=2, random_state=42)
tsne_transformed_data = tsne_model.fit_transform(feature_selected_data)
x_1 = tsne_transformed_data[:, 0]
x_2 = tsne_transformed_data[:, 1]

seperation_constant = 0.6
scatter = plt.scatter(x_1, x_2, c=predictions, cmap='viridis', alpha=seperation_constant)
plt.colorbar(scatter, label='Cluster')
plt.title('DBSCAN Clustering Results (t-SNE Visualization)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('DBScan_Cluster_Visualization.png')
plt.close()

unique_predictions = np.unique(predictions)
for i in unique_predictions:
    if i != -1:
        print("Cluster " + str(i) + ": " + str(np.sum(predictions == i)) + " points")
        continue 
    print("Noise points: " + str(np.sum(predictions == i)))
        





