import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA  # For dimensionality reduction

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target  # Add the target column for the actual species

# Separate features from the target
features = df.iloc[:, :-1]  

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-Means Clustering
n_clusters = 3  # The number of clusters for K-Means (matching the number of species)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_features)

# Get the cluster labels
labels = kmeans.labels_

# Calculate the silhouette score
silhouette_avg = silhouette_score(scaled_features, labels)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Dimensionality Reduction using PCA (optional for better visualization)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Plotting the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=labels, palette='viridis', s=100)
plt.title('K-Means Clustering of Iris Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.show()
