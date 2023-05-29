import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
import prince
from gap_statistic import OptimalK
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Load the dataset
customer_data = pd.read_csv("Mall_Customers.csv")

# Gender Visualization
gender_counts = customer_data['Gender'].value_counts()
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=customer_data)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Age Distribution Visualization
plt.figure(figsize=(8, 6))
sns.histplot(x='Age', data=customer_data)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Analysis of Annual Income
plt.figure(figsize=(8, 6))
sns.histplot(x='Annual Income (k$)', data=customer_data)
plt.title('Annual Income Distribution')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Count')
plt.show()

# Analysis of Spending Score
plt.figure(figsize=(8, 6))
sns.histplot(x='Spending Score (1-100)', data=customer_data)
plt.title('Spending Score Distribution')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Count')
plt.show()

# Select relevant columns for clustering
segmentation_data = customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Perform feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)

# K-means Algorithm - Determining Optimal Clusters: Elbow method
inertia = []
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))

# Elbow Curve
plt.plot(range(2, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

# Average Silhouette Method
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Average Silhouette Method')
plt.show()


# K-means Algorithm - Determining Optimal Clusters: Elbow method
kmeans = KMeans(random_state=42)
visualizer = KElbowVisualizer(kmeans, k=(2, 11), metric='distortion', timings=False)

visualizer.fit(scaled_data)
visualizer.show()

# K-means Algorithm - Determining Optimal Clusters: Silhouette method
visualizer = SilhouetteVisualizer(kmeans, metric='euclidean', colormap='viridis', timings=False)

visualizer.fit(scaled_data)
visualizer.show()

# Perform K-means clustering with chosen number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the original dataset
customer_data['Cluster'] = kmeans.labels_

# Visualizing Clustering Results using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
cluster_labels = kmeans.labels_
pca_df = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'Cluster': cluster_labels})

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set1')
plt.title('Clustering Results - PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Visualizing Clustering Results using Prince's PCA
mca = prince.MCA(n_components=2)
mca_result = mca.fit_transform(segmentation_data)
mca_df = pd.DataFrame({'PC1': mca_result.iloc[:, 0], 'PC2': mca_result.iloc[:, 1], 'Cluster': cluster_labels})

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=mca_df, palette='Set1')
plt.title('Clustering Results - MCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()