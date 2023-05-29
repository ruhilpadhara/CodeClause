import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
customer_data = pd.read_csv("Mall_Customers.csv")

# Select the relevant columns for segmentation
segmentation_data = customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Perform feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)

# Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(range(1, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

# Choose the optimal number of clusters and perform K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(scaled_data)

# Add the cluster labels to the dataset
customer_data['Cluster'] = kmeans.labels_

# View the segmented customer data
# print(customer_data.head())
