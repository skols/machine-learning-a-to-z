# K-Means clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
from sklearn.cluster import KMeans

# %reset -f

# Importing the mall dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part4_Clustering/Section21_K_Means_Clustering")
dataset = pd.read_csv("Mall_Customers.csv")

# Create a matrix with the values we need for clustering
# Income and Spending Score
X = dataset.iloc[:, 3:].values

# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Use the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10,
                  random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()
# clusters = 5

# Applying k-means to the mall dataset
kmeans=KMeans(n_clusters=5, init="k-means++", max_iter=300, n_init=10,
              random_state=0)
y_kmeans = kmeans.fit_predict(X)

colors = ["red", "blue", "green", "cyan", "magenta"]

# Visualising the clusters
for i in range(0, 5):
    plt.scatter(X[y_kmeans==i, 0], X[y_kmeans==i, 1], s=100, c=colors[i],
            label="Cluster {0}".format(i + 1))
#plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c="red",
#            label="Careful")
#plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c="blue",
#            label="Standard")
#plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c="green",
#            label="Target")
#plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c="cyan",
#            label="Careless")
#plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c="magenta",
#            label="Sensible")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c="yellow", label="Centroids")
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# After seeing the clusters, give them a name
clusters = ["Careful", "Standard", "Target", "Careless", "Sensible"]

for i in range(0, 5):
    plt.scatter(X[y_kmeans==i, 0], X[y_kmeans==i, 1], s=100, c=colors[i],
            label=clusters[i])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c="yellow", label="Centroids")
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
