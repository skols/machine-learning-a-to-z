# Hierarchical clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
# from sklearn.cross_validation import train_test_split  # deprecated
import scipy.cluster.hierarchy as sch  # for dendrogram
from sklearn.cluster import AgglomerativeClustering


# %reset -f

# To see the full array, run the following
np.set_printoptions(threshold=np.nan)

# Importing the mall dataset
os.chdir("C:/Development/Courses/Kirill Eremenko Data Science Courses/\
Machine_Learning_A-Z/Part4_Clustering/Section22_Hierarchical_Clustering")
dataset = pd.read_csv("Mall_Customers.csv")

# Create a matrix with the values we need for clustering
# Income and Spending Score
X = dataset.iloc[:, 3:].values

# Using the dendrogram to find the optimal number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distances")
plt.show()
# 5 clusters

# Fitting hierarchical clustering to the mall dataset
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean",
                             linkage="ward")
y_hc = hc.fit_predict(X)

colors = ["red", "blue", "green", "cyan", "magenta"]

# Visualising the clusters
for i in range(0, 5):
    plt.scatter(X[y_hc==i, 0], X[y_hc==i, 1], s=100, c=colors[i],
            label="Cluster {0}".format(i + 1))
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# After seeing the clusters, give them a name
clusters = ["Careful", "Standard", "Target", "Careless", "Sensible"]

for i in range(0, 5):
    plt.scatter(X[y_hc==i, 0], X[y_hc==i, 1], s=100, c=colors[i],
            label=clusters[i])
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
